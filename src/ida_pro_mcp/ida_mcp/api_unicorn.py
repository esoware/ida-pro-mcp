"""Unicorn-based CPU emulation for IDA Pro MCP.

This module bolts a real CPU emulator (the Unicorn Engine) onto the static
database so the agent can *run* code instead of only reading it: decrypt a
string/blob, compute a hash or checksum, resolve an opaque predicate, or step a
VM handler and read back the result.

It is deliberately NOT a debugger. Nothing here attaches to a live process,
touches the OS, or leaves the sandbox. Unicorn is a pure in-memory CPU: memory
is mapped lazily from the IDB, external `call`s never actually execute anything
in a DLL, and the only side effects are inside the emulator's own RAM. That is
why these tools are safe to expose unconditionally (no `@unsafe`, no `@ext`).

Two layers:

  * `unicorn_emulate`  -- one-shot. Create a throwaway context, set up
    registers/memory, run a function to its return (or an address), read back
    registers/memory, discard. This is the tool you usually want.

  * `unicorn_create` + `unicorn_run`/`unicorn_step`/`unicorn_read`/
    `unicorn_write`/`unicorn_map`/`unicorn_state`/`unicorn_reset`/
    `unicorn_destroy`/`unicorn_list` -- a persistent context you keep around
    across calls to build up state, single-step, poke memory, and continue.
    A context lives inside the current IDB worker; it is gone when the worker
    exits.

Memory model
    Segments are mapped *lazily*: a page is pulled from the IDB the first time
    emulation touches it (so a 137 MB binary costs nothing until executed).
    BSS / uninitialised bytes read back as zero. Reads/writes to addresses not
    in any segment get zero-filled scratch pages (stack growth, heap, etc.).

Calls that leave the image
    Real code calls imports / OS functions. Since nothing is mapped there, an
    unhandled external call faults at fetch. You control what happens via
    `stubs` (and `skip_calls`): skip a call and force a return value, or run a
    built-in model of a common libc function (malloc/memcpy/...). See the
    `stubs` docstrings for the grammar.
"""

import time
from typing import Annotated, Any, Optional

import idaapi
import idc
import ida_bytes
import ida_nalt
import ida_segment

from .rpc import tool
from .sync import idasync, tool_timeout, IDAError
from .utils import normalize_list_input, parse_address, read_bytes_bss_safe

# ---------------------------------------------------------------------------
# Optional dependency: keep the whole package importable even if unicorn is
# missing. Every tool checks _require() first and returns a clear error.
# ---------------------------------------------------------------------------
try:
    import unicorn as _U
    from unicorn import x86_const as _X

    _UNI_OK = True
    _UNI_ERR = ""
except Exception as _e:  # pragma: no cover - only when dependency absent
    _U = None
    _X = None
    _UNI_OK = False
    _UNI_ERR = str(_e)


PAGE = 0x1000
_COPY_CAP = 16 * 1024 * 1024  # cap model memcpy/memset lengths
_STRLEN_CAP = 1 << 20
_MODELS = {
    "malloc",
    "calloc",
    "free",
    "memcpy",
    "memmove",
    "memset",
    "memcmp",
    "strlen",
    "strcpy",
    "strncpy",
    "strcmp",
}

_CONTEXTS: dict[str, "EmuContext"] = {}
_CTX_SEQ = [0]
_REG_CACHE: dict[str, int] = {}


def _require() -> None:
    if not _UNI_OK:
        raise IDAError(
            "The 'unicorn' package is not installed in this environment. "
            f"Install it (e.g. `uv add unicorn`) and restart. ({_UNI_ERR})"
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _align_down(x: int) -> int:
    return x & ~(PAGE - 1)


def _align_up(x: int, a: int = PAGE) -> int:
    return (x + a - 1) & ~(a - 1)


def _coerce_int(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return 0
        try:
            return int(s, 0)
        except ValueError:
            return parse_address(s)
    raise IDAError(f"cannot interpret integer value: {v!r}")


def _reg_table() -> dict[str, int]:
    """Lowercase register-name -> unicorn x86 register constant."""
    if _REG_CACHE:
        return _REG_CACHE
    names = {
        "rax": "RAX", "rbx": "RBX", "rcx": "RCX", "rdx": "RDX",
        "rsi": "RSI", "rdi": "RDI", "rbp": "RBP", "rsp": "RSP", "rip": "RIP",
        "r8": "R8", "r9": "R9", "r10": "R10", "r11": "R11",
        "r12": "R12", "r13": "R13", "r14": "R14", "r15": "R15",
        "eax": "EAX", "ebx": "EBX", "ecx": "ECX", "edx": "EDX",
        "esi": "ESI", "edi": "EDI", "ebp": "EBP", "esp": "ESP", "eip": "EIP",
        "ax": "AX", "al": "AL",
        "eflags": "EFLAGS", "flags": "EFLAGS",
        "fs_base": "FS_BASE", "gs_base": "GS_BASE",
        "xmm0": "XMM0", "xmm1": "XMM1", "xmm2": "XMM2", "xmm3": "XMM3",
    }
    for k, v in names.items():
        c = getattr(_X, "UC_X86_REG_" + v, None)
        if c is not None:
            _REG_CACHE[k] = c
    return _REG_CACHE


def _reg_const(name: str) -> int:
    c = _reg_table().get(name.strip().lower())
    if c is None:
        raise IDAError(f"unknown register: {name!r}")
    return c


def _is_64bit() -> bool:
    try:
        import ida_ida

        return bool(ida_ida.inf_is_64bit())
    except Exception:
        return bool(idc.get_inf_attr(idc.INF_LFLAG) & idc.LFLG_64BIT)


def _detect_arch() -> tuple[int, str]:
    """Return (bits, abi) for the current database. x86/x64 only in this build."""
    proc = (idc.get_inf_attr(idc.INF_PROCNAME) or "").lower()
    if not proc.startswith(("metapc", "8086", "80x86", "p")):
        raise IDAError(
            f"unicorn emulation supports x86/x64 only in this build (processor: {proc!r})"
        )
    bits = 64 if _is_64bit() else 32
    ft = idc.get_inf_attr(idc.INF_FILETYPE)
    abi = "ms" if ft == getattr(idc, "FT_PE", 11) else "sysv"
    return bits, abi


def _seg_ranges() -> list[tuple[int, int]]:
    import idautils

    out = []
    for s_ea in idautils.Segments():
        seg = ida_segment.getseg(s_ea)
        if seg:
            out.append((seg.start_ea, seg.end_ea))
    return out


_CALL_ITYPES = None


def _call_itypes() -> set[int]:
    global _CALL_ITYPES
    if _CALL_ITYPES is None:
        _CALL_ITYPES = {
            getattr(idaapi, n)
            for n in ("NN_call", "NN_callfi", "NN_callni")
            if hasattr(idaapi, n)
        }
    return _CALL_ITYPES


# ---------------------------------------------------------------------------
# Emulation context
# ---------------------------------------------------------------------------


class EmuContext:
    def __init__(self, cid: str, bits: int, abi: str):
        _require()
        self.id = cid
        self.bits = bits
        self.abi = abi
        self.ptr = bits // 8
        self.mask = (1 << bits) - 1
        arch = _U.UC_ARCH_X86
        mode = _U.UC_MODE_64 if bits == 64 else _U.UC_MODE_32
        self.uc = _U.Uc(arch, mode)
        rt = _reg_table()
        self.pc_reg = rt["rip"] if bits == 64 else rt["eip"]
        self.sp_reg = rt["rsp"] if bits == 64 else rt["esp"]
        self.ret_reg = rt["rax"] if bits == 64 else rt["eax"]

        self.pages: set[int] = set()  # page bases currently mapped
        self.page_tag: dict[int, str] = {}

        # scratch layout (stack / heap / return-trap), placed above the image
        self.seg_ranges = _seg_ranges()
        self.stack_base = 0
        self.stack_size = 0
        self.heap_base = 0
        self.heap_ptr = 0
        self.heap_end = 0
        self.trap = 0

        # stub / run policy
        self.stubs: dict[str, dict] = {}
        self.default_action: Optional[dict] = None
        self.skip_calls = False
        self.import_eas: set[int] = set()

        # per-run scratch
        self.until: Optional[int] = None
        self.insns = 0
        self.stop_reason: Optional[str] = None
        self.fault: Optional[tuple] = None
        self.trace_enabled = False
        self.trace: list[str] = []
        self.trace_max = 4096
        self.log: list[dict] = []
        self.last: dict = {}

        self.uc.hook_add(_U.UC_HOOK_CODE, _code_hook, self)
        self.uc.hook_add(
            _U.UC_HOOK_MEM_READ_UNMAPPED
            | _U.UC_HOOK_MEM_WRITE_UNMAPPED
            | _U.UC_HOOK_MEM_FETCH_UNMAPPED,
            _mem_hook,
            self,
        )

    # -- mapping ----------------------------------------------------------
    def map(self, base: int, size: int, data: Optional[bytes] = None, tag: str = "") -> None:
        base = _align_down(base)
        size = _align_up(size)
        p = base
        # map only pages not already mapped (avoid UC_ERR_MAP on overlap)
        run_start = None
        while p < base + size:
            if p in self.pages:
                if run_start is not None:
                    self._map_run(run_start, p, tag)
                    run_start = None
            else:
                if run_start is None:
                    run_start = p
            p += PAGE
        if run_start is not None:
            self._map_run(run_start, base + size, tag)
        if data:
            self.uc.mem_write(base, bytes(data[:size]))

    def _map_run(self, start: int, end: int, tag: str) -> None:
        self.uc.mem_map(start, end - start, _U.UC_PROT_ALL)
        for pg in range(start, end, PAGE):
            self.pages.add(pg)
            self.page_tag[pg] = tag

    def ensure(self, addr: int, size: int) -> None:
        if size <= 0:
            size = 1
        p = _align_down(addr)
        last = _align_down(addr + size - 1)
        while p <= last:
            if p not in self.pages:
                self._lazy_page(p)
            p += PAGE

    def _lazy_page(self, page: int) -> None:
        seg = ida_segment.getseg(page) or ida_segment.getseg(page + PAGE - 1)
        if seg is not None:
            data = read_bytes_bss_safe(page, PAGE)
            self.map(page, PAGE, data, tag="idb")
        else:
            self.map(page, PAGE, b"\x00" * PAGE, tag="scratch")

    def uread(self, addr: int, size: int) -> bytes:
        self.ensure(addr, size)
        return bytes(self.uc.mem_read(addr, size))

    def uwrite(self, addr: int, data: bytes) -> None:
        self.ensure(addr, len(data) or 1)
        self.uc.mem_write(addr, data)

    def read_uint(self, addr: int, size: int) -> int:
        return int.from_bytes(self.uread(addr, size), "little")

    # -- registers --------------------------------------------------------
    def rget(self, name: str) -> int:
        return self.uc.reg_read(_reg_const(name))

    def rset(self, name: str, value: int) -> None:
        self.uc.reg_write(_reg_const(name), value & self.mask)

    def pc(self) -> int:
        return self.uc.reg_read(self.pc_reg)

    def sp(self) -> int:
        return self.uc.reg_read(self.sp_reg)

    # -- heap -------------------------------------------------------------
    def halloc(self, size: int) -> int:
        p = self.heap_ptr
        self.heap_ptr += _align_up(max(size, 1), 16)
        self.ensure(p, self.heap_ptr - p)
        return p


def _log(ctx: EmuContext, entry: dict) -> None:
    ctx.log.append(entry)
    if len(ctx.log) > 200:
        del ctx.log[: len(ctx.log) - 200]


# ---------------------------------------------------------------------------
# Hooks (module-level; ctx passed as user_data)
# ---------------------------------------------------------------------------


def _mem_hook(uc, access, address, size, value, ud):
    ctx: EmuContext = ud
    in_idb = ida_segment.getseg(address) is not None or ida_bytes.is_loaded(address)
    if in_idb:
        ctx._lazy_page(_align_down(address))
        return True
    if access == _U.UC_MEM_FETCH_UNMAPPED:
        ctx.fault = ("fetch_unmapped", address)
        return False
    # unknown read/write -> zero scratch page (stack growth, unknown globals)
    ctx.map(_align_down(address), PAGE, b"\x00" * PAGE, tag="scratch")
    return True


def _code_hook(uc, address, size, ud):
    ctx: EmuContext = ud
    ctx.insns += 1
    if ctx.until is not None and address == ctx.until:
        ctx.stop_reason = "until"
        uc.emu_stop()
        return
    if ctx.trace_enabled and len(ctx.trace) < ctx.trace_max:
        ctx.trace.append(hex(address))
    if not (ctx.stubs or ctx.default_action or ctx.skip_calls):
        return
    insn = idaapi.insn_t()
    if idaapi.decode_insn(insn, address) <= 0:
        return
    if insn.itype not in _call_itypes():
        return
    target = _resolve_target(ctx, insn)
    act = _match_stub(ctx, address, target)
    if act is not None:
        _apply_stub(ctx, address, size, target, act)


def _resolve_target(ctx: EmuContext, insn) -> Optional[int]:
    op = insn.ops[0]
    t = op.type
    if t in (idaapi.o_near, idaapi.o_far, idaapi.o_mem):
        return op.addr
    if t == idaapi.o_reg:
        nm = idaapi.get_reg_name(op.reg, ctx.ptr)
        c = _reg_table().get((nm or "").lower())
        if c is not None:
            return ctx.uc.reg_read(c)
    # o_displ / o_phrase (register-indirect through memory): not statically
    # resolvable here; caught by skip_calls (as external) or an explicit
    # call-site-address stub.
    return None


def _match_stub(ctx: EmuContext, callsite: int, target: Optional[int]) -> Optional[dict]:
    ids = {hex(callsite)}
    if target is not None and target != idaapi.BADADDR and target >= 0:
        ids.add(hex(target))
        nm = idc.get_name(target)
        if nm:
            ids.add(nm)
            ids.add(nm.lower())
            if nm.startswith("__imp_"):
                ids.add(nm[6:])
                ids.add(nm[6:].lower())
    for i in ids:
        if i in ctx.stubs:
            return ctx.stubs[i]
    if ctx.default_action is not None:
        return ctx.default_action
    if ctx.skip_calls and _is_external(ctx, target):
        return {"kind": "skip"}
    return None


def _is_external(ctx: EmuContext, target: Optional[int]) -> bool:
    if target is None or target < 0 or target == idaapi.BADADDR:
        return True
    if not ida_bytes.is_loaded(target):
        return True
    if target in ctx.import_eas:
        return True
    f = idaapi.get_func(target)
    if f is None:
        return ida_segment.getseg(target) is None
    return bool(f.flags & idaapi.FUNC_THUNK)


def _apply_stub(ctx: EmuContext, address: int, size: int, target: Optional[int], act: dict) -> None:
    kind = act["kind"]
    if kind == "run":
        return
    ret: Optional[int] = None
    if kind == "model":
        ret = _run_model(ctx, act["name"])
    elif kind in ("return", "skip"):
        ret = act.get("value", 0)
    elif kind == "nop":
        ret = None
    if ret is not None:
        ctx.uc.reg_write(ctx.ret_reg, ret & ctx.mask)
    ctx.uc.reg_write(ctx.pc_reg, address + size)
    pop = act.get("pop", 0)
    if pop:
        ctx.uc.reg_write(ctx.sp_reg, ctx.sp() + pop)
    nm = None
    if target is not None and target >= 0 and target != idaapi.BADADDR:
        nm = idc.get_name(target) or None
    _log(
        ctx,
        {
            "event": "call",
            "at": hex(address),
            "target": hex(target) if (target is not None and target >= 0) else None,
            "name": nm,
            "action": act.get("name") or kind,
            "ret": hex(ret) if ret is not None else None,
        },
    )


# ---------------------------------------------------------------------------
# Argument reading + built-in models
# ---------------------------------------------------------------------------

_ARG_REGS = {
    ("ms", 64): ["rcx", "rdx", "r8", "r9"],
    ("sysv", 64): ["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
}


def _arg(ctx: EmuContext, i: int) -> int:
    """Read the i-th integer argument at the moment of a `call` (before it
    pushes the return address)."""
    if ctx.bits == 64:
        regs = _ARG_REGS[(ctx.abi, 64)]
        if i < len(regs):
            return ctx.rget(regs[i])
        # stack spill: return address not yet pushed at hook time
        if ctx.abi == "ms":
            off = 0x20 + 8 * (i - 4)  # after 32-byte shadow space
        else:
            off = 8 * (i - len(regs))
        return ctx.read_uint(ctx.sp() + off, 8)
    # x86 cdecl/stdcall: args already pushed, return addr not yet -> [sp + 4*i]
    return ctx.read_uint(ctx.sp() + 4 * i, 4)


def _run_model(ctx: EmuContext, name: str) -> int:
    name = name.lower()
    if name == "malloc":
        return ctx.halloc(_arg(ctx, 0))
    if name == "calloc":
        n, sz = _arg(ctx, 0), _arg(ctx, 1)
        total = min(n * sz, _COPY_CAP)
        p = ctx.halloc(total)
        ctx.uwrite(p, b"\x00" * total)
        return p
    if name == "free":
        return 0
    if name in ("memcpy", "memmove"):
        d, s, n = _arg(ctx, 0), _arg(ctx, 1), min(_arg(ctx, 2), _COPY_CAP)
        ctx.uwrite(d, ctx.uread(s, n))
        return d
    if name == "memset":
        d, c, n = _arg(ctx, 0), _arg(ctx, 1), min(_arg(ctx, 2), _COPY_CAP)
        ctx.uwrite(d, bytes([c & 0xFF]) * n)
        return d
    if name == "strlen":
        s = _arg(ctx, 0)
        n = 0
        while n < _STRLEN_CAP and ctx.uread(s + n, 1)[0] != 0:
            n += 1
        return n
    if name in ("strcpy", "strncpy"):
        d, s = _arg(ctx, 0), _arg(ctx, 1)
        cap = _arg(ctx, 2) if name == "strncpy" else _STRLEN_CAP
        buf = bytearray()
        while len(buf) < cap:
            b = ctx.uread(s + len(buf), 1)[0]
            buf.append(b)
            if b == 0:
                break
        if name == "strncpy" and len(buf) < cap:
            buf.extend(b"\x00" * (cap - len(buf)))
        ctx.uwrite(d, bytes(buf))
        return d
    if name == "strcmp":
        a, b = _arg(ctx, 0), _arg(ctx, 1)
        i = 0
        while i < _STRLEN_CAP:
            ca, cb = ctx.uread(a + i, 1)[0], ctx.uread(b + i, 1)[0]
            if ca != cb:
                return (ca - cb) & ctx.mask
            if ca == 0:
                return 0
            i += 1
        return 0
    if name == "memcmp":
        a, b, n = _arg(ctx, 0), _arg(ctx, 1), min(_arg(ctx, 2), _COPY_CAP)
        da, db = ctx.uread(a, n), ctx.uread(b, n)
        for ca, cb in zip(da, db):
            if ca != cb:
                return (ca - cb) & ctx.mask
        return 0
    raise IDAError(f"unknown model: {name!r}")


# ---------------------------------------------------------------------------
# Stub-spec normalisation
# ---------------------------------------------------------------------------


def _coerce_action(v: Any) -> dict:
    if isinstance(v, bool):
        return {"kind": "run"} if v else {"kind": "skip"}
    if isinstance(v, int):
        return {"kind": "return", "value": v}
    if isinstance(v, dict):
        if "model" in v:
            return {"kind": "model", "name": str(v["model"]), "pop": int(v.get("pop", 0))}
        kind = str(v.get("kind", "")).lower()
        if kind == "model":
            return {"kind": "model", "name": str(v.get("name", "")), "pop": int(v.get("pop", 0))}
        if kind in ("return", "ret"):
            return {"kind": "return", "value": _coerce_int(v.get("value", 0)), "pop": int(v.get("pop", 0))}
        if kind in ("skip", "nop", "run"):
            out = {"kind": kind, "pop": int(v.get("pop", 0))}
            if "value" in v:
                out["value"] = _coerce_int(v["value"])
            return out
        if "return" in v:
            return {"kind": "return", "value": _coerce_int(v["return"]), "pop": int(v.get("pop", 0))}
        return {"kind": "skip"}
    if isinstance(v, str):
        s = v.strip()
        sl = s.lower()
        if sl in ("skip", "skipcall"):
            return {"kind": "skip"}
        if sl == "nop":
            return {"kind": "nop"}
        if sl in ("run", "none", "descend"):
            return {"kind": "run"}
        if sl in _MODELS:
            return {"kind": "model", "name": sl}
        if sl.startswith("model:"):
            return {"kind": "model", "name": sl[6:]}
        if sl.startswith("return:") or sl.startswith("ret:"):
            return {"kind": "return", "value": _coerce_int(s.split(":", 1)[1])}
        try:
            return {"kind": "return", "value": int(s, 0)}
        except ValueError:
            return {"kind": "skip"}
    return {"kind": "skip"}


def _norm_stubs(raw: Optional[dict]) -> tuple[dict[str, dict], Optional[dict]]:
    stubs: dict[str, dict] = {}
    default: Optional[dict] = None
    if not raw:
        return stubs, default
    if not isinstance(raw, dict):
        raise IDAError("stubs must be an object mapping call target -> action")
    for k, v in raw.items():
        act = _coerce_action(v)
        if str(k).strip() in ("*", "any", "all"):
            default = act
            continue
        key = str(k).strip()
        idset = {key, key.lower()}
        try:
            ea = parse_address(key)
            if ea != idaapi.BADADDR:
                idset.add(hex(ea))
        except Exception:
            pass
        for ident in idset:
            stubs[ident] = act
    return stubs, default


# ---------------------------------------------------------------------------
# Context construction + run loop
# ---------------------------------------------------------------------------


def _collect_import_eas() -> set[int]:
    out: set[int] = set()
    n = ida_nalt.get_import_module_qty()
    for i in range(n):
        def cb(ea, name, ordinal):
            out.add(ea)
            return True

        ida_nalt.enum_import_names(i, cb)
    return out


def _new_context(
    bits: int,
    abi: str,
    stack_size: int,
    heap_size: int,
    stack_base: int = 0,
) -> EmuContext:
    _CTX_SEQ[0] += 1
    cid = f"uc{_CTX_SEQ[0]}"
    ctx = EmuContext(cid, bits, abi)
    ctx.import_eas = _collect_import_eas()

    stack_size = _align_up(max(stack_size, PAGE * 4))
    heap_size = _align_up(max(heap_size, PAGE * 4))

    max_end = max((e for _, e in ctx.seg_ranges), default=0x400000)
    base = _align_up(max_end, 0x100000) + 0x100000
    if stack_base:
        base = _align_up(stack_base, PAGE)

    ctx.stack_base = base
    ctx.stack_size = stack_size
    ctx.map(ctx.stack_base, stack_size, tag="stack")

    ctx.heap_base = _align_up(ctx.stack_base + stack_size + 0x10000, 0x10000)
    ctx.heap_end = ctx.heap_base + heap_size
    ctx.heap_ptr = ctx.heap_base
    ctx.map(ctx.heap_base, heap_size, tag="heap")

    ctx.trap = _align_up(ctx.heap_end + 0x10000, PAGE)  # never mapped; used as until

    # initial stack pointer, with headroom at the top
    ctx.uc.reg_write(ctx.sp_reg, ctx.stack_base + stack_size - 0x800)
    return ctx


def _reg_dump(ctx: EmuContext, names: Optional[list]) -> dict:
    if names:
        sel = normalize_list_input(names)
    elif ctx.bits == 64:
        sel = [
            "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "rip",
            "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eflags",
        ]
    else:
        sel = ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "eip", "eflags"]
    out = {}
    for nm in sel:
        try:
            out[nm] = hex(ctx.rget(nm))
        except Exception as e:
            out[nm] = f"<{e}>"
    return out


def _apply_regs(ctx: EmuContext, regs: Optional[dict]) -> None:
    if not regs:
        return
    for k, v in regs.items():
        ctx.rset(k, _coerce_int(v))


def _apply_mem(ctx: EmuContext, mem: Optional[list]) -> list:
    written = []
    for item in mem or []:
        addr = parse_address(item["addr"])
        data = bytes.fromhex("".join(str(item["data"]).split()))
        ctx.uwrite(addr, data)
        written.append({"addr": hex(addr), "bytes": len(data)})
    return written


def _read_mem(ctx: EmuContext, specs: Optional[list]) -> list:
    out = []
    for item in specs or []:
        addr = parse_address(item["addr"])
        size = min(int(item["size"]), 0x10000)
        out.append({"addr": hex(addr), "size": size, "data": ctx.uread(addr, size).hex()})
    return out


def _run_ctx(
    ctx: EmuContext,
    start: str,
    until: str,
    max_insns: int,
    timeout_ms: int,
    push_return: bool,
) -> dict:
    ctx.until = parse_address(until) if until else None
    ctx.insns = 0
    ctx.stop_reason = None
    ctx.fault = None

    begin = parse_address(start) if start else ctx.pc()
    if push_return:
        sp = ctx.sp() - ctx.ptr
        ctx.uwrite(sp, ctx.trap.to_bytes(ctx.ptr, "little"))
        ctx.uc.reg_write(ctx.sp_reg, sp)

    count = max_insns if max_insns and max_insns > 0 else 0
    timeout = int(timeout_ms * 1000) if timeout_ms and timeout_ms > 0 else 0  # us

    log_start = len(ctx.log)  # so the result reports only THIS run's calls
    err = None
    t0 = time.monotonic()
    try:
        ctx.uc.emu_start(begin, ctx.trap, timeout=timeout, count=count)
    except _U.UcError as e:
        err = e
    elapsed = time.monotonic() - t0

    pc = ctx.pc()
    result: dict = {
        "ctx": ctx.id,
        "start": hex(begin),
        "pc": hex(pc),
        "insns": ctx.insns,
        "elapsed_ms": round(elapsed * 1000, 1),
    }
    if err is not None:
        result["stop_reason"] = _fault_reason(ctx, err, pc)
        result["ok"] = False
    elif ctx.stop_reason == "until":
        result["stop_reason"] = "until"
        result["ok"] = True
    elif pc == ctx.trap:
        result["stop_reason"] = "returned"
        result["ok"] = True
        result["return"] = hex(ctx.uc.reg_read(ctx.ret_reg))
    elif count and ctx.insns >= count:
        result["stop_reason"] = "max_insns"
        result["ok"] = True
    elif timeout and elapsed * 1e6 >= timeout * 0.9:
        result["stop_reason"] = "timeout"
        result["ok"] = True
    else:
        result["stop_reason"] = "stopped"
        result["ok"] = True
    new_calls = ctx.log[log_start:]
    if new_calls:
        result["calls"] = new_calls
    ctx.last = {"stop_reason": result["stop_reason"], "pc": result["pc"], "insns": ctx.insns}
    return result


def _fault_reason(ctx: EmuContext, err, pc: int) -> str:
    errno = getattr(err, "errno", None)
    names = {
        getattr(_U, "UC_ERR_READ_UNMAPPED", 6): "read_unmapped",
        getattr(_U, "UC_ERR_WRITE_UNMAPPED", 7): "write_unmapped",
        getattr(_U, "UC_ERR_FETCH_UNMAPPED", 8): "fetch_unmapped",
        getattr(_U, "UC_ERR_INSN_INVALID", 10): "invalid_insn",
        getattr(_U, "UC_ERR_WRITE_PROT", 12): "write_prot",
        getattr(_U, "UC_ERR_READ_PROT", 13): "read_prot",
        getattr(_U, "UC_ERR_FETCH_PROT", 14): "fetch_prot",
    }
    label = names.get(errno, f"err{errno}")
    if ctx.fault and ctx.fault[0] == "fetch_unmapped":
        addr = ctx.fault[1]
        return (
            f"fault:unstubbed_call -> {hex(addr)} "
            f"(execution left the image; add a stub for this target or set skip_calls=true)"
        )
    return f"fault:{label} at {hex(pc)}"


def _ctx_summary(ctx: EmuContext) -> dict:
    return {
        "ctx": ctx.id,
        "bits": ctx.bits,
        "abi": ctx.abi,
        "pc": hex(ctx.pc()),
        "sp": hex(ctx.sp()),
        "pages_mapped": len(ctx.pages),
        "stack": {"base": hex(ctx.stack_base), "size": hex(ctx.stack_size)},
        "heap": {"base": hex(ctx.heap_base), "ptr": hex(ctx.heap_ptr), "end": hex(ctx.heap_end)},
        "skip_calls": ctx.skip_calls,
        "stub_keys": sorted({k for k in ctx.stubs})[:50],
        "has_default_stub": ctx.default_action is not None,
        "last": ctx.last,
    }


def _get_ctx(cid: str) -> EmuContext:
    ctx = _CONTEXTS.get(str(cid).strip())
    if ctx is None:
        raise IDAError(f"no such emulation context: {cid!r} (use unicorn_list / unicorn_create)")
    return ctx


# ===========================================================================
# Tools
# ===========================================================================


@tool
@idasync
@tool_timeout(150.0)
def unicorn_emulate(
    start: Annotated[str, "Function/address to start emulating (hex, decimal, or name)"],
    until: Annotated[str, "Stop when this address is reached; empty = run until the function returns"] = "",
    regs: Annotated[Optional[dict], "Initial register values, e.g. {'rcx':'0x1000','rdx':64}"] = None,
    mem: Annotated[Optional[list], "Bytes to write before running: [{'addr','data'(hex)}]"] = None,
    stubs: Annotated[Optional[dict], "Call handling: {target: action}. target = hex addr / call-site addr / import name / '*'. action = int (return value), 'skip', 'nop', 'run', 'model:malloc' (or a bare model name), 'return:N'"] = None,
    skip_calls: Annotated[bool, "Auto-skip calls that leave the image (imports/thunks), returning 0"] = False,
    read_regs: Annotated[Optional[list], "Registers to read back (default: the common GPR set)"] = None,
    read_mem: Annotated[Optional[list], "Memory to read back after running: [{'addr','size'}]"] = None,
    max_insns: Annotated[int, "Instruction budget (0 = unlimited; default 1,000,000)"] = 1_000_000,
    timeout_ms: Annotated[int, "Wall-clock cap in ms (default 15000)"] = 15000,
    stack: Annotated[str, "Optional stack base override (hex); empty = auto-place above the image"] = "",
    heap_size: Annotated[int, "Scratch heap size for malloc models (default 4 MB)"] = 4 * 1024 * 1024,
    trace: Annotated[bool, "Record the executed instruction addresses (capped)"] = False,
) -> dict:
    """One-shot emulation: run a routine in a fresh CPU sandbox and read back the result.

    Memory is mapped lazily from the IDB (BSS reads back as zero). By default a
    return address is pushed so `start` is treated as a function entry and
    emulation stops when it returns, with the return value in rax/eax reported
    as `return`. Provide `until` to stop at a specific address instead.

    Use `stubs`/`skip_calls` for external calls: emulation faults if it tries to
    execute an import (nothing is mapped there). On a fault the `stop_reason`
    tells you which call target to stub.
    """
    _require()
    bits, abi = _detect_arch()
    ctx = _new_context(bits, abi, stack_size=1024 * 1024, heap_size=heap_size,
                       stack_base=parse_address(stack) if stack else 0)
    ctx.stubs, ctx.default_action = _norm_stubs(stubs)
    ctx.skip_calls = bool(skip_calls)
    ctx.trace_enabled = bool(trace)
    try:
        _apply_regs(ctx, regs)
        _apply_mem(ctx, mem)
        result = _run_ctx(ctx, start, until, max_insns, timeout_ms, push_return=True)
        result["regs"] = _reg_dump(ctx, read_regs)
        if read_mem:
            result["mem"] = _read_mem(ctx, read_mem)
        if trace:
            result["trace"] = ctx.trace
        result.pop("ctx", None)
        return result
    except _U.UcError as e:
        return {"ok": False, "error": f"unicorn: {e}"}


@tool
@idasync
def unicorn_create(
    stack_size: Annotated[int, "Stack size in bytes (default 1 MB)"] = 1024 * 1024,
    heap_size: Annotated[int, "Scratch heap size for malloc models (default 4 MB)"] = 4 * 1024 * 1024,
    stack_base: Annotated[str, "Optional stack base override (hex); empty = auto"] = "",
    stubs: Annotated[Optional[dict], "Initial call-stub policy (see unicorn_emulate)"] = None,
    skip_calls: Annotated[bool, "Auto-skip external (import/thunk) calls"] = False,
) -> dict:
    """Create a persistent emulation context bound to the current IDB.

    Returns a context id (`ctx`) to pass to unicorn_run/step/read/write/etc.
    Memory maps lazily from the IDB on first access. The context lives until
    unicorn_destroy or the IDB worker exits.
    """
    _require()
    bits, abi = _detect_arch()
    ctx = _new_context(bits, abi, stack_size, heap_size,
                       parse_address(stack_base) if stack_base else 0)
    ctx.stubs, ctx.default_action = _norm_stubs(stubs)
    ctx.skip_calls = bool(skip_calls)
    _CONTEXTS[ctx.id] = ctx
    return _ctx_summary(ctx)


@tool
@idasync
@tool_timeout(150.0)
def unicorn_run(
    ctx: Annotated[str, "Context id from unicorn_create"],
    start: Annotated[str, "Start address; empty = continue from current pc"] = "",
    until: Annotated[str, "Stop when this address is reached; empty = run until return-trap/limits"] = "",
    stubs: Annotated[Optional[dict], "Merge/replace stub policy for this and later runs"] = None,
    skip_calls: Annotated[Optional[bool], "Override skip_calls for this and later runs"] = None,
    push_return: Annotated[bool, "Push a return address so `start` runs as a called function"] = False,
    max_insns: Annotated[int, "Instruction budget (0 = unlimited; default 1,000,000)"] = 1_000_000,
    timeout_ms: Annotated[int, "Wall-clock cap in ms (default 15000)"] = 15000,
    read_regs: Annotated[Optional[list], "Registers to read back after the run"] = None,
    read_mem: Annotated[Optional[list], "Memory to read back: [{'addr','size'}]"] = None,
) -> dict:
    """Run (or continue) a persistent context until it returns, hits `until`, or
    exhausts the instruction/time budget."""
    _require()
    c = _get_ctx(ctx)
    if stubs is not None:
        s, d = _norm_stubs(stubs)
        c.stubs.update(s)
        if d is not None:
            c.default_action = d
    if skip_calls is not None:
        c.skip_calls = bool(skip_calls)
    try:
        result = _run_ctx(c, start, until, max_insns, timeout_ms, push_return=push_return)
        result["regs"] = _reg_dump(c, read_regs)
        if read_mem:
            result["mem"] = _read_mem(c, read_mem)
        return result
    except _U.UcError as e:
        return {"ok": False, "ctx": c.id, "error": f"unicorn: {e}"}


@tool
@idasync
@tool_timeout(120.0)
def unicorn_step(
    ctx: Annotated[str, "Context id"],
    count: Annotated[int, "Number of instructions to execute (default 1)"] = 1,
    read_regs: Annotated[Optional[list], "Registers to read back (default: common GPRs)"] = None,
) -> dict:
    """Single-step (or step N instructions), then report the new pc, the next
    instruction's disassembly, and registers."""
    _require()
    c = _get_ctx(ctx)
    if count <= 0:
        count = 1
    try:
        result = _run_ctx(c, "", "", max_insns=count, timeout_ms=10000, push_return=False)
        pc = c.pc()
        result["disasm"] = idc.generate_disasm_line(pc, 0) or ""
        result["regs"] = _reg_dump(c, read_regs)
        return result
    except _U.UcError as e:
        return {"ok": False, "ctx": c.id, "error": f"unicorn: {e}"}


@tool
@idasync
def unicorn_read(
    ctx: Annotated[str, "Context id"],
    regs: Annotated[Optional[list], "Register names to read (default: common GPRs)"] = None,
    mem: Annotated[Optional[list], "Memory regions to read: [{'addr','size'}]"] = None,
) -> dict:
    """Read registers and/or memory from a context without running it."""
    _require()
    c = _get_ctx(ctx)
    out: dict = {"ctx": c.id, "regs": _reg_dump(c, regs)}
    if mem:
        out["mem"] = _read_mem(c, mem)
    return out


@tool
@idasync
def unicorn_write(
    ctx: Annotated[str, "Context id"],
    regs: Annotated[Optional[dict], "Register values to set, e.g. {'rax':'0x10'}"] = None,
    mem: Annotated[Optional[list], "Bytes to write: [{'addr','data'(hex)}]"] = None,
) -> dict:
    """Set registers and/or write memory (pages auto-map on demand)."""
    _require()
    c = _get_ctx(ctx)
    _apply_regs(c, regs)
    written = _apply_mem(c, mem)
    return {"ctx": c.id, "ok": True, "wrote": written, "regs": _reg_dump(c, list(regs) if regs else None)}


@tool
@idasync
def unicorn_map(
    ctx: Annotated[str, "Context id"],
    addr: Annotated[str, "Base address (page-aligned automatically)"],
    size: Annotated[int, "Region size in bytes"],
    data: Annotated[str, "Optional hex bytes to fill from the base"] = "",
    from_idb: Annotated[bool, "Fill the region with the IDB's bytes at this address"] = False,
) -> dict:
    """Explicitly map a memory region (e.g. an output buffer to pass to a
    function). Usually unnecessary — memory maps lazily — but handy for scratch
    buffers whose address you want to choose."""
    _require()
    c = _get_ctx(ctx)
    base = parse_address(addr)
    if from_idb:
        c.ensure(base, size)
    else:
        c.map(base, size, tag="user")
    if data:
        c.uwrite(base, bytes.fromhex("".join(data.split())))
    return {"ctx": c.id, "ok": True, "addr": hex(_align_down(base)), "size": hex(_align_up(size))}


@tool
@idasync
def unicorn_state(
    ctx: Annotated[str, "Context id"],
    regions: Annotated[bool, "Include the list of mapped memory regions"] = False,
) -> dict:
    """Full snapshot of a context: arch, pc/sp, mapped page count, stack/heap
    layout, stub policy, and the recent call/stub log."""
    _require()
    c = _get_ctx(ctx)
    out = _ctx_summary(c)
    out["regs"] = _reg_dump(c, None)
    out["recent_calls"] = list(c.log)[-25:]
    if regions:
        out["regions"] = [
            {"start": hex(s), "end": hex(e + 1), "perms": p} for (s, e, p) in c.uc.mem_regions()
        ]
    return out


@tool
@idasync
def unicorn_reset(
    ctx: Annotated[str, "Context id"],
) -> dict:
    """Reset a context to a fresh CPU: zero all registers, drop mapped pages
    (memory re-maps lazily from the IDB again), reset the heap and stack
    pointer. Stub policy and arch are kept."""
    _require()
    c = _get_ctx(ctx)
    for pg in list(c.pages):
        try:
            c.uc.mem_unmap(pg, PAGE)
        except Exception:
            pass
    c.pages.clear()
    c.page_tag.clear()
    c.heap_ptr = c.heap_base
    c.log.clear()
    c.trace.clear()
    c.last = {}
    c.map(c.stack_base, c.stack_size, tag="stack")
    c.map(c.heap_base, c.heap_end - c.heap_base, tag="heap")
    c.uc.reg_write(c.sp_reg, c.stack_base + c.stack_size - 0x800)
    return {"ctx": c.id, "ok": True, **_ctx_summary(c)}


@tool
@idasync
def unicorn_destroy(
    ctx: Annotated[str, "Context id, or 'all' to free every context"],
) -> dict:
    """Free a persistent context (or all of them)."""
    _require()
    if str(ctx).strip() == "all":
        n = len(_CONTEXTS)
        _CONTEXTS.clear()
        return {"ok": True, "destroyed": n}
    c = _CONTEXTS.pop(str(ctx).strip(), None)
    return {"ok": c is not None, "destroyed": 1 if c else 0}


@tool
@idasync
def unicorn_list() -> dict:
    """List active emulation contexts with a brief summary of each."""
    _require()
    return {
        "n": len(_CONTEXTS),
        "contexts": [_ctx_summary(c) for c in _CONTEXTS.values()],
    }
