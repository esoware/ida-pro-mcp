"""Patch management and export operations for IDA Pro MCP.

Ported from the IDA GURU plugin. Provides the patch-workflow capabilities the
core API lacks: enumerating patched bytes, reverting patches to their original
values, bulk NOP/fill ranges, writing the patched binary back out to disk, and
exporting a disassembly listing.

Byte-level patching itself lives in api_memory.patch / api_modify.patch_asm;
this module deliberately does not duplicate those.
"""

from typing import Annotated, NotRequired, TypedDict

import idaapi
import idc
import ida_bytes
import ida_loader
import ida_fpro
import ida_nalt

from .rpc import tool
from .sync import idasync
from .utils import parse_address


class PatchEntry(TypedDict):
    addr: str
    original: int
    current: int


class RevertResult(TypedDict):
    addr: str
    reverted: int
    error: NotRequired[str]


class RangePatchResult(TypedDict):
    start: str
    end: str
    bytes: int
    error: NotRequired[str]


class ExportResult(TypedDict):
    ok: bool
    path: str
    patched_bytes: NotRequired[int]
    lines: NotRequired[int]
    error: NotRequired[str]


def _nop_byte() -> bytes:
    """Return the architecture's NOP fill byte (0x90 on x86, else 0x00)."""
    proc = (idc.get_inf_attr(idc.INF_PROCNAME) or "").lower()
    return b"\x90" if proc.startswith(("metapc", "8086", "80x86")) else b"\x00"


@tool
@idasync
def get_patch_list() -> list[PatchEntry]:
    """List every byte whose current value differs from its original value.

    Walks the database comparing get_original_byte against get_byte, so it
    reports all applied patches (from any source), not just ones made this
    session.
    """
    out: list[PatchEntry] = []

    def collect(ea: int, fpos: int, org_val: int, patch_val: int) -> int:
        out.append({"addr": hex(ea), "original": org_val, "current": patch_val})
        return 0

    ida_bytes.visit_patched_bytes(0, idaapi.BADADDR, collect)
    return out


@tool
@idasync
def revert_bytes(
    addr: Annotated[str, "Start address to revert (hex, decimal, or name)"],
    size: Annotated[int, "Number of bytes to revert to their original value"] = 1,
) -> RevertResult:
    """Revert patched bytes back to their original values."""
    try:
        ea = parse_address(addr)
        reverted = 0
        for i in range(size):
            if ida_bytes.revert_byte(ea + i):
                reverted += 1
        return {"addr": hex(ea), "reverted": reverted}
    except Exception as e:
        return {"addr": addr, "reverted": 0, "error": str(e)}


@tool
@idasync
def nop_range(
    start: Annotated[str, "Start address (inclusive; hex, decimal, or name)"],
    end: Annotated[str, "End address (exclusive; hex, decimal, or name)"],
) -> RangePatchResult:
    """Patch a range with NOP instructions (0x90 on x86, 0x00 otherwise)."""
    try:
        s = parse_address(start)
        e = parse_address(end)
        if e <= s:
            raise ValueError("end must be greater than start")
        ida_bytes.patch_bytes(s, _nop_byte() * (e - s))
        return {"start": hex(s), "end": hex(e), "bytes": e - s}
    except Exception as ex:
        return {"start": start, "end": end, "bytes": 0, "error": str(ex)}


@tool
@idasync
def fill_range(
    start: Annotated[str, "Start address (inclusive; hex, decimal, or name)"],
    end: Annotated[str, "End address (exclusive; hex, decimal, or name)"],
    value: Annotated[int, "Byte value to fill with (0-255)"],
) -> RangePatchResult:
    """Fill a range with a single repeated byte value."""
    try:
        s = parse_address(start)
        e = parse_address(end)
        if e <= s:
            raise ValueError("end must be greater than start")
        ida_bytes.patch_bytes(s, bytes([value & 0xFF]) * (e - s))
        return {"start": hex(s), "end": hex(e), "bytes": e - s}
    except Exception as ex:
        return {"start": start, "end": end, "bytes": 0, "error": str(ex)}


@tool
@idasync
def apply_patches_to_file(
    path: Annotated[str, "Absolute output path for the patched binary"],
) -> ExportResult:
    """Write a new copy of the input file on disk with all IDB patches applied.

    Copies the original input file and overwrites each patched byte at its file
    offset. This mirrors IDA's "Apply patches to input file" and, unlike
    gen_file(OFILE_EXE), works reliably for PE/ELF/Mach-O.
    """
    try:
        if not path:
            raise ValueError("An output path is required")

        input_path = ida_nalt.get_input_file_path()
        if not input_path:
            raise ValueError("Input file path is unavailable")
        with open(input_path, "rb") as fh:
            buf = bytearray(fh.read())

        applied = [0]

        def collect(ea: int, fpos: int, org_val: int, patch_val: int) -> int:
            if 0 <= fpos < len(buf):
                buf[fpos] = patch_val & 0xFF
                applied[0] += 1
            return 0

        ida_bytes.visit_patched_bytes(0, idaapi.BADADDR, collect)

        with open(path, "wb") as fh:
            fh.write(buf)
        return {"ok": True, "path": path, "patched_bytes": applied[0]}
    except Exception as e:
        return {"ok": False, "path": path, "error": str(e)}


@tool
@idasync
def export_asm(
    path: Annotated[str, "Absolute output path for the .asm listing"],
) -> ExportResult:
    """Export the full disassembly listing to an .asm file on disk."""
    qf = None
    try:
        if not path:
            raise ValueError("An output path is required")
        # gen_file needs a FILE* handle (a path string is rejected), so open the
        # destination through IDA's cross-module qfile_t and hand over its fp.
        qf = ida_fpro.qfile_t()
        if not qf.open(path, "wt"):
            raise ValueError(f"Could not open output file: {path}")
        lines = ida_loader.gen_file(
            ida_loader.OFILE_ASM, qf.get_fp(), 0, idaapi.BADADDR, 0
        )
        if lines == -1:
            raise ValueError("gen_file failed to generate the listing")
        return {"ok": True, "path": path, "lines": int(lines)}
    except Exception as e:
        return {"ok": False, "path": path, "error": str(e)}
    finally:
        if qf is not None:
            qf.close()
