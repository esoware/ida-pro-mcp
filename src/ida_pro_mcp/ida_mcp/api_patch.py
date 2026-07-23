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
    """Write a new copy of the input file on disk with all IDB patches applied."""
    try:
        if not path:
            raise ValueError("An output path is required")
        ok = ida_loader.gen_file(ida_loader.OFILE_EXE, path, 0, idaapi.BADADDR, 0)
        return {"ok": bool(ok), "path": path}
    except Exception as e:
        return {"ok": False, "path": path, "error": str(e)}


@tool
@idasync
def export_asm(
    path: Annotated[str, "Absolute output path for the .asm listing"],
) -> ExportResult:
    """Export the full disassembly listing to an .asm file on disk."""
    try:
        if not path:
            raise ValueError("An output path is required")
        ok = ida_loader.gen_file(ida_loader.OFILE_ASM, path, 0, idaapi.BADADDR, 0)
        return {"ok": bool(ok), "path": path}
    except Exception as e:
        return {"ok": False, "path": path, "error": str(e)}
