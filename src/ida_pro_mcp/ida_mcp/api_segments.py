"""Segment management operations for IDA Pro MCP.

Ported from the IDA GURU plugin. The core API can read segment layout via
survey_binary, but cannot create, delete, or re-permission segments; these
tools add that.

Permission bits (bitwise-OR): 1 = execute, 2 = write, 4 = read.
"""

from typing import Annotated, NotRequired, TypedDict

import ida_segment
import idautils

from .rpc import tool
from .sync import idasync
from .utils import parse_address


class SegmentInfo(TypedDict):
    name: str
    start: str
    end: str
    perm: int
    class_: str


class SegmentOpResult(TypedDict):
    ok: bool
    start: NotRequired[str]
    end: NotRequired[str]
    error: NotRequired[str]


def _find_seg(ea: int):
    seg = ida_segment.getseg(ea)
    return seg if seg else None


@tool
@idasync
def list_segments() -> list[SegmentInfo]:
    """List all segments with their address range, permissions, and class."""
    out: list[SegmentInfo] = []
    for s_ea in idautils.Segments():
        seg = ida_segment.getseg(s_ea)
        if not seg:
            continue
        out.append(
            {
                "name": ida_segment.get_segm_name(seg),
                "start": hex(seg.start_ea),
                "end": hex(seg.end_ea),
                "perm": int(getattr(seg, "perm", 0)),
                "class_": ida_segment.get_segm_class(seg) or "",
            }
        )
    return out


@tool
@idasync
def create_segment(
    start: Annotated[str, "Start address (hex, decimal, or name)"],
    end: Annotated[str, "End address (exclusive; hex, decimal, or name)"],
    name: Annotated[str, "Segment name"] = "",
    klass: Annotated[str, "Segment class (CODE, DATA, BSS, ...)"] = "DATA",
    perm: Annotated[int, "Permission bits: 1=exec, 2=write, 4=read"] = 6,
) -> SegmentOpResult:
    """Create a new segment over an address range."""
    try:
        s = parse_address(start)
        e = parse_address(end)
        if e <= s:
            raise ValueError("end must be greater than start")
        if not ida_segment.add_segm(0, s, e, name or "", klass):
            return {"ok": False, "error": "add_segm failed"}
        seg = _find_seg(s)
        if seg is not None:
            seg.perm = perm
            seg.update()
        return {"ok": True, "start": hex(s), "end": hex(e)}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@tool
@idasync
def delete_segment(
    addr: Annotated[str, "Any address inside the segment to delete"],
) -> SegmentOpResult:
    """Delete the segment containing the given address."""
    try:
        seg = _find_seg(parse_address(addr))
        if not seg:
            return {"ok": False, "error": "segment not found"}
        ok = ida_segment.del_segm(seg.start_ea, 0)
        return {"ok": bool(ok)}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@tool
@idasync
def set_segment_permissions(
    addr: Annotated[str, "Any address inside the target segment"],
    perm: Annotated[int, "Permission bits: 1=exec, 2=write, 4=read"],
) -> SegmentOpResult:
    """Set the permission bits of the segment containing the given address."""
    try:
        seg = _find_seg(parse_address(addr))
        if not seg:
            return {"ok": False, "error": "segment not found"}
        seg.perm = perm
        seg.update()
        return {"ok": True}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}
