"""Miscellaneous operations for IDA Pro MCP.

Ported from the IDA GURU plugin: importing a named type library (TIL) and
listing named labels within a segment. These fill small gaps in the core API.
"""

from typing import Annotated, NotRequired, TypedDict

import ida_segment
import ida_typeinf
import idautils

from .rpc import tool
from .sync import idasync
from .utils import parse_address


class ImportTilResult(TypedDict):
    ok: bool
    til: str
    error: NotRequired[str]


class Label(TypedDict):
    addr: str
    name: str


@tool
@idasync
def import_til(
    name: Annotated[str, "Type library name to load (e.g. 'mssdk', 'gnulnx_x64')"],
) -> ImportTilResult:
    """Load a named type library (TIL) into the database's type system."""
    try:
        rc = ida_typeinf.add_til(name, ida_typeinf.ADDTIL_DEFAULT)
        # add_til returns ADDTIL_OK (1) on success, higher values carry warnings.
        return {"ok": rc >= 1, "til": name}
    except Exception as e:
        return {"ok": False, "til": name, "error": str(e)}


@tool
@idasync
def list_labels(
    segment: Annotated[str, "Any address inside the segment (hex, decimal, or name)"],
) -> list[Label]:
    """List all named labels within the segment containing the given address."""
    seg = ida_segment.getseg(parse_address(segment))
    if not seg:
        return []
    out: list[Label] = []
    for ea, name in idautils.Names():
        if seg.start_ea <= ea < seg.end_ea:
            out.append({"addr": hex(ea), "name": name})
    return out
