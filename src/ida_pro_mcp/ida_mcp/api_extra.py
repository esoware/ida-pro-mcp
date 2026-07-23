"""Miscellaneous operations for IDA Pro MCP.

Ported from the IDA GURU plugin: importing a named type library (TIL) and
listing named labels within a segment. Also adds a ranked/paginated string
listing that the core API only exposes via regex search or the survey top-15.
"""

import re
from typing import Annotated, Any, NotRequired, TypedDict

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


class StringInfo(TypedDict):
    addr: str
    string: str
    length: int
    section: str
    xrefs: int


class ListStringsResult(TypedDict, total=False):
    n: int
    total: int
    strings: list[StringInfo]
    cursor: dict[str, Any]
    error: str


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


def _xref_count(ea: int) -> int:
    count = 0
    for _ in idautils.XrefsTo(ea):
        count += 1
    return count


@tool
@idasync
def list_strings(
    filter: Annotated[str, "Substring (or regex if regex=true) to match, case-insensitive"] = "",
    regex: Annotated[bool, "Treat filter as a regex instead of a substring"] = False,
    section: Annotated[str, "Only strings in this segment name, e.g. '.rdata'"] = "",
    min_length: Annotated[int, "Minimum string length to include"] = 1,
    sort: Annotated[str, "Sort order: 'xrefs' (default), 'addr', or 'length'"] = "xrefs",
    limit: Annotated[int, "Max results per page (default: 100, max: 500)"] = 100,
    offset: Annotated[int, "Skip first N results after sorting"] = 0,
) -> ListStringsResult:
    """List strings with filtering and sorting.

    Never dumps everything: results are filtered, sorted (by xref count by
    default so the most-referenced strings surface first), and paginated. Use
    this for discovery when you don't yet have a pattern; use find_regex when
    you do.
    """
    from .api_core import _get_strings_cache

    if limit <= 0:
        limit = 100
    if limit > 500:
        limit = 500
    if offset < 0:
        offset = 0

    matcher = None
    if filter:
        if regex:
            try:
                matcher = re.compile(filter, re.IGNORECASE).search
            except re.error as e:
                return {"n": 0, "total": 0, "strings": [], "error": f"Invalid regex: {e}"}
        else:
            needle = filter.lower()
            matcher = lambda text, _n=needle: _n in text.lower()

    # Filter first (cheap), so xref counting only runs on the surviving subset.
    filtered: list[tuple[int, str]] = []
    for ea, text in _get_strings_cache():
        if len(text) < min_length:
            continue
        if matcher and not matcher(text):
            continue
        if section:
            seg = ida_segment.getseg(ea)
            if not seg or ida_segment.get_segm_name(seg) != section:
                continue
        filtered.append((ea, text))

    total = len(filtered)

    rows: list[StringInfo] = []
    for ea, text in filtered:
        seg = ida_segment.getseg(ea)
        rows.append(
            {
                "addr": hex(ea),
                "string": text,
                "length": len(text),
                "section": ida_segment.get_segm_name(seg) if seg else "",
                "xrefs": _xref_count(ea),
            }
        )

    if sort == "addr":
        rows.sort(key=lambda r: int(r["addr"], 16))
    elif sort == "length":
        rows.sort(key=lambda r: r["length"], reverse=True)
    else:  # "xrefs" (default), tie-broken by address for stable output
        rows.sort(key=lambda r: (-r["xrefs"], int(r["addr"], 16)))

    page = rows[offset : offset + limit]
    more = offset + limit < total
    return {
        "n": len(page),
        "total": total,
        "strings": page,
        "cursor": {"next": offset + limit} if more else {"done": True},
    }
