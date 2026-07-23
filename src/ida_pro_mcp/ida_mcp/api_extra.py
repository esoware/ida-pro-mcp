"""Miscellaneous operations for IDA Pro MCP.

Ported from the IDA GURU plugin: importing a named type library (TIL) and
listing named labels within a segment. Also adds a ranked/paginated string
listing that the core API only exposes via regex search or the survey top-15.
"""

import re
from typing import Annotated, Any, NotRequired, TypedDict

import idaapi
import idc
import ida_bytes
import ida_funcs
import ida_segment
import ida_typeinf
import idautils

try:
    import ida_hexrays
except ImportError:  # decompiler not available
    ida_hexrays = None

from .rpc import tool
from .sync import idasync
from .utils import (
    Page,
    Xref,
    get_function,
    normalize_dict_list,
    normalize_list_input,
    paginate,
    parse_address,
    pattern_filter,
)


class ImportTilResult(TypedDict):
    ok: bool
    til: str
    error: NotRequired[str]


class Label(TypedDict):
    addr: str
    name: str


class Export(TypedDict):
    addr: str
    name: str
    ordinal: int


class ExportQuery(TypedDict, total=False):
    """Export query with filtering and pagination"""

    filter: Annotated[str, "Name glob/regex"]
    offset: Annotated[int, "Start index"]
    count: Annotated[int, "Max results (0=all)"]


class ExportsQueryPage(TypedDict):
    data: list[Export]
    next_offset: int | None


class XrefsFromResult(TypedDict, total=False):
    addr: str
    xrefs: list[Xref] | None
    more: bool
    xref_count: int
    message: str
    error: str


class CommentReadResult(TypedDict, total=False):
    addr: str
    regular: str | None
    repeatable: str | None
    function: str | None
    decompiler: list[str]
    error: str


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


@tool
@idasync
def xrefs_from(
    addrs: Annotated[list[str] | str, "Addresses or names to find cross-references FROM (e.g. '0x11a9', 'main')"],
    limit: Annotated[int, "Max xrefs per address (default: 100, max: 1000)"] = 100,
) -> list[XrefsFromResult]:
    """Return xrefs from address(es) — the targets each address references."""
    addrs = normalize_list_input(addrs)

    if limit <= 0 or limit > 1000:
        limit = 1000

    results = []
    for addr in addrs:
        try:
            ea = parse_address(addr)
            if not ida_bytes.is_mapped(ea):
                results.append(
                    {"addr": addr, "xrefs": None, "error": f"Address not mapped: {addr}"}
                )
                continue

            xrefs = []
            more = False
            for xref in idautils.XrefsFrom(ea, 0):
                if len(xrefs) >= limit:
                    more = True
                    break
                xrefs.append(
                    Xref(
                        addr=hex(xref.to),
                        type="code" if xref.iscode else "data",
                        fn=get_function(xref.to, raise_error=False),
                    )
                )
            entry: XrefsFromResult = {
                "addr": addr,
                "xrefs": xrefs,
                "more": more,
                "xref_count": len(xrefs),
            }
            if not xrefs:
                entry["message"] = "No cross-references from this address"
            results.append(entry)
        except Exception as e:
            results.append({"addr": addr, "xrefs": None, "error": str(e)})

    return results


def _read_decompiler_comments(func_ea: int, ea: int) -> list[str]:
    """Read saved decompiler user-comments at `ea` without decompiling."""
    if ida_hexrays is None or not ida_hexrays.init_hexrays_plugin():
        return []
    out: list[str] = []
    try:
        umc = ida_hexrays.restore_user_cmts(func_ea)
        if not umc:
            return []
        try:
            it = ida_hexrays.user_cmts_begin(umc)
            end = ida_hexrays.user_cmts_end(umc)
            while it != end:
                tl = ida_hexrays.user_cmts_first(it)
                cmt = ida_hexrays.user_cmts_second(it)
                if getattr(tl, "ea", idaapi.BADADDR) == ea:
                    text = str(cmt)
                    if text:
                        out.append(text)
                it = ida_hexrays.user_cmts_next(it)
        finally:
            ida_hexrays.user_cmts_free(umc)
    except Exception:
        return out
    return out


@tool
@idasync
def get_comments(
    addrs: Annotated[list[str] | str, "Addresses or names to read comments from"],
) -> list[CommentReadResult]:
    """Read back comments at address(es): regular/repeatable disassembly, the
    function comment (when the address is a function start), and decompiler
    user-comments. This is the read counterpart to set_comments/append_comments.
    """
    addrs = normalize_list_input(addrs)

    results: list[CommentReadResult] = []
    for addr in addrs:
        try:
            ea = parse_address(addr)
            if not ida_bytes.is_mapped(ea):
                results.append({"addr": addr, "error": f"Address not mapped: {addr}"})
                continue

            row: CommentReadResult = {
                "addr": addr,
                "regular": idaapi.get_cmt(ea, False) or None,
                "repeatable": idaapi.get_cmt(ea, True) or None,
            }

            fn = ida_funcs.get_func(ea)
            if fn and fn.start_ea == ea:
                row["function"] = (
                    idc.get_func_cmt(ea, True) or idc.get_func_cmt(ea, False) or None
                )

            if fn:
                decomp = _read_decompiler_comments(fn.start_ea, ea)
                if decomp:
                    row["decompiler"] = decomp

            results.append(row)
        except Exception as e:
            results.append({"addr": addr, "error": str(e)})

    return results


def _collect_exports() -> list[Export]:
    """Collect all exports / entry points in the current database."""
    out: list[Export] = []
    for _index, ordinal, ea, name in idautils.Entries():
        out.append(Export(addr=hex(ea), name=name or f"#{ordinal}", ordinal=ordinal))
    return out


@tool
@idasync
def exports(
    offset: Annotated[int, "Starting pagination index (default: 0)"],
    count: Annotated[int, "Maximum rows (0 returns all exports)"],
) -> Page[Export]:
    """List exports / entry points with ordinals using offset/count pagination."""
    return paginate(_collect_exports(), offset, count)


@tool
@idasync
def exports_query(
    queries: Annotated[
        list[ExportQuery] | ExportQuery,
        "Export query with name filter and pagination",
    ],
) -> list[ExportsQueryPage]:
    """Query exports with richer filtering than exports(offset,count)."""
    queries = normalize_dict_list(queries)
    all_exports = _collect_exports()
    results = []

    for query in queries:
        filtered = all_exports
        name_filter = query.get("filter", "")
        if name_filter:
            filtered = pattern_filter(filtered, name_filter, "name")

        results.append(
            paginate(filtered, query.get("offset", 0), query.get("count", 100))
        )

    return results


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
