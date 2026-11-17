"""Stateless paging shared by the Query and Status capabilities.

A large result is never dumped into the LLM or the chat at once - each call
returns one page (PAGE_SIZE rows) plus the true total, and the agent
attaches a ready-made next_page_prompt ("..., rows 21-40") the caller can
send straight back. Nothing is stored server-side between pages, so paging
survives restarts and works across multiple workers/instances."""

import re

PAGE_SIZE = 20


def page_meta(total: int, skip: int, returned: int) -> dict:
    """count is always the true total; has_more/truncated say whether rows
    exist beyond this page (truncated kept for existing callers/prompts)."""
    has_more = skip + returned < total
    return {"count": total, "skip": skip, "returned": returned, "has_more": has_more, "truncated": has_more}


def page_slice(rows: list[dict], skip: int, top: int) -> tuple[list[dict], dict]:
    """Pages an already-materialized list (Excel engine results, flattened stock rows)."""
    skip = max(0, skip)
    page = rows[skip : skip + top]
    return page, page_meta(len(rows), skip, len(page))


def next_page_prompt(subject: str, result: dict) -> str | None:
    """e.g. "Show production orders for plant 1710, rows 21-40" - or None on the last page."""
    if not result.get("has_more"):
        return None
    start = result["skip"] + result["returned"] + 1
    end = min(start + PAGE_SIZE - 1, result["count"])
    return f"Show {subject}, rows {start}-{end}"


_ROWS_RE = re.compile(r"\brows?\s+(\d+)\s*(?:-|–|to)\s*\d+", re.IGNORECASE)


def skip_from_text(text: str) -> int | None:
    """Parses "rows 21-40" out of a follow-up prompt into skip=20 - a
    deterministic backstop so paging never depends on the LLM doing the
    arithmetic itself."""
    match = _ROWS_RE.search(text or "")
    if not match:
        return None
    return max(0, int(match.group(1)) - 1)
