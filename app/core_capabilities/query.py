"""Query capability: read-only lookups against any data source in _s4_apis.py
(the Smart Material CDS views, OData V4 service ZSD_SMC_MATERIAL) - with
simple filters, free conditions and a sort, paged server-side."""

from app.core_capabilities._s4_apis import get_source, normalize_filters
from app.core_capabilities.paging import page_meta
from app.services.odata_utils import extract_count, extract_rows
from app.services.s4_client import get_s4_client

# Max rows any one call fetches from S/4 (the page size `top` is clamped to
# this). Without it, a broad request can pull thousands of rows straight into
# the LLM's tool-result message and blow the model's context window (seen
# live: 525k tokens on one COOIS fetch). Callers page through larger results
# with skip - $count=true returns the true total in the same request.
QUERY_ROW_LIMIT = 200


def _clamp_top(top: int) -> int:
    return max(1, min(int(top), QUERY_ROW_LIMIT))


def _params(source_name: str, filters: dict | None, conditions: list[str] | None, order_by: str | None) -> tuple:
    source = get_source(source_name)
    params: dict = {}
    odata_filter = source.build_filter(filters, conditions)
    if odata_filter:
        params["$filter"] = odata_filter
    odata_orderby = source.build_orderby(order_by)
    if odata_orderby:
        params["$orderby"] = odata_orderby
    return source, params


def query_source(
    source_name: str,
    filters: dict | None = None,
    conditions: list[str] | None = None,
    order_by: str | None = None,
    skip: int = 0,
    top: int = QUERY_ROW_LIMIT,
) -> dict:
    """One page of rows from a data source. count is the true total matching
    the filters; has_more says whether rows exist beyond this page."""
    source, params = _params(source_name, filters, conditions, order_by)
    skip = max(0, int(skip or 0))
    params.update({"$top": _clamp_top(top), "$skip": skip, "$count": "true"})
    data = get_s4_client().get(source.path, params=params)
    rows = extract_rows(data)[: _clamp_top(top)]
    total = extract_count(data)
    if total is None:
        total = skip + len(rows)
    return {
        "query_type": source.name,
        "label": source.label,
        "filters": normalize_filters(filters),
        "conditions": list(conditions or []),
        "order_by": order_by or None,
        "results": rows,
        **page_meta(total, skip, len(rows)),
    }


def count_records(source_name: str, filters: dict | None = None, conditions: list[str] | None = None) -> dict:
    """How many rows of a data source match - the real S/4 total ($count with
    $top=1), without pulling the rows back."""
    source, params = _params(source_name, filters, conditions, None)
    params.update({"$top": 1, "$count": "true"})
    data = get_s4_client().get(source.path, params=params)
    count = extract_count(data)
    if count is None:
        count = len(extract_rows(data))
    return {
        "query_type": source.name,
        "label": source.label,
        "filters": normalize_filters(filters),
        "conditions": list(conditions or []),
        "count_only": True,
        "count": count,
    }
