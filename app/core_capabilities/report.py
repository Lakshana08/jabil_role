"""Report capability: build a downloadable report (PDF/Excel/CSV) from any
data source in _s4_apis.py - the Smart Material CDS views (OData V4 service
ZSD_SMC_MATERIAL) - with the same filters / conditions / sort as a query.

A report is where a large result is meant to end up in full, so rows are
fetched page by page ($top/$skip) until S/4's own total is reached - a
single unpaged GET can come back silently short when the gateway applies
server-side paging. REPORT_MAX_ROWS is the safety cap that keeps one huge
request from exhausting the app's memory; hitting it is reported, never
silent.
"""

from app.core_capabilities._s4_apis import get_source, normalize_filters
from app.services.odata_utils import extract_count, extract_rows
from app.services.report_builder import PDF_MAX_ROWS, get_report_builder
from app.services.s4_client import get_s4_client

REPORT_PAGE_SIZE = 1000
REPORT_MAX_ROWS = 50_000


def generate_material_report(
    report_type: str = "material_stock",
    identifiers: list[str] | None = None,
    plant: str | None = None,
    report_format: str = "pdf",
    filters: dict | None = None,
    conditions: list[str] | None = None,
    order_by: str | None = None,
) -> dict:
    """identifiers / plant are the older inputs, still accepted: identifiers
    scope the source's main key (material or order), plant adds a plant filter
    where the source has one."""
    source = get_source(report_type)
    filters = normalize_filters(filters)
    if identifiers:
        filters.setdefault(source.key_filter, ",".join(str(i) for i in identifiers))
    if plant and "plant" in source.filters:
        filters.setdefault("plant", plant)

    params: dict = {}
    odata_filter = source.build_filter(filters, conditions)
    if odata_filter:
        params["$filter"] = odata_filter
    odata_orderby = source.build_orderby(order_by)
    if odata_orderby:
        params["$orderby"] = odata_orderby

    rows, truncated = _fetch_all_pages(source.path, params)

    result = get_report_builder().build(rows, report_format, base_filename=f"{source.name}_report")
    notes = []
    if truncated:
        notes.append(
            f"The data has more than {REPORT_MAX_ROWS:,} rows - the report contains the first "
            f"{REPORT_MAX_ROWS:,}. Narrow it with a filter to get the rest."
        )
    if result["rows_in_file"] < len(rows):
        notes.append(
            f"PDF reports are limited to {PDF_MAX_ROWS:,} rows - this one shows the first "
            f"{result['rows_in_file']:,} of {len(rows):,}. Use CSV or Excel for the full data."
        )
    return {
        "download_url": result["download_url"],
        "filename": result["filename"],
        "row_count": len(rows),
        "rows_in_file": result["rows_in_file"],
        "truncated": truncated,
        "report_type": source.name,
        "label": source.label,
        "filters": filters,
        "conditions": list(conditions or []),
        **({"note": " ".join(notes)} if notes else {}),
    }


def _fetch_all_pages(path: str, params: dict) -> tuple[list[dict], bool]:
    """Fetches every page of an entity set, up to REPORT_MAX_ROWS rows.
    Returns (rows, truncated)."""
    rows: list[dict] = []
    skip = 0
    while True:
        page_params = {**params, "$top": REPORT_PAGE_SIZE, "$skip": skip, "$count": "true"}
        data = get_s4_client().get(path, params=page_params)
        page = extract_rows(data)
        rows.extend(page)
        skip += len(page)
        total = extract_count(data)
        # Trust S/4's total when it gives one - a short page alone doesn't mean
        # the end, since the gateway may cap page size below REPORT_PAGE_SIZE.
        if total is not None:
            done = not page or skip >= total
        else:
            done = len(page) < REPORT_PAGE_SIZE
        if done:
            return rows, False
        if len(rows) >= REPORT_MAX_ROWS:
            return rows[:REPORT_MAX_ROWS], True
