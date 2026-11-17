"""Report capability: build a downloadable report from any of the four
S/4 query sources (material master / material stock / production order /
material serial number).

A report is where a large result is meant to end up in full, so rows are
fetched page by page ($top/$skip) until S/4's own total is reached - a
single unpaged GET can come back silently short when the gateway applies
server-side paging. REPORT_MAX_ROWS is the safety cap that keeps one huge
request from exhausting the app's memory; hitting it is reported, never
silent.
"""

from app.core_capabilities._s4_apis import BY_NAME
from app.core_capabilities.query import flatten_stock_rows
from app.services.odata_utils import extract_count, extract_rows, odata_literal
from app.services.report_builder import PDF_MAX_ROWS, get_report_builder
from app.services.s4_client import get_s4_client

REPORT_PAGE_SIZE = 1000
REPORT_MAX_ROWS = 50_000


def generate_material_report(
    report_type: str = "material_stock",
    identifiers: list[str] | None = None,
    plant: str | None = None,
    report_format: str = "pdf",
) -> dict:
    if report_type not in BY_NAME:
        raise ValueError(f"Unknown report_type '{report_type}', expected one of {sorted(BY_NAME)}")

    rows, truncated = _fetch_report_rows(report_type, identifiers, plant)

    result = get_report_builder().build(rows, report_format, base_filename=f"{report_type}_report")
    notes = []
    if truncated:
        notes.append(
            f"The data has more than {REPORT_MAX_ROWS:,} rows - the report contains the first "
            f"{REPORT_MAX_ROWS:,}. Narrow it by material/product/order or plant to get the rest."
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
        "report_type": report_type,
        **({"note": " ".join(notes)} if notes else {}),
    }


def _fetch_all_pages(path: str, params: dict) -> tuple[list[dict], bool]:
    """Fetches every page of an entity set, up to REPORT_MAX_ROWS rows.
    Returns (rows, truncated)."""
    rows: list[dict] = []
    skip = 0
    while True:
        page_params = {**params, "$top": REPORT_PAGE_SIZE, "$skip": skip, "$inlinecount": "allpages"}
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


def _fetch_report_rows(
    report_type: str, identifiers: list[str] | None, plant: str | None
) -> tuple[list[dict], bool]:
    api = BY_NAME[report_type]

    if report_type == "material_stock":
        # Same treatment as query_material_stock(): Plant isn't a top-level
        # filter field on A_MaterialStock, so expand the nested association
        # and flatten into real line-item rows instead. Paging applies to the
        # parent records; the cap is re-checked on the flattened line items.
        params = {"$expand": "to_MatlStkInAcctMod"}
        if identifiers:
            params["$filter"] = " or ".join(f"Material eq '{odata_literal(i)}'" for i in identifiers)
        parents, truncated = _fetch_all_pages(api.path, params)
        rows = flatten_stock_rows(parents, plant)
        if len(rows) > REPORT_MAX_ROWS:
            return rows[:REPORT_MAX_ROWS], True
        return rows, truncated

    filters = []
    if identifiers:
        filters.append(
            " or ".join(f"{api.key_field} eq '{odata_literal(identifier)}'" for identifier in identifiers)
        )
    if plant and api.has_plant_field:
        filters.append(f"Plant eq '{odata_literal(plant)}'")
    params = {"$filter": " and ".join(f"({f})" for f in filters)} if filters else {}
    return _fetch_all_pages(api.path, params)
