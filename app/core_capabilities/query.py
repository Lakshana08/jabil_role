"""Query capability: read-only S/4HANA lookups - MM03 (material master),
MMBE (stock, and serialized stock), COOIS (production order info)."""

from app.core_capabilities._s4_apis import (
    BY_NAME,
    MATERIAL_SERIAL_NUMBER,
    MATERIAL_STOCK,
    PRODUCTION_ORDER,
    PRODUCT_MASTER,
)
from app.core_capabilities.paging import page_meta, page_slice
from app.services.odata_utils import extract_count, extract_rows, odata_literal, strip_odata_noise
from app.services.s4_client import get_s4_client

# Max rows any one call fetches from S/4 (the page size `top` is clamped to
# this). Without it, a broad filter (e.g. "production orders in plant 1710"
# with no order/material narrowing) can pull back thousands of rows straight
# into the LLM's tool-result message and blow the model's context window
# (seen live: 525k tokens on one COOIS fetch). Callers page through larger
# results with skip - $inlinecount=allpages returns the true total in the
# same request, so every page knows how many rows exist in all.
QUERY_ROW_LIMIT = 200


def _clamp_top(top: int) -> int:
    return max(1, min(top, QUERY_ROW_LIMIT))


def _page_params(skip: int, top: int) -> dict:
    return {"$top": _clamp_top(top), "$skip": max(0, skip), "$inlinecount": "allpages"}


def _paged(data: dict, rows: list[dict], skip: int) -> dict:
    total = extract_count(data)
    if total is None:
        # No inline count returned - all we can vouch for is this page's rows.
        total = max(0, skip) + len(rows)
    return page_meta(total, max(0, skip), len(rows))


def query_material_master(product: str) -> dict:
    """MM03 - material/product master data. Descriptions live on the
    related A_ProductDescription entity, reached via to_Description and
    flattened into a plain "descriptions" list here. Filtered on the key, so
    at most one row - no paging needed."""
    data = get_s4_client().get(
        PRODUCT_MASTER.path,
        params={
            "$filter": f"Product eq '{odata_literal(product)}'",
            "$expand": "to_Description",
            **_page_params(0, QUERY_ROW_LIMIT),
        },
    )
    rows = extract_rows(data)[:QUERY_ROW_LIMIT]
    for row in rows:
        row["descriptions"] = _clean_nested(row.pop("to_Description", None))
    return {"product": product, "results": rows, **_paged(data, rows, 0)}


def _clean_nested(nested) -> list[dict]:
    """Normalizes an expanded OData v2 nav property into a plain, cleaned list."""
    if isinstance(nested, dict):
        nested = nested.get("results", [])
    if not isinstance(nested, list):
        return []
    return [strip_odata_noise(entry) for entry in nested]


def query_material_stock(
    material: str, plant: str | None = None, skip: int = 0, top: int = QUERY_ROW_LIMIT
) -> dict:
    """MMBE - stock overview, optionally scoped to a plant. Plant isn't a
    top-level $filter field, so results are flattened and filtered client-side.

    $skip/$top can't page here: they apply to the A_MaterialStock parent
    records (one per material), not the plant/storage-location line items
    nested under them. So the line items are flattened first and the page is
    cut from that list - count is the true number of line items."""
    params = {
        "$filter": f"Material eq '{odata_literal(material)}'",
        "$expand": "to_MatlStkInAcctMod",
        "$top": QUERY_ROW_LIMIT,
    }
    data = get_s4_client().get(MATERIAL_STOCK.path, params=params)
    all_rows = flatten_stock_rows(extract_rows(data)[:QUERY_ROW_LIMIT], plant)
    rows, meta = page_slice(all_rows, skip, _clamp_top(top))
    return {"material": material, "plant": plant, "results": rows, **meta}


def flatten_stock_rows(rows: list[dict], plant: str | None = None) -> list[dict]:
    """Flattens A_MaterialStock rows into one row per plant/storage-location
    line item, optionally filtered to a plant. Shared with report.py."""
    flattened = []
    for row in rows:
        nested = row.get("to_MatlStkInAcctMod")
        if isinstance(nested, dict):
            nested = nested.get("results", [])
        if not isinstance(nested, list):
            continue
        for entry in nested:
            if plant and entry.get("Plant") != plant:
                continue
            # entry comes from the raw expanded sub-collection, not through
            # extract_rows(), so it still carries __metadata/__deferred noise.
            flattened.append(strip_odata_noise({"Material": row.get("Material"), **entry}))
    return flattened


def query_production_order(
    production_order: str | None = None,
    material: str | None = None,
    plant: str | None = None,
    skip: int = 0,
    top: int = QUERY_ROW_LIMIT,
) -> dict:
    """COOIS - production order info, filterable by order/material/plant.
    Paged server-side with $skip/$top."""
    filters = []
    if production_order:
        filters.append(f"ManufacturingOrder eq '{odata_literal(production_order)}'")
    if material:
        filters.append(f"Material eq '{odata_literal(material)}'")
    if plant:
        filters.append(f"Plant eq '{odata_literal(plant)}'")
    params = _page_params(skip, top)
    if filters:
        params["$filter"] = " and ".join(filters)
    data = get_s4_client().get(PRODUCTION_ORDER.path, params=params)
    rows = extract_rows(data)[: _clamp_top(top)]
    return {
        "production_order": production_order,
        "material": material,
        "plant": plant,
        "results": rows,
        **_paged(data, rows, skip),
    }


def query_material_serial_numbers(
    material: str,
    plant: str | None = None,
    serial_number: str | None = None,
    skip: int = 0,
    top: int = QUERY_ROW_LIMIT,
) -> dict:
    """MMBE (serialized stock) - stock broken down by serial number/equipment.
    Paged server-side with $skip/$top."""
    filters = [f"Material eq '{odata_literal(material)}'"]
    if plant:
        filters.append(f"Plant eq '{odata_literal(plant)}'")
    if serial_number:
        filters.append(f"SerialNumber eq '{odata_literal(serial_number)}'")
    params = {"$filter": " and ".join(filters), **_page_params(skip, top)}
    data = get_s4_client().get(MATERIAL_SERIAL_NUMBER.path, params=params)
    rows = extract_rows(data)[: _clamp_top(top)]
    return {
        "material": material,
        "plant": plant,
        "serial_number": serial_number,
        "results": rows,
        **_paged(data, rows, skip),
    }


def count_all_records(query_type: str) -> dict:
    """Total record count for a query source, with no filter at all - answers
    "how many X are there in total" for any of the 4 sources. Uses
    $inlinecount=allpages with $top=1 so S/4 computes the real total without
    this actually pulling back the full dataset."""
    if query_type not in BY_NAME:
        raise ValueError(f"Unknown query_type '{query_type}', expected one of {sorted(BY_NAME)}")
    api = BY_NAME[query_type]
    data = get_s4_client().get(api.path, params={"$top": 1, "$inlinecount": "allpages"})
    count = extract_count(data)
    if count is None:
        # Server didn't return an inline count (shouldn't happen given
        # $inlinecount=allpages, but fall back to counting returned rows
        # rather than reporting an unknown total).
        count = len(extract_rows(data))
    return {"query_type": query_type, "count": count}