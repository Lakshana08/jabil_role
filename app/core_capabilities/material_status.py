"""Bridges the Status Agent's tools to app/engine/*.py - reads sample_data/
through the loader's file cache (Excel-only, no live-S4 adapter). No LLM
code here."""

from app.core_capabilities.paging import page_slice
from app.engine.aging import compute_aging
from app.engine.loader import (
    load_component_issuance,
    load_inventory_aging,
    load_order_master,
    load_order_status,
)
from app.engine.machine_head import compute_machine_head_candidates
from app.engine.material_status import compute_material_status


# Largest page any one call returns. A broad/unscoped check can match
# thousands of rows; count is always the true total, and callers page through
# the rest with skip.
DEFAULT_RESULT_LIMIT = 1000


def _page(results: list[dict], skip: int, limit: int) -> dict:
    rows, meta = page_slice(results, skip, max(1, min(limit, DEFAULT_RESULT_LIMIT)))
    return {**meta, "results": rows}


def query_material_status(
    *,
    work_order: str | None = None,
    material: str | None = None,
    storage_location: str | None = None,
    status_filter: str | None = None,
    skip: int = 0,
    limit: int = DEFAULT_RESULT_LIMIT,
) -> dict:
    """status_filter is one of 'non_controlled', 'over_control', 'available', or None for all.
    Safe to call with no filters at all - a broad/unscoped query is valid, not an error."""
    components = load_component_issuance()
    stock = load_inventory_aging()
    order_status = load_order_status()

    results = compute_material_status(
        components,
        stock,
        order_status,
        work_order=work_order,
        material=material,
        storage_location=storage_location,
    )
    if status_filter:
        results = [r for r in results if r["material_status"] == status_filter]

    return _page(results, skip, limit)


def query_aging(
    *,
    material: str | None = None,
    storage_location: str | None = None,
    skip: int = 0,
    limit: int = DEFAULT_RESULT_LIMIT,
) -> dict:
    """Returns only rows where Aging Days > 14 - matching AgingCheck's stated
    purpose. Safe to call with no filters at all."""
    aging_rows = load_inventory_aging()
    results = compute_aging(aging_rows, material=material, storage_location=storage_location)
    aged_only = [r for r in results if r["is_aged"]]
    return _page(aged_only, skip, limit)


def query_machine_head_candidates(
    *, order_type: str = "ZNPC", skip: int = 0, limit: int = DEFAULT_RESULT_LIMIT
) -> dict:
    work_orders = load_order_master()
    components = load_component_issuance()
    order_status = load_order_status()
    results = compute_machine_head_candidates(work_orders, components, order_status, order_type=order_type)
    return _page(results, skip, limit)
