"""Reads the four raw SAP report exports in sample_data/ into plain lists
of dicts, keyed by each sheet's real column headers. openpyxl only, no
pandas - the only place in the engine that touches a file at all."""

import os
import threading
from functools import lru_cache

import openpyxl

_SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sample_data"
)


_parse_lock = threading.Lock()


def _read_sheet(filename: str, sheet_name: str) -> list[dict]:
    """Parsed once per file version, not on every request - openpyxl parsing
    is slow and memory-heavy, and every Status/dashboard call reads several
    sheets. Keyed on the file's mtime, so replacing a file in sample_data/
    is picked up automatically. Returns fresh row dicts each call, so a caller
    mutating its rows can never corrupt the cached copy."""
    path = os.path.join(_SAMPLE_DATA_DIR, filename)
    with _parse_lock:
        rows = _parse_sheet(path, sheet_name, os.path.getmtime(path))
    return [dict(row) for row in rows]


@lru_cache(maxsize=16)
def _parse_sheet(path: str, sheet_name: str, mtime: float) -> tuple[dict, ...]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet_name]
        rows = worksheet.iter_rows(values_only=True)
        header = list(next(rows))
        return tuple(dict(zip(header, row)) for row in rows if any(value is not None for value in row))
    finally:
        workbook.close()


def load_order_master() -> list[dict]:
    """RAW_PP134_OrderMaster.xlsx - work order master data (PP-134)."""
    return _read_sheet("RAW_PP134_OrderMaster.xlsx", "Order Master (PP-134)")


def load_component_issuance() -> list[dict]:
    """RAW_14C_ComponentIssuance.xlsx (14C). Columns K/L are both literally
    named 'B' in the source (export quirk) - neither is used, so harmless."""
    return _read_sheet("RAW_14C_ComponentIssuance.xlsx", "Component Issuance (14C)")


def load_order_status() -> list[dict]:
    """RAW_COOIS_OrderStatus.xlsx - order status flags (COOIS). Two columns: Order, User Status."""
    return _read_sheet("RAW_COOIS_OrderStatus.xlsx", "Order Status (COOIS)")


def load_inventory_aging() -> list[dict]:
    """RAW_ZAGEDINV_Inventory.xlsx - stock and aging by material/storage location (Z_AGEDINV)."""
    return _read_sheet("RAW_ZAGEDINV_Inventory.xlsx", "Inventory & Aging (Z_AGEDINV)")
