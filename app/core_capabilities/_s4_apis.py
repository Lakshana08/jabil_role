"""Registry of every data source the Query and Report agents can read - one
per entity set of the Smart Material OData V4 service ZSD_SMC_MATERIAL
(service binding ZSB_SMC_MATERIAL), exactly the list in SMC_OData_V4_URLs.xlsx.
The standard APIs this app used to read (API_PRODUCT_SRV, API_MATERIAL_STOCK_SRV,
API_PRODUCTION_ORDER_2_SRV) are no longer used for reads.

Each DataSource names its view, the simple filters it supports (material /
plant / storage_location / order / ... -> the view's real field), and a short
description + key fields that go into the agents' prompts. On top of the simple
filters a caller can pass free conditions ("AgingDays > 90") and a sort
("ShortageQty desc") on any field of the view; field names are checked against
the service's $metadata (app/services/odata_metadata.py) and every value is
escaped, so nothing from the user reaches the URL unvalidated.

Field names below were taken from the live system's $metadata (2026-11-18).
Re-check with: python -m tools.smc_metadata_check --fields"""

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.services.odata_metadata import EntitySetSchema, Property, get_metadata_cache
from app.services.odata_utils import odata_literal
from app.services.s4_client import S4ClientError

# Every filter name a caller may use, with what it means - shown to the LLM.
FILTER_NAMES = {
    "material": "material / product number",
    "plant": "plant (e.g. CN16)",
    "storage_location": "storage location (e.g. A201)",
    "order": "production / work order number",
    "component": "component material of an order or BOM",
    "status": "status code or text of the view (see its description)",
    "material_group": "material / product group",
    "material_type": "material / product type (ZROH, HALB, FERT, ZNBW)",
    "order_type": "order type (ZPPC, ZNPC, ZRWC)",
    "mrp_controller": "MRP controller (e.g. ZAA)",
    "work_center": "work center (e.g. AP7MS150)",
}

_UNSPLIT_FILTERS = frozenset({"status"})

# Old parameter names still accepted from structured callers (Joule functions).
FILTER_ALIASES = {"product": "material", "material_number": "material", "production_order": "order"}

_NUMERIC_TYPES = {"Edm.Byte", "Edm.SByte", "Edm.Int16", "Edm.Int32", "Edm.Int64", "Edm.Decimal", "Edm.Double", "Edm.Single"}
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONDITION_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|<>|==|=|>|<|\bcontains\b|\bstartswith\b|\bne\b|\beq\b|\bgt\b|\bge\b|\blt\b|\ble\b)\s*(.*?)\s*$",
    re.IGNORECASE,
)
_OPERATORS = {
    "=": "eq", "==": "eq", "eq": "eq",
    "!=": "ne", "<>": "ne", "ne": "ne",
    ">": "gt", "gt": "gt", ">=": "ge", "ge": "ge",
    "<": "lt", "lt": "lt", "<=": "le", "le": "le",
    "contains": "contains", "startswith": "startswith",
}


class S4SchemaError(S4ClientError):
    """A filter, condition or sort the view can't take - the message says why."""

    def __init__(self, message: str):
        self.status_code = 0
        self.detail = message
        Exception.__init__(self, message)


@dataclass(frozen=True)
class DataSource:
    name: str  # query_type / report_type
    entity_set: str
    label: str  # singular, for prompts: "over-control material"
    description: str
    filters: dict[str, str]  # filter name -> the view's field
    key_filter: str  # what report `identifiers` scope on
    key_fields: tuple[str, ...]  # the most useful fields, listed to the LLM
    contains_filters: frozenset[str] = field(default_factory=frozenset)  # matched with contains(), not eq

    @property
    def path(self) -> str:
        return f"{get_settings().s4_smc_service_path.rstrip('/')}/{self.entity_set}"

    def schema(self) -> EntitySetSchema | None:
        return get_metadata_cache().entity_set(get_settings().s4_smc_service_path, self.entity_set)

    def prop(self, field_name: str) -> Property:
        """The view's property for a field name (case-insensitive). Without
        $metadata the name is trusted as written, as a plain string field."""
        schema = self.schema()
        if schema is None:
            if not _FIELD_RE.match(field_name):
                raise S4SchemaError(f"'{field_name}' is not a valid field name.")
            return Property(name=field_name)
        found = schema.find((field_name,))
        if found is None:
            raise S4SchemaError(
                f"The {self.entity_set} view has no field '{field_name}'. "
                f"Its fields are: {', '.join(schema.properties)}."
            )
        return found

    def build_filter(self, filters: dict[str, str] | None = None, conditions: list[str] | None = None) -> str | None:
        """$filter for the simple filters (values may be comma-separated for
        "any of") AND-ed with the free conditions. None when there's nothing to filter."""
        clauses = []
        for name, value in normalize_filters(filters).items():
            if name not in self.filters:
                supported = ", ".join(self.filters) or "none"
                raise S4SchemaError(
                    f"The {self.label} data can't be filtered by {name}. Filters it supports: {supported}."
                )
            prop = self.prop(self.filters[name])
            # Status values are free text and may contain commas themselves
            # ("Short - warehouse insufficient, expedite") - never split those.
            if name in _UNSPLIT_FILTERS:
                values = [str(value).strip()]
            else:
                values = [v.strip() for v in str(value).split(",") if v.strip()]
            if name in self.contains_filters:
                parts = [f"contains({prop.name},{_literal(v, prop)})" for v in values]
            else:
                parts = [f"{prop.name} eq {lit}" for v in values for lit in _eq_literals(v, prop)]
            clauses.append("(" + " or ".join(dict.fromkeys(parts)) + ")")
        for condition in conditions or []:
            clauses.append(self._condition(condition))
        return " and ".join(clauses) if clauses else None

    def build_orderby(self, order_by: str | None) -> str | None:
        """'AgingDays desc, Material' -> validated $orderby."""
        if not order_by or not order_by.strip():
            return None
        parts = []
        for item in order_by.split(","):
            words = item.split()
            if not words:
                continue
            direction = words[1].lower() if len(words) > 1 else "asc"
            if direction not in ("asc", "desc") or len(words) > 2:
                raise S4SchemaError(f"Can't sort by '{item.strip()}' - use '<field> asc' or '<field> desc'.")
            parts.append(f"{self.prop(words[0]).name} {direction}")
        return ",".join(parts) or None

    def _condition(self, condition: str) -> str:
        match = _CONDITION_RE.match(condition or "")
        if not match:
            raise S4SchemaError(
                f"Can't read the condition '{condition}' - write it as '<field> <operator> <value>', "
                "operators: = != > >= < <= contains startswith."
            )
        field_name, operator, value = match.groups()
        prop = self.prop(field_name)
        op = _OPERATORS[operator.lower()]
        value = value.strip().strip("'\"")
        if op in ("contains", "startswith"):
            return f"{op}({prop.name},'{odata_literal(value)}')"
        return f"{prop.name} {op} {_literal(value, prop)}"


def normalize_filters(filters: dict | None) -> dict[str, str]:
    """Drops empty values, maps old parameter names, lower-cases names."""
    normalized: dict[str, str] = {}
    for name, value in (filters or {}).items():
        if value is None or str(value).strip() == "":
            continue
        key = str(name).strip().lower().replace(" ", "_")
        normalized[FILTER_ALIASES.get(key, key)] = str(value).strip()
    return normalized


def _literal(value: str, prop: Property) -> str:
    if prop.type in _NUMERIC_TYPES:
        if not _NUMBER_RE.match(value):
            raise S4SchemaError(f"{prop.name} is a number - '{value}' isn't one.")
        return value
    if prop.type == "Edm.Date":
        if not _DATE_RE.match(value):
            raise S4SchemaError(f"{prop.name} is a date - write it as YYYY-MM-DD, not '{value}'.")
        return value
    if prop.type == "Edm.Boolean":
        if value.lower() not in ("true", "false"):
            raise S4SchemaError(f"{prop.name} is true/false - '{value}' isn't.")
        return value.lower()
    return f"'{odata_literal(value)}'"


def _eq_literals(value: str, prop: Property) -> list[str]:
    """Numeric strings also match their zero-padded / unpadded form - order
    and material numbers may be stored with or without SAP's leading zeros."""
    if prop.type != "Edm.String" or not value.isdigit():
        return [_literal(value, prop)]
    variants = [value, value.lstrip("0") or "0"]
    if prop.max_length and len(value) < prop.max_length:
        variants.append(value.zfill(prop.max_length))
        if prop.max_length > 18 and len(value) < 18:
            variants.append(value.zfill(18))
    return [_literal(v, prop) for v in dict.fromkeys(variants)]


_MATERIAL_STATUS = "status = MaterialStatus: AV (available to issue), OC (over-control / over-issued), NC (non-controlled)."

SOURCES: tuple[DataSource, ...] = (
    # --- Material master (MaterialMaster.json) --------------------------------
    DataSource(
        name="material_master",
        entity_set="MaterialMaster",
        label="material master",
        description="Material master (MM03) header data: type, description, base unit, group, weight, cross-plant status, created on/by.",
        filters={"material": "Product", "material_group": "ProductGroup", "material_type": "ProductType", "status": "CrossPlantStatus"},
        key_filter="material",
        key_fields=("Product", "ProductType", "ProductDescription", "BaseUnit", "ProductGroup", "GrossWeight", "WeightUnit", "CrossPlantStatus", "CreationDate", "CreatedByUser"),
    ),
    DataSource(
        name="material_plant",
        entity_set="MaterialPlant",
        label="material plant data",
        description="Plant-level material data (MRP view): MRP type, MRP controller, procurement type (E in-house, F external), ABC indicator, delivery and GR days, backflush.",
        filters={"material": "Product", "plant": "Plant", "mrp_controller": "MRPResponsible"},
        key_filter="material",
        key_fields=("Product", "Plant", "MRPType", "MRPResponsible", "ProcurementType", "SpecialProcurementType", "ABCIndicator", "PlannedDeliveryDurationInDays", "GoodsReceiptDuration", "BackflushIndicator", "IsBulkMaterial"),
    ),
    DataSource(
        name="material_valuation",
        entity_set="MaterialValuation",
        label="material valuation",
        description="Material valuation / price: standard price, price unit, currency, valuation class (valuation area = plant).",
        filters={"material": "Product", "plant": "ValuationArea"},
        key_filter="material",
        key_fields=("Product", "ValuationArea", "ValuationClass", "PriceDeterminationControl", "StandardPrice", "PriceUnitQty", "Currency"),
    ),
    DataSource(
        name="material_storage_location",
        entity_set="MaterialStorageLocation",
        label="material storage location",
        description="Which storage locations a material is set up in, with storage location name and warehouse bin.",
        filters={"material": "Product", "plant": "Plant", "storage_location": "StorageLocation"},
        key_filter="material",
        key_fields=("Product", "Plant", "StorageLocation", "StorageLocationName", "WarehouseStorageBin"),
    ),
    DataSource(
        name="material_bom",
        entity_set="MaterialBOMItem",
        label="BOM item",
        description="Bill of material items: header material -> components with quantity, unit, scrap %, issue storage location, operation.",
        filters={"material": "Material", "plant": "Plant", "component": "BillOfMaterialComponent"},
        key_filter="material",
        key_fields=("Material", "Plant", "BillOfMaterialItemNumber", "BillOfMaterialComponent", "ComponentDescription", "BillOfMaterialItemQuantity", "BillOfMaterialItemUnit", "ComponentScrapInPercent", "ProdOrderIssueLocation", "FixedQuantity", "IsBulkMaterial"),
    ),
    # --- Inventory (Inventory.json) -------------------------------------------
    DataSource(
        name="material_stock",
        entity_set="MaterialStock",
        label="material stock",
        description="Stock quantity (MMBE) per material / plant / storage location / batch / stock type.",
        filters={"material": "Material", "plant": "Plant", "storage_location": "StorageLocation"},
        key_filter="material",
        key_fields=("Material", "Plant", "StorageLocation", "Batch", "InventoryStockType", "InventorySpecialStockType", "MatlWrhsStkQtyInMatlBaseUnit", "MaterialBaseUnit"),
    ),
    DataSource(
        name="inventory",
        entity_set="Inventory",
        label="inventory",
        description="Inventory with value and aging: unrestricted / blocked / quality stock, standard price, stock value, aging days, last placement date.",
        filters={"material": "Material", "plant": "Plant", "storage_location": "StorageLocation", "material_group": "ProductGroup", "material_type": "ProductType"},
        key_filter="material",
        key_fields=("Material", "Plant", "StorageLocation", "ProductDescription", "ProductType", "AgingDays", "LastPlacementDate", "Unrestricted", "Blocked", "QualityInspection", "StandardPricePerUnit", "UnrestrictedStockValue", "Currency"),
    ),
    # --- Z_AGEDINV4 aged inventory workbook -----------------------------------
    DataSource(
        name="aged_inventory",
        entity_set="Sheet1_Data",
        label="aged inventory",
        description=f"Z_AGEDINV aged-inventory list (sheet 1): every material / storage location with aging days (aged = AgingDays > {get_settings().aged_threshold_days}), stock by type and stock value.",
        filters={"material": "Material", "plant": "Plant", "storage_location": "StorageLocation", "material_group": "MaterialGroup", "material_type": "MaterialType"},
        key_filter="material",
        key_fields=("Material", "MaterialDescription", "Plant", "StorageLocation", "AgingDays", "LastPlacementDate", "MaterialType", "MaterialGroup", "Unrestricted", "UnrestrictedConsignment", "Blocked", "QualityInspection", "StandardPricePerUnit", "UnRestrictStkVal", "Currency"),
    ),
    DataSource(
        name="wo_component",
        entity_set="Sheet2_WOComponent",
        label="work order component",
        description="Z_AGEDINV sheet 2: work-order component requirements - theoretical usage, net goods issue (NetGICO11N = issued via CO11N confirmation, NetGIOther = issued any other way), actual usage and remaining quantity per order and component.",
        filters={"order": "OrderNumber", "material": "Material", "component": "Component", "storage_location": "StorageLocation", "order_type": "OrderType", "mrp_controller": "MRPController"},
        key_filter="order",
        key_fields=("OrderNumber", "OrderType", "Material", "MaterialDescription", "Component", "StorageLocation", "TotalOrderQty", "QtyPer", "TheoUsageWithScrap", "NetGICO11N", "NetGIOther", "ActualUsage", "RemainingQty", "SumRemainingQty"),
    ),
    DataSource(
        name="wip_issuable",
        entity_set="Sheet3_WIPIssuable",
        label="WIP issuable material",
        description="Z_AGEDINV sheet 3: issuable quantity per storage location and material - unrestricted stock, controlled remaining qty, available-to-issue qty and status. " + _MATERIAL_STATUS,
        filters={"material": "Material", "storage_location": "StorageLocation", "status": "MaterialStatus"},
        key_filter="material",
        key_fields=("Material", "MaterialDescription", "StorageLocation", "AgingDays", "SumUnrestricted", "ControlledRemainingQty", "AvailableToIssueQty", "MaterialStatus", "MaterialStatusText"),
    ),
    DataSource(
        name="over_control",
        entity_set="Sheet4_OverControl",
        label="over-control material",
        description="Z_AGEDINV sheet 4: over-control / over-issued materials - controlled remaining qty is above unrestricted stock, so AvailableToIssueQty is negative. No plant field.",
        filters={"material": "Material", "storage_location": "StorageLocation"},
        key_filter="material",
        key_fields=("Material", "MaterialDescription", "StorageLocation", "AgingDays", "SumUnrestricted", "ControlledRemainingQty", "AvailableToIssueQty", "MaterialStatusText"),
    ),
    DataSource(
        name="putaway_pending",
        entity_set="Sheet5_PutawayPending",
        label="putaway-pending material",
        description="Z_AGEDINV sheet 5: material waiting for putaway in the xx40 putaway storage locations (A340, C840, ...), all non-controlled.",
        filters={"material": "Material", "storage_location": "StorageLocation"},
        key_filter="material",
        key_fields=("Material", "MaterialDescription", "StorageLocation", "AgingDays", "SumUnrestricted", "MaterialStatusText"),
    ),
    DataSource(
        name="suffix_s_material",
        entity_set="Sheet6_SuffixS",
        label="-S suffix (dummy) material",
        description="Z_AGEDINV sheet 6: materials whose number ends in -S (dummy parts) with stock, available-to-issue qty and status. " + _MATERIAL_STATUS,
        filters={"material": "Material", "storage_location": "StorageLocation", "status": "MaterialStatus"},
        key_filter="material",
        key_fields=("Material", "MaterialDescription", "StorageLocation", "AgingDays", "SumUnrestricted", "AvailableToIssueQty", "MaterialStatus", "MaterialStatusText"),
    ),
    # --- COOIS order status ---------------------------------------------------
    DataSource(
        name="order_status",
        entity_set="COOIS_OrderStatus",
        label="COOIS order status",
        description="COOIS user status per order. NMVT (no material movement) orders are excluded from material-control calculations (IsExcluded = X).",
        filters={"order": "OrderNumber", "status": "UserStatus"},
        key_filter="order",
        key_fields=("OrderNumber", "UserStatus", "IsExcluded"),
    ),
    # --- PP-GP-014C component issuance ----------------------------------------
    DataSource(
        name="component_issuance",
        entity_set="MOD14C_Data",
        label="14C component issuance",
        description="PP-GP-014C component issuance per order and component: BOM theoretical qty, net goods issue (NetGICO11N = issued via CO11N confirmation, NetGIOther = issued any other way), actual usage, remaining qty.",
        filters={"order": "OrderNumber", "material": "Material", "component": "Component", "storage_location": "StorageLocation", "order_type": "OrderType", "mrp_controller": "MRPController"},
        key_filter="order",
        key_fields=("OrderNumber", "OrderType", "Material", "MaterialDescription", "Component", "StorageLocation", "TotalOrderQty", "QtyPer", "SMEBOMTheoreticalQty", "NetGICO11N", "NetGIOther", "ActualUsage", "SMERemainingQty"),
    ),
    DataSource(
        name="component_issuance_summary",
        entity_set="MOD14C_Sheet1",
        label="14C need vs stock summary",
        description="PP-GP-014C summary per storage location + material (RowLabel = storage location followed by material, e.g. A201ACPE21680): need qty, stock qty, issuable qty.",
        filters={"material": "RowLabel", "storage_location": "RowLabel"},
        key_filter="material",
        key_fields=("RowLabel", "NeedQty", "StockQty", "IssuableQty"),
        contains_filters=frozenset({"material", "storage_location"}),
    ),
    # --- Production order (ProductionOrder.json) ------------------------------
    DataSource(
        name="production_order",
        entity_set="ProductionOrder",
        label="production order",
        description="Production order header (COOIS): material, order type, quantities (total / confirmed yield / scrap / delivered), planned and scheduled dates, work center, status flags (OrderIsReleased, OrderIsConfirmed, OrderIsDelivered, OrderIsTechnicallyCompleted = X when true).",
        filters={"order": "ManufacturingOrder", "material": "Material", "plant": "Plant", "storage_location": "StorageLocation", "order_type": "ManufacturingOrderType", "mrp_controller": "MRPController", "work_center": "WorkCenter"},
        key_filter="order",
        key_fields=("ManufacturingOrder", "ManufacturingOrderType", "Material", "ProductDescription", "Plant", "StorageLocation", "TotalQuantity", "MfgOrderConfirmedYieldQty", "ActualDeliveredQuantity", "MfgOrderPlannedStartDate", "MfgOrderPlannedEndDate", "WorkCenter", "MRPController", "OrderIsReleased", "OrderIsConfirmed", "OrderIsDelivered", "MfgOrderCreationDate", "CreatedByUser"),
    ),
    DataSource(
        name="production_order_component",
        entity_set="ProductionOrderComponent",
        label="production order component",
        description="Components (reservation items) of production orders: component material, required and withdrawn quantity, requirement date, storage location, backflush / bulk / phantom flags.",
        filters={"order": "ManufacturingOrder", "material": "Material", "plant": "Plant", "storage_location": "StorageLocation"},
        key_filter="order",
        key_fields=("ManufacturingOrder", "ReservationItem", "Material", "ProductDescription", "StorageLocation", "RequiredQuantity", "WithdrawnQuantity", "BaseUnit", "MatlCompRequirementDate", "MatlCompIsMarkedForBackflush", "IsBulkMaterialComponent", "DataQualityIssue"),
    ),
    DataSource(
        name="production_order_operation",
        entity_set="ProductionOrderOperation",
        label="production order operation",
        description="Operations of production orders: operation text, work center, planned vs confirmed yield / scrap quantity, scheduled start/end, confirmation flags.",
        filters={"order": "ManufacturingOrder", "plant": "Plant", "work_center": "WorkCenter"},
        key_filter="order",
        key_fields=("ManufacturingOrder", "ManufacturingOrderOperation", "OperationText", "WorkCenter", "OpPlannedTotalQuantity", "OpTotalConfirmedYieldQty", "OpTotalConfirmedScrapQty", "OpErlstSchedldExecStrtDte", "OpErlstSchedldExecEndDte", "OperationIsConfirmed"),
    ),
    DataSource(
        name="production_order_status",
        entity_set="ProductionOrderStatus",
        label="production order status",
        description="All system/user statuses per production order, one row each. status = StatusShortName: REL released, MSPT missing parts, PCNF partially confirmed, CNF confirmed, PDLV partially delivered, DLV delivered, TECO technically completed, NMVT no material movement, GMPS goods movement posted, ...",
        filters={"order": "ManufacturingOrder", "status": "StatusShortName"},
        key_filter="order",
        key_fields=("ManufacturingOrder", "StatusShortName", "StatusName", "IsUserStatus"),
    ),
    # --- Work order shortage (WorkOrderShortage.json) -------------------------
    DataSource(
        name="work_order_shortage",
        entity_set="WorkOrderShortage",
        label="work order shortage",
        description="Shortage status per work order: missing-parts count and OrderShortageStatus ('Missing parts' or 'No shortage'), with order quantity, confirmed yield, scheduled start.",
        filters={"order": "ManufacturingOrder", "material": "Material", "storage_location": "StorageLocation", "order_type": "ManufacturingOrderType", "mrp_controller": "MRPController", "status": "OrderShortageStatus"},
        key_filter="order",
        key_fields=("ManufacturingOrder", "ManufacturingOrderType", "Material", "ProductDescription", "StorageLocation", "TotalQuantity", "MfgOrderConfirmedYieldQty", "MfgOrderScheduledStartDate", "MissingPartsCount", "OrderShortageStatus", "OrderStatus"),
    ),
    DataSource(
        name="component_shortage",
        entity_set="ComponentShortage",
        label="component shortage",
        description="Shortage per order component: open demand, line-side stock, warehouse stock, ShortageQty and ShortageStatus ('Short - warehouse insufficient, expedite', 'Short at line - replenish from warehouse', 'Covered by line-side stock', 'No open demand'); ExcludeReason says why a line is excluded.",
        filters={"order": "ManufacturingOrder", "material": "Material", "storage_location": "StorageLocation", "status": "ShortageStatus"},
        key_filter="order",
        key_fields=("ManufacturingOrder", "Material", "ProductDescription", "StorageLocation", "MatlCompRequirementDate", "RequiredQuantity", "OpenDemandQty", "LineSideStock", "WarehouseStock", "ShortageQty", "ShortageStatus", "ExcludeReason"),
    ),
)

BY_NAME: dict[str, DataSource] = {source.name: source for source in SOURCES}

# For tools/smc_metadata_check.py: entity sets grouped by the spreadsheet's source file.
SMC_ENTITY_SETS: dict[str, tuple[str, ...]] = {
    "Z_AGEDINV4 (aged inventory)": ("Sheet1_Data", "Sheet2_WOComponent", "Sheet3_WIPIssuable", "Sheet4_OverControl", "Sheet5_PutawayPending", "Sheet6_SuffixS"),
    "COOIS (order status)": ("COOIS_OrderStatus",),
    "PP-GP-014C (component issuance)": ("MOD14C_Data", "MOD14C_Sheet1"),
    "Production order": ("ProductionOrder", "ProductionOrderComponent", "ProductionOrderOperation", "ProductionOrderStatus"),
    "Work order shortage": ("WorkOrderShortage", "ComponentShortage"),
    "Material master": ("MaterialMaster", "MaterialPlant", "MaterialValuation", "MaterialStorageLocation", "MaterialBOMItem"),
    "Inventory": ("Inventory", "MaterialStock"),
}


def source_catalog() -> str:
    """One block per source for the agents' system prompts."""
    lines = []
    for source in SOURCES:
        lines.append(
            f"- {source.name}: {source.description}\n"
            f"  filters: {', '.join(source.filters)}\n"
            f"  key fields: {', '.join(source.key_fields)}"
        )
    return "\n".join(lines)


_SCOPE_WORDS = {
    "material": "for material",
    "plant": "at plant",
    "storage_location": "in storage location",
    "order": "for order",
    "component": "for component",
    "status": "with status",
    "material_group": "in material group",
    "material_type": "of material type",
    "order_type": "of order type",
    "mrp_controller": "for MRP controller",
    "work_center": "at work center",
}


def describe_scope(filters: dict | None, conditions: list[str] | None = None, order_by: str | None = None) -> str:
    """' for material TG20 at plant CN16 where AgingDays > 90 sorted by AgingDays desc'
    - used in report / next-page prompts, so the agent reading it back gets the
    same filters again."""
    parts = [f"{_SCOPE_WORDS.get(name, name)} {value}" for name, value in normalize_filters(filters).items()]
    if conditions:
        parts.append("where " + " and ".join(conditions))
    if order_by:
        parts.append(f"sorted by {order_by}")
    return (" " + " ".join(parts)) if parts else ""


def get_source(name: str) -> DataSource:
    source = BY_NAME.get((name or "").strip().lower())
    if source is None:
        raise S4SchemaError(f"Unknown data source '{name}'. Available: {', '.join(BY_NAME)}.")
    return source
