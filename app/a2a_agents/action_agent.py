import asyncio
import logging
import re
from contextvars import ContextVar

from langchain_core.tools import tool

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentSkill

from app.a2a_agents.common import build_agent_card, emit_response, get_structured_input
from app.core_capabilities.action import perform_material_action, perform_production_order_action
from app.services import pending_changes
from app.services.ai_core import run_agent
from app.services.pending_changes import PendingChangeError
from app.services.s4_client import S4ClientError

logger = logging.getLogger(__name__)

_PRODUCTION_ORDER_ACTIONS = {"create_production_order", "update_production_order"}
_MATERIAL_ACTIONS = {"update_status", "update_weight", "update_description", "create_material", "delete_description"}
ALL_ACTIONS = _MATERIAL_ACTIONS | _PRODUCTION_ORDER_ACTIONS

# Set only by propose_actions(): while it holds a list, the tools below record
# the change they WOULD make instead of sending it to S/4, so a chat channel
# (the Teams bot, via the Assistant Agent) can show it to the user and execute
# exactly that recorded change after they confirm. Joule and direct callers
# never set it, so their behaviour is unchanged. LangChain copies the context
# into the tool thread, so the tools see the same list.
_proposals: ContextVar[list[dict] | None] = ContextVar("action_proposals", default=None)

SKILL = AgentSkill(
    id="perform_material_action",
    name="Perform Material Action",
    description=(
        "Updates material master fields in S/4HANA (T-code MM02) and creates new "
        "material masters (T-code MM01), via SAP's Product Master API (API_PRODUCT_SRV, "
        "entities A_Product / A_ProductDescription). Five actions supported: 'update_status' "
        "(payload.status - the new CrossPlantStatus code), 'update_weight' "
        "(payload.gross_weight, payload.weight_unit), 'update_description' "
        "(payload.description, optional payload.language, default 'EN'), "
        "'create_material' (requires material_number plus payload.product_type, "
        "payload.industry_sector, payload.base_unit, and payload.description - this system "
        "rejects a create with no description at all; optional payload.language, "
        "payload.material_group), and 'delete_description' "
        "(removes a material's description for a given payload.language, default "
        "'EN'). Accepts either structured JSON (material_number, action, payload) "
        "or a plain free-text instruction. Also creates and updates production orders "
        "(T-code CO01/CO02) via SAP's Production Order API (API_PRODUCTION_ORDER_2_SRV, "
        "entity A_ProductionOrder_2): 'create_production_order' (payload.material, "
        "payload.production_plant, payload.manufacturing_order_type, payload.total_quantity, "
        "payload.mfg_order_planned_end_date - ISO 8601, e.g. '2026-12-01T00:00:00', and "
        "payload.production_version - this system rejects a create with no production version "
        "at all; SAP assigns the order number, so no order_number is given for this action) "
        "and 'update_production_order' (requires order_number - the "
        "ManufacturingOrder number - plus at least one of payload.total_quantity or "
        "payload.mfg_order_planned_end_date)."
    ),
    tags=[
        "s4hana",
        "material",
        "action",
        "mm01",
        "mm02",
        "co01",
        "co02",
        "product-master-api",
        "api_product_srv",
        "production-order-api",
        "api_production_order_2_srv",
    ],
    examples=[
        "Change the status of material TG20 to Z1",
        "Update the gross weight of material TG20 to 12.5 KG",
        "Update the description of material TG20 to 'Trade Item 20'",
        "Create a new material TG99, type FERT, industry sector M, base unit EA, description 'New Trade Item'",
        "Delete the German description of material TG20",
        "Create a production order for material FG126, plant 1710, order type YBM1, quantity 5, planned end 2026-12-01",
        "Update production order 1002200 to quantity 10 and planned end date 2026-12-05",
    ],
    input_modes=["application/json", "text/plain"],
    output_modes=["application/json", "text/plain"],
)


def _material_action(material_number: str, action: str, payload: dict) -> dict:
    pending = _proposals.get()
    if pending is None:
        return perform_material_action(material_number=material_number, action=action, payload=payload)
    proposal = {"action": action, "material_number": material_number, "payload": payload}
    pending.append(proposal)
    return {"status": "awaiting_confirmation", "change": describe_action(proposal)}


def _production_order_action(order_number: str | None, action: str, payload: dict) -> dict:
    pending = _proposals.get()
    if pending is None:
        return perform_production_order_action(order_number=order_number, action=action, payload=payload)
    proposal = {"action": action, "order_number": order_number, "payload": payload}
    pending.append(proposal)
    return {"status": "awaiting_confirmation", "change": describe_action(proposal)}


def describe_action(proposal: dict) -> str:
    """Plain-English one-liner of a change, shown to the user before they confirm it."""
    action = proposal.get("action")
    payload = proposal.get("payload") or {}
    material = proposal.get("material_number")
    language = payload.get("language") or "EN"
    if action == "update_status":
        return f"Set the status of material {material} to {payload.get('status')}"
    if action == "update_weight":
        return f"Set the gross weight of material {material} to {payload.get('gross_weight')} {payload.get('weight_unit')}"
    if action == "update_description":
        return f'Set the {language} description of material {material} to "{payload.get("description")}"'
    if action == "create_material":
        details = (
            f"type {payload.get('product_type')}, industry sector {payload.get('industry_sector')}, "
            f"base unit {payload.get('base_unit')}"
        )
        if payload.get("material_group"):
            details += f", material group {payload['material_group']}"
        return f'Create material {material} ({details}) with {language} description "{payload.get("description")}"'
    if action == "delete_description":
        return f"Delete the {language} description of material {material}"
    if action == "create_production_order":
        return (
            f"Create a production order for material {payload.get('material')} in plant "
            f"{payload.get('production_plant')}: order type {payload.get('manufacturing_order_type')}, "
            f"quantity {payload.get('total_quantity')}, planned end {payload.get('mfg_order_planned_end_date')}, "
            f"production version {payload.get('production_version')}"
        )
    if action == "update_production_order":
        changes = []
        if payload.get("total_quantity") is not None:
            changes.append(f"quantity to {payload['total_quantity']}")
        if payload.get("mfg_order_planned_end_date") is not None:
            changes.append(f"planned end date to {payload['mfg_order_planned_end_date']}")
        return f"Update production order {proposal.get('order_number')}: set " + " and ".join(changes or ["nothing"])
    return f"{action} {payload}"


@tool
def update_material_status(material_number: str, status: str) -> dict:
    """Update the cross-plant status (CrossPlantStatus) of a material master record (T-code MM02).
    Uses SAP's Product Master API (API_PRODUCT_SRV)."""
    return _material_action(material_number=material_number, action="update_status", payload={"status": status})


@tool
def update_material_weight(material_number: str, gross_weight: str, weight_unit: str) -> dict:
    """Update the gross weight (GrossWeight) and weight unit (WeightUnit) of a material master record (T-code MM02).
    Uses SAP's Product Master API (API_PRODUCT_SRV)."""
    return _material_action(
        material_number=material_number,
        action="update_weight",
        payload={"gross_weight": gross_weight, "weight_unit": weight_unit},
    )


@tool
def update_material_description(material_number: str, description: str, language: str = "EN") -> dict:
    """Update the description of a material master record (T-code MM02). language is a 2-letter SAP language code, defaults to EN.
    Uses SAP's Product Master API (API_PRODUCT_SRV)."""
    return _material_action(
        material_number=material_number,
        action="update_description",
        payload={"description": description, "language": language},
    )


@tool
def create_material(
    material_number: str,
    product_type: str,
    industry_sector: str,
    base_unit: str,
    description: str,
    material_group: str | None = None,
    language: str = "EN",
) -> dict:
    """Create a new material master record (T-code MM01). material_number is the
    external material number to assign. product_type is the SAP material type code
    (e.g. 'FERT', 'HAWA', 'ROH'). industry_sector is the SAP industry sector code
    (e.g. 'M', 'C', 'P', 'R'). base_unit is the base unit of measure (e.g. 'EA', 'KG').
    description is required - this system rejects a create with no description at
    all. material_group is optional; language is a 2-letter SAP language code for
    the description, defaults to EN. Uses SAP's Product Master API (API_PRODUCT_SRV)."""
    payload = {
        "product_type": product_type,
        "industry_sector": industry_sector,
        "base_unit": base_unit,
        "description": description,
        "material_group": material_group,
        "language": language,
    }
    return _material_action(material_number=material_number, action="create_material", payload=payload)


@tool
def delete_material_description(material_number: str, language: str = "EN") -> dict:
    """Delete a material master's description for a given language (T-code MM02).
    language is a 2-letter SAP language code, defaults to EN.
    Uses SAP's Product Master API (API_PRODUCT_SRV)."""
    return _material_action(
        material_number=material_number, action="delete_description", payload={"language": language}
    )


@tool
def create_production_order(
    material: str,
    production_plant: str,
    manufacturing_order_type: str,
    total_quantity: str,
    mfg_order_planned_end_date: str,
    production_version: str,
) -> dict:
    """Create a new production order in S/4HANA (T-code CO01) via the Production Order API
    (API_PRODUCTION_ORDER_2_SRV). material is the material to produce. production_plant is
    the SAP plant code. manufacturing_order_type is the SAP order type code (e.g. 'YBM1').
    total_quantity is the order quantity. mfg_order_planned_end_date is an ISO 8601 datetime
    string (e.g. '2026-12-01T00:00:00'). production_version is required - this system rejects
    a create with no production version at all. SAP assigns the resulting production order
    number - it's returned in the result."""
    payload = {
        "material": material,
        "production_plant": production_plant,
        "manufacturing_order_type": manufacturing_order_type,
        "total_quantity": total_quantity,
        "mfg_order_planned_end_date": mfg_order_planned_end_date,
        "production_version": production_version,
    }
    return _production_order_action(order_number=None, action="create_production_order", payload=payload)


@tool
def update_production_order(
    order_number: str,
    total_quantity: str | None = None,
    mfg_order_planned_end_date: str | None = None,
) -> dict:
    """Update the total quantity and/or planned end date of an existing production order in
    S/4HANA (T-code CO02) via the Production Order API (API_PRODUCTION_ORDER_2_SRV). order_number
    is the production order (ManufacturingOrder) number to update - "production order 1000000",
    "order 1000000" and "manufacturing order 1000000" all mean order_number='1000000'. mfg_order_planned_end_date is an ISO 8601
    datetime string (e.g. '2026-12-05T00:00:00'). At least one of total_quantity or
    mfg_order_planned_end_date must be given."""
    payload = {"total_quantity": total_quantity, "mfg_order_planned_end_date": mfg_order_planned_end_date}
    return _production_order_action(
        order_number=order_number, action="update_production_order", payload=payload
    )


_TOOLS = [
    update_material_status,
    update_material_weight,
    update_material_description,
    create_material,
    delete_material_description,
    create_production_order,
    update_production_order,
]

_AGENT_SYSTEM_PROMPT = """You update material master fields, create new materials, delete \
material descriptions, and create/update production orders in S/4HANA using the tools \
available. Only call a tool if the user's instruction clearly states all the required values \
- never guess or invent values (especially product_type, industry_sector, and base_unit for \
create_material; material, production_plant, manufacturing_order_type, total_quantity, and \
mfg_order_planned_end_date for create_production_order). If anything required is missing or \
ambiguous, ask the user to clarify instead of calling a tool. Keep answers short and factual \
(1-2 sentences)."""


_CONFIRMATION_NOTE = """

IMPORTANT - in this mode calling a tool does NOT change anything in SAP: it only PREPARES the change, \
and the system itself then shows the user Confirm / Cancel buttons. So whenever the instruction states \
all required values, ALWAYS call the matching tool right away - never ask the user to confirm or \
double-check first (the system does that), and never answer without calling the tool. Only skip the \
tool when a required value is genuinely missing, and then ask for that value. After the tool returns \
status "awaiting_confirmation", reply in one short sentence saying what will change once they confirm - \
never say it has been done."""

# The model, told that changes need confirmation, occasionally answers "please confirm this change"
# as plain text instead of calling the tool - which prepares nothing, so there is no code and no
# buttons. Such a reply gets one automatic retry with this reminder.
_SELF_CONFIRM_RE = re.compile(r"\bconfirm", re.IGNORECASE)
_RETRY_REMINDER = (
    "\n\n(System note: call the matching tool now - calling it only prepares the change; "
    "the user confirms afterwards with a button.)"
)


async def propose_actions(text: str) -> tuple[tuple[str, list] | None, list[dict]]:
    """Free-text instruction -> the change(s) it would make, without touching S/4.
    Returns (run_agent result, proposals); proposals is empty when the model
    asked a clarifying question instead of calling a tool."""
    proposals: list[dict] = []
    token = _proposals.set(proposals)
    try:
        agent_result = await run_agent(_AGENT_SYSTEM_PROMPT + _CONFIRMATION_NOTE, _TOOLS, text)
    finally:
        _proposals.reset(token)
    return agent_result, proposals


async def run_structured_action(args: dict) -> tuple[str, dict]:
    """Executes one structured action ({action, material_number | order_number,
    payload}) against S/4 - no LLM involved. Returns (summary, data); failures
    come back as a summary plus {"error": ...}, never as an exception. Shared by
    this agent's structured input and by confirmed changes from the Teams bot."""
    action = args.get("action")
    payload = args.get("payload")
    if not action:
        return "I need an action to perform.", {"error": "missing_required_fields"}

    if action in _PRODUCTION_ORDER_ACTIONS:
        order_number = args.get("order_number")
        if action == "update_production_order" and not order_number:
            return "I need an order_number to update a production order.", {"error": "missing_required_fields"}
        try:
            result = await asyncio.to_thread(
                perform_production_order_action, order_number=order_number, action=action, payload=payload
            )
        except S4ClientError as exc:
            return f"Action failed against S/4HANA: {exc}", {"error": "s4_error"}
        except ValueError as exc:
            return str(exc), {"error": "invalid_action"}
        identifier = order_number or result.get("manufacturing_order")
    else:
        material_number = args.get("material_number")
        if not material_number:
            return "I need both a material_number and an action to perform.", {"error": "missing_required_fields"}
        try:
            result = await asyncio.to_thread(
                perform_material_action, material_number=material_number, action=action, payload=payload
            )
        except S4ClientError as exc:
            return f"Action failed against S/4HANA: {exc}", {"error": "s4_error"}
        except ValueError as exc:
            return str(exc), {"error": "invalid_action"}
        identifier = material_number

    if result.get("status") == "unchanged":
        return result["message"], result
    summary = f"Action '{action}' completed" + (f" for {identifier}." if identifier else ".")
    return summary, result


# "Confirm change CHG-7F3K2Q" / "Cancel change CHG-7F3K2Q" - what the Confirm / Cancel
# buttons send back (in Joule, Teams and the web UI alike).
_DECISION_RE = re.compile(r"^\s*(confirm|cancel)\b.*?\b(CHG-[A-Z0-9]{6})\b", re.IGNORECASE | re.DOTALL)


def parse_decision(text: str) -> tuple[str, str] | None:
    """("confirm" | "cancel", code) if the text is a Confirm/Cancel reply, else None."""
    match = _DECISION_RE.match(text or "")
    return (match.group(1).lower(), match.group(2).upper()) if match else None


async def prepare_change(text: str, requested_by: dict | None = None) -> tuple[str, dict | None]:
    """Free-text change request -> prepared change saved under a one-time code.
    Nothing is sent to S/4. Returns (reply text, data part); data carries the
    code and the exact Confirm / Cancel texts the channel's buttons send back.
    When the model asks for missing details instead, returns its question."""
    agent_result, proposals = await propose_actions(text)
    if agent_result is not None and not proposals and _SELF_CONFIRM_RE.search(agent_result[0]):
        logger.info("Model asked for confirmation itself instead of preparing the change - retrying once")
        agent_result, proposals = await propose_actions(text + _RETRY_REMINDER)
    if agent_result is None:
        return "I couldn't process that request right now (AI Core unavailable).", {"error": "ai_core_unavailable"}
    if not proposals:
        return agent_result[0], None
    changes = [describe_action(p) for p in proposals]
    code = pending_changes.save(proposals, changes, requested_by)
    reply = (
        "Please confirm this change in SAP S/4HANA:\n\n"
        + "\n".join(f"- {c}" for c in changes)
        + f"\n\nNothing has been changed yet. Confirmation code {code} is valid for 15 minutes."
    )
    data = {
        "pending_actions": proposals,
        "changes": changes,
        "confirm_code": code,
        "confirm_prompt": f"Confirm change {code}",
        "cancel_prompt": f"Cancel change {code}",
    }
    return reply, data


async def resolve_change(decision: str, code: str, requested_by: dict | None = None) -> tuple[str, dict]:
    """Confirms (executes exactly the stored actions) or cancels a prepared change.
    If the change was prepared for a known user (Teams), only that user may decide it."""
    who = requested_by or {}
    try:
        owner = (pending_changes.peek(code).get("requested_by") or {}).get("user_key")
        if owner and owner != who.get("user_key"):
            return "Only the person who asked for this change can confirm or cancel it.", {"error": "not_requester"}
        record = pending_changes.take(code)  # one-time: a second click finds nothing
    except PendingChangeError as exc:
        return str(exc), {"error": "invalid_or_used_code"}

    caller = who.get("email") or who.get("name") or who.get("user_key") or "unknown (Joule/web)"
    if decision == "cancel":
        logger.warning("SAP change %s cancelled by %s", code, caller)
        return f"Cancelled change {code} - nothing was changed in SAP.", {"cancelled": code}

    lines, results = [], []
    for action in record["actions"]:
        logger.warning("SAP change %s confirmed by %s: %s", code, caller, describe_action(action))
        summary, result = await run_structured_action(action)
        ok = "error" not in result
        logger.warning("SAP change %s by %s %s: %s", code, caller, "succeeded" if ok else "FAILED", summary)
        lines.append(f"{'✅' if ok else '❌'} {describe_action(action)} - {summary}")
        results.append(result)
    return "\n".join(lines), {"confirmed": code, "results": results}


def build_action_agent_card(base_url: str):
    return build_agent_card(
        name="Material Action Agent",
        description=(
            "Creates and updates material master records in S/4HANA via the Product Master API "
            "(API_PRODUCT_SRV): create a product master (T-code MM01), update a product master's "
            "status or weight, and update or delete a product master's description (T-code MM02). "
            "Also creates and updates production orders (T-code CO01/CO02) via the Production "
            "Order API (API_PRODUCTION_ORDER_2_SRV)."
        ),
        skill=SKILL,
        base_url=base_url,
    )


class ActionAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        args = get_structured_input(context)

        if args is not None:
            summary, result = await run_structured_action(args)
            await emit_response(event_queue, context, summary, result)
            return

        # Free text (Joule, web UI): never executed directly. A change request is
        # prepared and saved under a one-time code; only "Confirm change <code>"
        # (what the Confirm button sends) executes exactly that saved change.
        text = context.get_user_input()
        decision = parse_decision(text)
        if decision:
            reply, data = await resolve_change(*decision)
        else:
            reply, data = await prepare_change(text)
        await emit_response(event_queue, context, reply, data)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None