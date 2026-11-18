import asyncio

from langchain_core.tools import tool

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentSkill

from app.a2a_agents.common import build_agent_card, emit_response, get_structured_input
from app.core_capabilities._s4_apis import BY_NAME, FILTER_ALIASES, FILTER_NAMES, describe_scope, source_catalog
from app.core_capabilities.report import generate_material_report
from app.services.ai_core import run_agent
from app.services.report_builder import SUPPORTED_FORMATS
from app.services.s4_client import S4ClientError

SKILL = AgentSkill(
    id="generate_material_report",
    name="Generate Material Report",
    description=(
        "Builds a downloadable report (PDF/Excel/CSV) from any Smart Material CDS view in S/4HANA - "
        "material master and plant/valuation/BOM data, stock, inventory, aged inventory, WIP issuable, "
        "over-control, putaway-pending, -S materials, work-order components, 14C issuance, order status, "
        "production orders (components/operations/statuses) and shortages. "
        f"report_type selects the source: one of {', '.join(BY_NAME)}. Optional filters (material, plant, "
        "storage_location, order, component, status, ...), conditions (e.g. ['AgingDays > 90']) and order_by "
        "scope the rows; the older identifiers (main key) / plant inputs still work. Accepts structured JSON or "
        "free text. Returns a download_url rather than the file itself."
    ),
    tags=["s4hana", "material", "production-order", "inventory", "shortage", "report"],
    examples=[
        "Generate an over-control material report in storage location A301 in Excel format",
        "Export the aged inventory older than 90 days to CSV",
        "Give me a PDF of component shortages for order 121060001",
    ],
    input_modes=["application/json", "text/plain"],
    output_modes=["application/json", "text/plain"],
)


@tool
def generate_report(
    report_type: str,
    report_format: str = "pdf",
    filters: dict[str, str] | None = None,
    conditions: list[str] | None = None,
    order_by: str = "",
) -> dict:
    """Generate a downloadable report (all matching rows) of one S/4HANA data source.
    report_type: a source name from your instructions. report_format: 'pdf', 'xlsx' or 'csv'.
    filters: simple filters the source supports, e.g. {"storage_location": "A301"}; values may be
    comma-separated for "any of". conditions: e.g. ["AgingDays > 90"]. order_by: e.g. "AgingDays desc"."""
    try:
        return generate_material_report(
            report_type=report_type,
            report_format=report_format,
            filters=filters,
            conditions=conditions,
            order_by=order_by or None,
        )
    except (S4ClientError, ValueError) as exc:
        # Back to the LLM so it can fix the source/filter and retry, instead of aborting the turn.
        return {"error": str(exc), "report_type": report_type}


_TOOLS = [generate_report]

# Deliberately tells the model NOT to state the download link itself - the
# literal URL is always appended by the executor after the tool runs, so a
# model paraphrasing/garbling it in its answer can't break the actual link.
#
# Deliberately does NOT say "default to X if unclear" - that previously made
# the model silently generate an unfiltered report for unrelated questions
# and present it as if it answered the question.
_AGENT_SYSTEM_PROMPT = f"""You generate downloadable S/4HANA reports with the generate_report tool. \
report_format must be one of {sorted(SUPPORTED_FORMATS)} ("Excel" -> xlsx). Pick report_type from these \
data sources, and pass the filters / conditions / sort the request asks for (none = all rows):

{source_catalog()}

Filter names: {", ".join(FILTER_NAMES)}.

Choosing the source: "aged"/"aging" -> aged_inventory (aged = AgingDays > 14); "over-issued"/"over-control" -> \
over_control; "non-controlled" -> wip_issuable with status NC; "missing parts"/order shortage -> \
work_order_shortage; short components -> component_shortage (really short = ShortageQty > 0); "stock" -> \
material_stock; stock value -> inventory; BOM -> material_bom; "14C" -> component_issuance; NMVT / excluded \
orders -> order_status. Messages like "Generate a <label> report for ... where ... in CSV format" come from \
report buttons - map the label back to its source and keep every filter and condition in them.

Use a filter only if the chosen source lists it; otherwise use a condition on a key field. If the tool returns \
"error", fix the call and try once more; if it still fails, say what went wrong. If the request clearly doesn't \
match any source, do not call the tool - reply that this isn't available from the Report Agent, without naming \
any other agent.

After calling the tool, briefly describe what was generated (report type, filters and row count) in one short \
sentence - do NOT include any URL or link in your answer, that will be added separately. If the tool result has \
a "note" (row limits for very large data or for PDF), always pass it on to the user."""


def build_report_agent_card(base_url: str):
    return build_agent_card(
        name="Material Report Agent",
        description="Generates downloadable reports (PDF/Excel/CSV) of any Smart Material CDS view in S/4HANA.",
        skill=SKILL,
        base_url=base_url,
    )


class ReportAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        args = get_structured_input(context)

        if args is not None:
            report_type = args.get("report_type", "material_stock")
            identifiers = args.get("identifiers")
            plant = args.get("plant")
            report_format = args.get("report_format", "pdf")
            filters = dict(args.get("filters") or {})
            for name in list(FILTER_NAMES) + list(FILTER_ALIASES):
                if name != "plant" and args.get(name) and name not in filters:
                    filters[name] = args[name]

            if report_type not in BY_NAME:
                await emit_response(
                    event_queue,
                    context,
                    f"Unknown report_type '{report_type}', expected one of {', '.join(BY_NAME)}.",
                    {"error": "unknown_report_type"},
                )
                return

            if report_format not in SUPPORTED_FORMATS:
                await emit_response(
                    event_queue,
                    context,
                    f"Unsupported report_format '{report_format}', expected one of {sorted(SUPPORTED_FORMATS)}.",
                    {"error": "unsupported_format"},
                )
                return

            try:
                result = await asyncio.to_thread(
                    generate_material_report,
                    report_type=report_type,
                    identifiers=identifiers,
                    plant=plant,
                    report_format=report_format,
                    filters=filters,
                    conditions=args.get("conditions") or None,
                    order_by=args.get("order_by"),
                )
            except S4ClientError as exc:
                await emit_response(
                    event_queue, context, f"Couldn't build report from S/4HANA data: {exc}", {"error": "s4_error"}
                )
                return

            scope = describe_scope(result["filters"], result["conditions"])
            summary = f"{result['label'].capitalize()} report{scope} ready ({result['row_count']} rows): {result['download_url']}"
            if result.get("note"):
                summary += f" Note: {result['note']}"
            await emit_response(event_queue, context, summary, result)
            return

        # Free text - let the AI Core tool-calling agent decide what to call.
        text = context.get_user_input()
        agent_result = await run_agent(_AGENT_SYSTEM_PROMPT, _TOOLS, text)
        if agent_result is None:
            await emit_response(
                event_queue,
                context,
                "I couldn't process that request right now (AI Core unavailable).",
                {"error": "ai_core_unavailable"},
            )
            return

        answer, tool_outputs = agent_result
        download_url = next((t.get("download_url") for t in tool_outputs if isinstance(t, dict) and t.get("download_url")), None)
        final_text = f"{answer} {download_url}" if download_url else answer
        data = {"tool_results": tool_outputs} if tool_outputs else None
        await emit_response(event_queue, context, final_text, data)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None