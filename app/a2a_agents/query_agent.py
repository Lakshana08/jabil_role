import asyncio

from langchain_core.tools import tool

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentSkill

from app.a2a_agents.common import (
    REPORT_REQUEST_RE,
    build_agent_card,
    emit_response,
    get_structured_input,
    requested_skip,
    tool_data,
)
from app.core_capabilities.paging import PAGE_SIZE, next_page_prompt, skip_from_text
from app.core_capabilities.query import (
    QUERY_ROW_LIMIT,
    count_all_records,
    query_material_master,
    query_material_serial_numbers,
    query_material_stock,
    query_production_order,
)
from app.services.ai_core import run_agent
from app.services.odata_utils import humanize_odata_dates
from app.services.s4_client import S4ClientError

VALID_QUERY_TYPES = {"material_master", "material_stock", "production_order", "material_serial_number"}

SKILL = AgentSkill(
    id="query_material",
    name="Query Material",
    description=(
        "Looks up data in S/4HANA. Accepts either structured JSON "
        "(query_type + product/material/plant/production_order/serial_number, or "
        "count_only: true for a total count) or a plain free-text question. query_type "
        "selects the source: "
        "'material_master' (MM03 - needs product), "
        "'material_stock' (MMBE - needs material, optional plant), "
        "'production_order' (COOIS - production_order and/or material and/or plant), "
        "'material_serial_number' (MMBE serialized stock - needs material, optional plant/serial_number). "
        "Defaults to 'material_stock' if query_type is omitted. Also answers 'how many X are "
        "there in total' for any of the 4 sources, with no identifier needed - a real, unfiltered "
        "total count from S/4, not just the count of rows returned in one page."
    ),
    tags=["s4hana", "material", "production-order", "query"],
    examples=[
        "What's the stock for material MAT-1000?",
        "Look up material master data for product MAT-2000",
        "Show me production order 60001234",
        "List serial numbers for material MAT-1000 at plant 1010",
        "How many product masters are there in total?",
        "How many production orders are there?",
    ],
    input_modes=["application/json", "text/plain"],
    output_modes=["application/json", "text/plain"],
)


# Free-text tools fetch one PAGE_SIZE page per call, never query.py's full
# QUERY_ROW_LIMIT (200) - 200 full-width rows fed into the LLM's tool-calling
# loop blew its context window (confirmed live: 200 raw production order rows
# alone hit 141k tokens, over the model's 128k limit). count stays the real
# S/4 total, and the rest is reachable page by page via next_page_prompt.


_REPORT_TYPE_LABELS = {
    "material_master": "material master",
    "material_stock": "material stock",
    "production_order": "production order",
    "material_serial_number": "material serial number",
}


def _page_subject(report_type: str | None, result: dict) -> str | None:
    """What the next-page prompt asks for again, e.g. "production orders for
    plant 1710" - phrased so Joule routes it back to this same scenario."""
    if report_type == "production_order":
        subject = "production orders"
        if result.get("production_order"):
            subject += f" for production order {result['production_order']}"
        if result.get("material"):
            subject += f" for material {result['material']}"
    elif report_type == "material_stock":
        subject = f"material stock for material {result.get('material')}"
    elif report_type == "material_serial_number":
        subject = f"serial numbers for material {result.get('material')}"
        if result.get("serial_number"):
            subject += f" serial number {result['serial_number']}"
    else:
        return None
    if result.get("plant"):
        subject += f" at plant {result['plant']}"
    return subject


def _add_report_prompts(result: dict) -> dict:
    """Pre-computes the exact free-text prompts a caller would send to the
    Report Agent to turn this result into a downloadable file, as plain
    strings - e.g. report_prompt_csv = "Generate a material stock report for
    TG20 in CSV format". Mirrors the frontend's extractReportParams() JS
    logic, done once here in Python (simple, deterministic, testable)
    instead of in a Joule capability's SpEL, which confirmed live cannot
    reliably evaluate branching logic beyond a single flat property lookup -
    every attempt at nested ternaries there either silently returned null or
    crashed the rest of the action group. A consumer like a Joule button
    only ever needs a plain property reference to these fields now, no
    conditional logic of its own."""
    if "query_type" in result:
        report_type = result["query_type"]
        identifier = None
    elif "product" in result:
        report_type = "material_master"
        identifier = result.get("product")
    elif "production_order" in result:
        report_type = "production_order"
        identifier = result.get("production_order") or result.get("material")
    elif "serial_number" in result:
        report_type = "material_serial_number"
        identifier = result.get("material")
    elif "material" in result:
        report_type = "material_stock"
        identifier = result.get("material")
    else:
        report_type = None
        identifier = None

    label = _REPORT_TYPE_LABELS.get(report_type)
    base = f"Generate a {label} report" if label else "Generate a report"
    if identifier:
        base += f" for {identifier}"
    # Without the plant, "production orders for plant 1710" turned into a
    # report of every plant's production orders.
    if result.get("plant") and report_type != "material_master":
        base += f" at plant {result['plant']}"

    # Same idea for paging: a plain ready-made string ("..., rows 21-40"),
    # present only when more rows exist, so a "Show next" button needs no
    # logic of its own. The report prompts above always cover ALL rows.
    subject = _page_subject(report_type, result)
    return {
        **result,
        "report_prompt_csv": f"{base} in CSV format",
        "report_prompt_excel": f"{base} in Excel format",
        "report_prompt_pdf": f"{base} in PDF format",
        "next_page_prompt": next_page_prompt(subject, result) if subject else None,
    }


def _for_llm(result: dict) -> dict:
    return _add_report_prompts(humanize_odata_dates(result))


def _skip(skip: int) -> int:
    """The LLM's skip, or - if it passed 0 - the page parsed from the user's
    own "rows 21-40" text, so a next-page click always lands on that page."""
    return skip or requested_skip.get() or 0


@tool
def lookup_material_stock(material: str, plant: str = "", skip: int = 0) -> dict:
    """Look up stock overview (MMBE) for a material, optionally scoped to a plant. Use this for stock/inventory questions.
    Returns one page of rows; skip selects the page ("rows 21-40" -> skip=20)."""
    return _for_llm(query_material_stock(material=material, plant=plant or None, skip=_skip(skip), top=PAGE_SIZE))


@tool
def lookup_material_master(product: str) -> dict:
    """Look up material/product master data (MM03) for a material or product number."""
    return _for_llm(query_material_master(product=product))


@tool
def lookup_production_order(production_order: str = "", material: str = "", plant: str = "", skip: int = 0) -> dict:
    """Look up production order info (COOIS), filterable by production_order, material, and/or plant.
    Returns one page of rows; skip selects the page ("rows 21-40" -> skip=20)."""
    return _for_llm(
        query_production_order(
            production_order=production_order or None,
            material=material or None,
            plant=plant or None,
            skip=_skip(skip),
            top=PAGE_SIZE,
        )
    )


@tool
def lookup_material_serial_number(material: str, plant: str = "", serial_number: str = "", skip: int = 0) -> dict:
    """Look up serialized stock (MMBE, serial-number breakdown) for a material, optionally scoped to a plant and/or serial number.
    Returns one page of rows; skip selects the page ("rows 21-40" -> skip=20)."""
    return _for_llm(
        query_material_serial_numbers(
            material=material,
            plant=plant or None,
            serial_number=serial_number or None,
            skip=_skip(skip),
            top=PAGE_SIZE,
        )
    )


@tool
def count_records(query_type: str) -> dict:
    """Get the total, unfiltered count of records for a query source - use this for "how many X
    are there in total" style questions, where the user isn't asking about one specific
    material/product/order. query_type is one of 'material_master', 'material_stock',
    'production_order', 'material_serial_number'. Do not use the lookup_* tools for this - those
    require a specific identifier and only return up to the row cap, not a true total."""
    return _add_report_prompts(count_all_records(query_type=query_type))


_TOOLS = [
    lookup_material_stock,
    lookup_material_master,
    lookup_production_order,
    lookup_material_serial_number,
    count_records,
]

_AGENT_SYSTEM_PROMPT = f"""You answer questions about S/4HANA material master data, stock levels, \
production orders, and serialized stock (serial numbers) using the tools available - nothing else. \
Always call the appropriate tool to get real data before answering - never invent numbers or data. \
Keep answers short and factual (1-3 sentences, plus a compact table when listing rows). Only ask the user to clarify when a tool's one truly \
required identifier is missing: material for stock/serial-number lookups, product for material master. \
Plant and serial_number are optional filters on every tool that accepts them - if the user doesn't \
mention one, call the tool without it and return the unfiltered results; never ask for a plant code \
or serial number before running a lookup. production_order lookups need at least one of \
production_order/material/plant, not all three.

If the user asks "how many X are there" / "total number of X" for material masters, stock records, \
production orders, or serial numbers - with no specific material/product/order given - call \
count_records with the matching query_type instead of a lookup_* tool or refusing. This works with \
no identifier at all; never ask the user for one first.

Large results are paged. Every lookup_* tool returns ONE page of at most {PAGE_SIZE} rows: \
results (this page), count (the true total in S/4HANA), skip (rows before this page), returned, and \
has_more. Always tell the user which rows they are seeing out of the total, e.g. "Showing rows 1-20 \
of 3,412 production orders" (first row = skip+1, last row = skip+returned), then list the page rows \
as a compact table of the most relevant fields. Never claim the page is the full result when has_more \
is true. When has_more is true, end by saying they can ask for the next rows or download the full list \
as a CSV/Excel/PDF report. When the user asks for specific rows ("rows 21-40") or "the next rows", call \
the same tool with the same filters and skip set to the first requested row minus 1 (rows 21-40 -> \
skip=20).

Do NOT treat questions about over-control/over-issued status, non-controlled material, aging, \
machine-head material review, report generation, or ZPL pull-list actions as needing more parameters \
- those are not S/4HANA lookups this agent performs at all. If asked about any of them, do not call \
a tool or ask for material numbers - just say plainly that this isn't available from the Query Agent. \
Do not name any other agent or say who handles it.

Accuracy rules - confirmed live failure modes, follow these exactly:
1. Every date, number, code, or status value you state must be copied verbatim from a field that \
is literally present in the tool's JSON result - never invent or guess a value, even a plausible \
one. Field names describe their own content plainly (e.g. LastChangeDate is literally when the \
record was last changed, CreationDate is when it was created, LastChangedByUser is who last \
changed it) - when a question clearly maps to a field like this, use its value directly and \
confidently; don't refuse just because the mapping wasn't spelled out for you. Only say you don't \
have the information when no field in the result plausibly answers the question at all.
2. Never invent or assume a business meaning for a coded/technical field value (e.g. a special \
stock indicator, a status code) unless that meaning is explicitly present in the data itself (e.g. \
a description field). If asked what a code means and no description is provided, report the raw \
code and say its business meaning isn't available from this data - do not guess a plausible-sounding \
interpretation, even a common textbook one, since the actual configured meaning is client-specific.
3. The word "entries" (or "records") in a question always means the number of items in the tool's \
results array - it does NOT mean the number of distinct values within one field of those items, \
even when the question also says "different" (e.g. "how many different storage location entries \
exist" asks for the array length, not the count of distinct storage location codes). Confirmed \
live failure: 8 stock rows for one material, split across 2 storage location codes, was wrongly \
answered as "2" - the correct answer is 8. Only answer with a count of distinct values if the \
question asks for that specifically (e.g. "how many different storage locations are there", with \
no word like "entries"/"records" attached to it)."""


def build_query_agent_card(base_url: str):
    return build_agent_card(
        name="Material Query Agent",
        description=(
            "Answers questions about material master data, stock levels, production orders, "
            "and serialized stock in S/4HANA."
        ),
        skill=SKILL,
        base_url=base_url,
    )


def _page_note(result: dict) -> str:
    if not result.get("has_more"):
        return ""
    first = result["skip"] + 1
    return f" Showing rows {first}-{result['skip'] + result['returned']}; pass skip to get more."


def _run_structured_query(args: dict) -> tuple[dict, str]:
    """Deterministic dispatch for structured JSON callers - no LLM involved."""
    query_type = args.get("query_type", "material_stock")

    if query_type == "material_master":
        product = args.get("product") or args.get("material_number")
        result = query_material_master(product=product)
        summary = (
            f"Found {result['count']} material master record(s) for product {product}."
            if result["results"]
            else f"No material master data found for product {product}."
        )
        return result, summary

    page = {"skip": int(args.get("skip") or 0), "top": int(args.get("top") or QUERY_ROW_LIMIT)}

    if query_type == "production_order":
        result = query_production_order(
            production_order=args.get("production_order"),
            material=args.get("material") or args.get("material_number"),
            plant=args.get("plant"),
            **page,
        )
        summary = f"Found {result['count']} production order record(s)." + _page_note(result)
        return result, summary

    if query_type == "material_serial_number":
        material = args.get("material") or args.get("material_number")
        result = query_material_serial_numbers(
            material=material, plant=args.get("plant"), serial_number=args.get("serial_number"), **page
        )
        summary = (
            f"Found {result['count']} serial number record(s) for material {material}." + _page_note(result)
            if result["results"]
            else f"No serial number records found for material {material}."
        )
        return result, summary

    # default: material_stock
    material = args.get("material") or args.get("material_number")
    result = query_material_stock(material=material, plant=args.get("plant"), **page)
    summary = (
        f"Found {result['count']} stock record(s) for material {material}." + _page_note(result)
        if result["results"]
        else f"No stock records found for material {material}."
    )
    return result, summary


class QueryAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        args = get_structured_input(context)

        if args is not None:
            query_type = args.get("query_type", "material_stock")
            if query_type not in VALID_QUERY_TYPES:
                await emit_response(
                    event_queue,
                    context,
                    f"Unknown query_type '{query_type}', expected one of {sorted(VALID_QUERY_TYPES)}.",
                    {"error": "unknown_query_type"},
                )
                return

            if args.get("count_only"):
                try:
                    result = await asyncio.to_thread(count_all_records, query_type=query_type)
                except S4ClientError as exc:
                    await emit_response(
                        event_queue, context, f"Couldn't reach S/4HANA: {exc}", {"error": "s4_error"}
                    )
                    return
                summary = f"Total {query_type} count: {result['count']}."
                await emit_response(event_queue, context, summary, result)
                return

            required_field = {"material_master": "product", "production_order": None}.get(query_type, "material")
            identifier_present = bool(
                required_field and (args.get(required_field) or args.get("material_number"))
            )
            if required_field and not identifier_present:
                await emit_response(
                    event_queue,
                    context,
                    f"I need a '{required_field}' to run a {query_type} lookup.",
                    {"error": "missing_required_field", "query_type": query_type},
                )
                return

            try:
                result, summary = await asyncio.to_thread(_run_structured_query, args)
            except S4ClientError as exc:
                await emit_response(event_queue, context, f"Couldn't reach S/4HANA: {exc}", {"error": "s4_error"})
                return

            await emit_response(event_queue, context, summary, result)
            return

        # Free text - let the AI Core tool-calling agent decide what to call.
        text = context.get_user_input()
        if REPORT_REQUEST_RE.match(text):
            # A report button click that Joule routed here - hand it to the Report agent.
            from app.a2a_agents.report_agent import ReportAgentExecutor

            await ReportAgentExecutor().execute(context, event_queue)
            return
        requested_skip.set(skip_from_text(text))
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
        data = tool_data(tool_outputs)
        await emit_response(event_queue, context, answer, data)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Single-shot synchronous lookups - nothing runs long enough to cancel mid-flight.
        return None