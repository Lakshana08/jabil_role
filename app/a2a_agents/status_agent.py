"""Status Agent - the 4 material-status business rules (Technical Spec §6).
All logic lives in core_capabilities/material_status.py + engine/*.py, zero
LLM code; the LLM only picks a tool and narrates its result, never computes."""

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
from app.core_capabilities.material_status import (
    DEFAULT_RESULT_LIMIT,
    query_aging,
    query_machine_head_candidates,
    query_material_status,
)
from app.core_capabilities.paging import PAGE_SIZE, next_page_prompt, skip_from_text
from app.services.ai_core import run_agent

SKILL = AgentSkill(
    id="material_status_check",
    name="Material Status Check",
    description=(
        "Checks material control status, inventory aging, and machine-head "
        "material candidates from SAP source data (PP-134, 14C, COOIS, "
        "Z_AGEDINV). Accepts either structured JSON (check_type + "
        "work_order/material/storage_location) or a plain free-text "
        "question in English or Mandarin. check_type is one of "
        "'non_controlled', 'over_control', 'aging', 'machine_head_review'."
    ),
    tags=["material", "shortage", "aging", "over-control", "machine-head", "non-controlled"],
    examples=[
        "Is material SBC095160140 non-controlled?",
        "Any over-control materials in storage location S201?",
        "Show aged inventory for storage location A301",
        "Show machine-head material needing review",
    ],
    input_modes=["application/json", "text/plain"],
    output_modes=["application/json", "text/plain"],
)


# ---- Tools - business logic only. No LLM code inside any of these. ----
# Each returns ONE page of at most PAGE_SIZE rows (count = true total), never
# the whole list - a broad check can match thousands of rows, which used to
# go straight into the LLM (up to 1000 rows) and overflow its context.


def _skip(skip: int) -> int:
    """The LLM's skip, or - if it passed 0 - the page parsed from the user's
    own "rows 21-40" text, so a next-page click always lands on that page."""
    return skip or requested_skip.get() or 0


def _scope(material: str = "", storage_location: str = "", work_order: str = "") -> str:
    scope = ""
    if work_order:
        scope += f" for work order {work_order}"
    if material:
        scope += f" for material {material}"
    if storage_location:
        scope += f" in storage location {storage_location}"
    return scope


def _with_next_page(result: dict, subject: str) -> dict:
    return {**result, "next_page_prompt": next_page_prompt(subject, result)}



@tool
def check_non_controlled_material(material: str = "", storage_location: str = "", skip: int = 0) -> dict:
    """Check for Non-Controlled Material (无管制料) - WIP with no linked
    production order requirement, or where the requirement is already
    fulfilled. Optionally scoped by material and/or storage_location - but a
    broad call with NO arguments is valid and expected for questions like
    "show non-controlled material in the warehouse"; it returns
    one page of matching records plus the true total count; skip selects the page
    ("rows 21-40" -> skip=20)."""
    result = query_material_status(
        material=material or None,
        storage_location=storage_location or None,
        status_filter="non_controlled",
        skip=_skip(skip),
        limit=PAGE_SIZE,
    )
    return _with_next_page(result, "non-controlled material" + _scope(material, storage_location))


@tool
def check_over_control_material(
    work_order: str = "", material: str = "", storage_location: str = "", skip: int = 0
) -> dict:
    """Check for Over-Control / Over-Issued Material (领超管制) - unrestricted
    stock exceeds the work order's controlled remaining issuance requirement.
    Optionally scoped by work_order, material, and/or storage_location - but a
    broad call with NO arguments is valid for questions like "any over-control
    materials today"; it returns one page of matching records plus the true total
    count; skip selects the page ("rows 21-40" -> skip=20)."""
    result = query_material_status(
        work_order=work_order or None,
        material=material or None,
        storage_location=storage_location or None,
        status_filter="over_control",
        skip=_skip(skip),
        limit=PAGE_SIZE,
    )
    return _with_next_page(result, "over-control material" + _scope(material, storage_location, work_order))


@tool
def check_aging(material: str = "", storage_location: str = "", skip: int = 0) -> dict:
    """Check Aging - inventory that has not moved for more than 14 days
    (fixed threshold). Optionally scoped by material and/or storage_location -
    but a broad call with NO arguments is valid for questions like "show aged
    inventory"; it returns one page of matching records plus the true total count;
    skip selects the page ("rows 21-40" -> skip=20). Rows with no Aging Days value
    in the source data (aging_days_unknown=true) are conservatively included as
    aged rather than silently dropped, since their true age can't be confirmed."""
    result = query_aging(
        material=material or None, storage_location=storage_location or None, skip=_skip(skip), limit=PAGE_SIZE
    )
    return _with_next_page(result, "aged inventory" + _scope(material, storage_location))


@tool
def check_machine_head_review(order_type: str = "ZNPC", skip: int = 0) -> dict:
    """Surface Machine-Head Material (机头料) CANDIDATES for human review only
    - this is a heuristic flag, never a resolved/confirmed status. Returns one
    page of candidates; skip selects the page ("rows 21-40" -> skip=20)."""
    result = query_machine_head_candidates(order_type=order_type, skip=_skip(skip), limit=PAGE_SIZE)
    return _with_next_page(result, f"machine-head material needing review for order type {order_type}")


_TOOLS = [check_non_controlled_material, check_over_control_material, check_aging, check_machine_head_review]

_AGENT_SYSTEM_PROMPT = f"""You answer questions about material control status, inventory aging, and \
machine-head material review using the tools available. Always call the appropriate tool to get the \
computed result before answering - the numbers are already calculated for you by the tool, never \
estimate or invent them yourself.

Broad questions with no specific material/storage location/work order (e.g. "show non-controlled \
material in the warehouse", "any over-control materials today", "show aged inventory") are valid on \
their own - call the tool with no arguments, do not ask the user to narrow it down first.

Large results are paged. Every tool returns ONE page of at most {PAGE_SIZE} rows: results (this page), \
count (the true total), skip (rows before this page), returned, and has_more. When the question asks \
for a list ("show", "list", "any ... today", etc.) and the page contains rows, first say which rows \
are shown out of the total, e.g. "Showing rows 1-20 of 1,284" (first row = skip+1, last row = \
skip+returned), then list EVERY row of the page as a compact table (material, storage location/plant, \
and the status-specific fields such as aging_days or available_to_issue_qty) - never just state a bare \
count instead of the rows. When has_more is true, end by saying they can ask for the next rows or \
narrow by material or storage location. When the user asks for specific rows ("rows 21-40") or "the \
next rows", call the same tool with the same arguments and skip set to the first requested row minus \
1 (rows 21-40 -> skip=20). For questions that ask a \
yes/no or single-fact question (e.g. "is material X non-controlled"), a short 1-3 sentence answer is \
fine - no table needed. Only ask the user for missing information when the question itself refers to a \
specific thing you can't identify (e.g. "is it over-control" with no material named at all).

For machine-head review results, always phrase the answer as flagged candidates needing human review \
- never state it as a confirmed or resolved status."""


def build_status_agent_card(base_url: str):
    return build_agent_card(
        name="Status Agent",
        description=(
            "Checks material control status (non-controlled, over-control), inventory aging, "
            "and machine-head material review candidates."
        ),
        skill=SKILL,
        base_url=base_url,
    )


def _run_structured_check(args: dict) -> tuple[dict, str]:
    """Deterministic dispatch for structured JSON callers - no LLM involved.
    Pages with skip/limit (limit defaults to, and is capped at, DEFAULT_RESULT_LIMIT)."""
    check_type = args.get("check_type", "over_control")
    page = {"skip": int(args.get("skip") or 0), "limit": int(args.get("limit") or DEFAULT_RESULT_LIMIT)}

    if check_type == "non_controlled":
        result = query_material_status(
            material=args.get("material"),
            storage_location=args.get("storage_location"),
            status_filter="non_controlled",
            **page,
        )
    elif check_type == "aging":
        result = query_aging(material=args.get("material"), storage_location=args.get("storage_location"), **page)
    elif check_type == "machine_head_review":
        result = query_machine_head_candidates(order_type=args.get("order_type", "ZNPC"), **page)
    else:  # default: over_control
        result = query_material_status(
            work_order=args.get("work_order"),
            material=args.get("material"),
            storage_location=args.get("storage_location"),
            status_filter="over_control",
            **page,
        )

    summary = f"Found {result['count']} record(s) for check_type '{check_type}'."
    if result["has_more"]:
        summary += (
            f" Showing rows {result['skip'] + 1}-{result['skip'] + result['returned']}; pass skip to get more."
        )
    return result, summary


def _has_review_candidates(tool_outputs: list) -> bool:
    return any(
        isinstance(output, dict) and any(row.get("needs_review") for row in output.get("results", []))
        for output in tool_outputs
    )


def _has_unknown_aging(tool_outputs: list) -> bool:
    return any(
        isinstance(output, dict) and any(row.get("aging_days_unknown") for row in output.get("results", []))
        for output in tool_outputs
    )


class StatusAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        args = get_structured_input(context)

        if args is not None:
            result, summary = await asyncio.to_thread(_run_structured_check, args)
            await emit_response(event_queue, context, summary, result)
            return

        # Free text - the LLM decides which tool(s) to call and narrates the
        # result. No LLM call happens anywhere above this line.
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
        if _has_review_candidates(tool_outputs):
            # Forced server-side, not left to the model's phrasing - Technical
            # Spec §6.4: machine-head results must never be presented as resolved.
            answer += (
                "\n\nNote: this includes machine-head candidates flagged for human review, "
                "not a resolved status."
            )
        if _has_unknown_aging(tool_outputs):
            # Forced server-side for the same reason - a row with no Aging Days
            # value in the source data must not read as a confirmed 14+ day age.
            answer += (
                "\n\nNote: some results have no Aging Days value in the source data - they're "
                "conservatively included as aged pending verification, not a confirmed age."
            )

        data = tool_data(tool_outputs)
        await emit_response(event_queue, context, answer, data)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Single-shot synchronous lookups - nothing runs long enough to cancel mid-flight.
        return None
