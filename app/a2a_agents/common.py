"""Shared helpers for the three A2A agent executors.

Input contract: if the caller sends a JSON DataPart matching the
capability's parameter shape (e.g. a Joule Studio skill invocation with
pre-extracted fields), dispatch directly to core_capabilities - no LLM
involved. If the caller sends plain free text instead, route through
services/ai_core.run_agent() with the capability's functions bound as
LangChain tools, and let the model decide what to call.
"""

import asyncio
import re
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any

from a2a.helpers import get_data_parts, new_data_part, new_message, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Task, TaskState, TaskStatus
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol

# Page the caller asked for in free text ("..., rows 21-40" -> 20), set by
# the executor before run_agent(). Paged tools fall back to it when the LLM
# passes skip=0, so a next-page click never depends on the model doing the
# row arithmetic. LangChain copies the context into the tool thread.
requested_skip: ContextVar[int | None] = ContextVar("requested_skip", default=None)

# True while the Assistant Agent (Teams) is answering: the Teams bot turns
# Markdown tables into real Adaptive Card tables, so they are kept. Every other
# caller (Joule, web UI) shows the text as-is, where a Markdown table appears as
# raw "| a | b |" lines - so for them emit_response rewrites tables into labelled
# plain text (see tables_to_text).
keep_markdown_tables: ContextVar[bool] = ContextVar("keep_markdown_tables", default=False)

# "Generate a production order report at plant 1710 in Excel format" - the texts our CSV / Excel /
# PDF buttons send. Joule sometimes keeps such a click in the Planner scenario; those
# agents then hand it to the Report agent instead of refusing.
REPORT_REQUEST_RE = re.compile(r"^\s*generate\s+an?\b.*\breport\b", re.IGNORECASE)

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
# Fields per line in a multi-record entry - keeps lines short enough to read in a chat bubble.
_FIELDS_PER_LINE = 3


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_as_text(lines: list[str]) -> list[str]:
    rows = [_cells(line) for line in lines]
    rows = [row for row in rows if not all(_SEPARATOR_CELL.match(c) for c in row if c)]
    if len(rows) < 2:
        return lines
    header, records = rows[0], rows[1:]

    def pairs(record: list[str]) -> list[tuple[str, str]]:
        return [(h, v) for h, v in zip(header, record) if v not in ("", "-", "None", "null")]

    if len(records) == 1:
        # One record: a vertical "Field: value" list.
        return [f"• {h}: {v}" for h, v in pairs(records[0])]

    out: list[str] = []
    for number, record in enumerate(records, start=1):
        fields = pairs(record)
        if not fields:
            continue
        (first_h, first_v), rest = fields[0], fields[1:]
        out.append(f"{number}. {first_h}: {first_v}")
        for i in range(0, len(rest), _FIELDS_PER_LINE):
            out.append("    " + "  |  ".join(f"{h}: {v}" for h, v in rest[i : i + _FIELDS_PER_LINE]))
        out.append("")
    while out and not out[-1]:
        out.pop()
    return out


def tables_to_text(text: str) -> str:
    """Rewrites every Markdown table in text into clean labelled plain text, e.g.

        | Order | Material | Plant |        1. Order: 1000000
        |-------|----------|-------|   ->       Material: MZ-FG-R300  |  Plant: 1710
        | 1000000 | MZ-FG-R300 | 1710 |

    (a single record becomes a "• Field: value" list). Text outside tables is untouched."""
    out: list[str] = []
    table: list[str] = []
    for line in text.splitlines():
        if _TABLE_LINE.match(line):
            table.append(line)
            continue
        if table:
            out.extend(_table_as_text(table))
            table = []
        out.append(line)
    if table:
        out.extend(_table_as_text(table))
    return "\n".join(out)


class BoundedTaskStore(InMemoryTaskStore):
    """InMemoryTaskStore that keeps only the newest max_tasks tasks.

    The stock store never evicts, so every reply (with its data rows) stays
    in memory for the life of the process - under steady traffic that grows
    until Cloud Foundry OOM-kills the instance. Safe to evict here: every
    reply is a single already-completed Task (see emit_response) that no
    caller ever reads back."""

    def __init__(self, max_tasks: int = 500) -> None:
        super().__init__()
        self._max_tasks = max_tasks
        self._order: OrderedDict[str, ServerCallContext] = OrderedDict()
        self._order_lock = asyncio.Lock()

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await super().save(task, context)
        async with self._order_lock:
            self._order[task.id] = context
            self._order.move_to_end(task.id)
            evicted = []
            while len(self._order) > self._max_tasks:
                evicted.append(self._order.popitem(last=False))
        for task_id, task_context in evicted:
            await super().delete(task_id, task_context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        async with self._order_lock:
            self._order.pop(task_id, None)
        await super().delete(task_id, context)


def get_structured_input(context: RequestContext) -> dict[str, Any] | None:
    """Returns the JSON DataPart if the caller sent one, else None - None
    means the caller sent free text and should be routed through the AI
    Core tool-calling agent instead.
    """
    data_parts = get_data_parts(context.message.parts)
    if data_parts and isinstance(data_parts[0], dict):
        return data_parts[0]
    return None


def requester_from(context: RequestContext) -> dict[str, Any] | None:
    """The asking user, if the caller sent one in the message metadata as
    {"requested_by": {"user_key", "name", "email"}} - the Teams bot does, so a
    prepared SAP change can only be confirmed by the same user. Joule doesn't."""
    metadata = context.message.metadata if context.message else None
    if not metadata:
        return None
    if not isinstance(metadata, dict):
        from google.protobuf.json_format import MessageToDict

        metadata = MessageToDict(metadata)
    requested_by = metadata.get("requested_by")
    return requested_by if isinstance(requested_by, dict) else None


def tool_data(tool_outputs: list[Any]) -> dict[str, Any] | None:
    """Data part for a free-text (LLM) reply. The follow-up prompts are lifted
    to the top level, plus one "followup" flag naming which buttons apply
    ("next_and_reports" | "reports_only" | "next_only" | None) - so every Joule
    button group needs only a single flat comparison. Joule's expression
    language proved unreliable on nested or combined conditions."""
    if not tool_outputs:
        return None
    outputs = [t for t in tool_outputs if isinstance(t, dict)]
    next_prompt = next((t["next_page_prompt"] for t in outputs if t.get("next_page_prompt")), None)
    # Report buttons only for a result that has rows - no "report of nothing".
    report = next((t for t in outputs if t.get("report_prompt_csv") and (t.get("count") or 0) > 0), None)
    if next_prompt and report:
        followup = "next_and_reports"
    elif report:
        followup = "reports_only"
    elif next_prompt:
        followup = "next_only"
    else:
        followup = None
    data = {"tool_results": tool_outputs, "next_page_prompt": next_prompt, "followup": followup}
    if report:
        data.update({key: report[key] for key in ("report_prompt_csv", "report_prompt_excel", "report_prompt_pdf")})
    return data


async def emit_response(
    event_queue: EventQueue,
    context: RequestContext,
    summary_text: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Completes the task with a final agent message, in a single event.

    Joule's agent-request action doesn't treat a standalone A2A Message as a
    finished turn - it needs a Task. But enqueueing a submitted Task and then
    a *separate* TaskStatusUpdateEvent(completed) right after (e.g. via
    TaskUpdater) lets Joule capture the first (submitted) event as the
    message/send result before the completed update is ever applied -
    confirmed via Joule's own debug trace, which showed
    agentResult.body.status == {"state": "submitted"} with no message at
    all. Since this app is always single-shot (the full answer is already
    known by the time this runs, nothing genuinely continues in the
    background), enqueue one Task that's already in a terminal state -
    no submitted-then-completed transition to race against.
    """
    if not keep_markdown_tables.get():
        summary_text = tables_to_text(summary_text)
    parts = [new_text_part(summary_text)]
    if data is not None:
        parts.append(new_data_part(data))
    agent_message = new_message(parts, context_id=context.context_id, task_id=context.task_id)
    task = Task(
        id=context.task_id,
        context_id=context.context_id,
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED, message=agent_message),
    )
    await event_queue.enqueue_event(task)


def build_agent_card(*, name: str, description: str, skill: AgentSkill, base_url: str) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        default_input_modes=["application/json", "text/plain"],
        default_output_modes=["application/json", "text/plain"],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False, extended_agent_card=False),
        supported_interfaces=[
            AgentInterface(
                protocol_binding=TransportProtocol.JSONRPC,
                url=base_url,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        skills=[skill],
    )
