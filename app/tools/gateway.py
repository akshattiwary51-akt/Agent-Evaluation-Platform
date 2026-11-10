"""Tool gateway: the single choke point where agent actions are observed and controlled.

Per request: record attempt -> allowlist -> parse/validate args -> (identity comes from the
environment, never from the model) -> business checks -> execute -> record result.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.environments.support.env import SupportEnvironment, ToolError
from app.execution.limits import Limits
from app.telemetry.recorder import TraceRecorder


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OrderArgs(_Args):
    order_id: str


class PolicyArgs(_Args):
    query: str


class RequestApprovalArgs(_Args):
    order_id: str
    amount_minor: int = Field(gt=0)
    reason: str


class IssueRefundArgs(_Args):
    order_id: str
    amount_minor: int = Field(gt=0)
    approval_id: str


class UpdateTicketArgs(_Args):
    ticket_id: str
    status: str
    note: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]


TOOL_SPECS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("get_order", "Look up one of the customer's orders.", OrderArgs),
    ToolSpec("get_return_status", "Get the return status for an order.", OrderArgs),
    ToolSpec("get_refund_status", "Get the refunded amount for an order.", OrderArgs),
    ToolSpec("search_policy", "Search the support policy.", PolicyArgs),
    ToolSpec("request_refund_approval", "Request approval for a refund.", RequestApprovalArgs),
    ToolSpec("issue_refund", "Issue an approved refund.", IssueRefundArgs),
    ToolSpec("update_ticket", "Update the support ticket status and note.", UpdateTicketArgs),
]}


@dataclass
class ToolResult:
    ok: bool
    content: dict[str, Any]
    error_code: str | None = None

    def as_tool_message(self, tool_call_id: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(self.content)}


@dataclass
class ToolCallRecord:
    """Attempted call, kept separately from the environment's record of effects."""
    tool_call_id: str
    tool_name: str
    raw_arguments: Any
    arguments: dict[str, Any] | None
    outcome: str  # ok | rejected | error
    reason_code: str | None = None
    result: dict[str, Any] | None = None
    duration_ms: float = 0.0


@dataclass
class ToolGateway:
    env: SupportEnvironment
    recorder: TraceRecorder
    limits: Limits
    log: list[ToolCallRecord] = field(default_factory=list)

    def visible_schemas(self) -> list[dict[str, Any]]:
        """Schemas to show a model. Never includes identity, fixtures, or grading data."""
        return [{"name": s.name, "description": s.description,
                 "parameters": s.args_model.model_json_schema()} for s in TOOL_SPECS.values()]

    async def execute(self, tool_name: str, raw_arguments: Any, tool_call_id: str) -> ToolResult:
        self.limits.register_tool_call()  # may raise LimitExceeded; runner handles it
        started = time.monotonic()
        self.recorder.emit("tool_requested", {"tool_call_id": tool_call_id,
                                              "tool": tool_name, "arguments": raw_arguments})

        def finish(record: ToolCallRecord, result: ToolResult, event: str) -> ToolResult:
            record.duration_ms = (time.monotonic() - started) * 1000
            record.result = result.content
            self.log.append(record)
            self.recorder.emit(event, {"tool_call_id": tool_call_id, "tool": tool_name,
                                       "reason_code": record.reason_code,
                                       "result": result.content},
                               status=record.outcome, duration_ms=record.duration_ms)
            return result

        record = ToolCallRecord(tool_call_id, tool_name, raw_arguments, None, "rejected")

        spec = TOOL_SPECS.get(tool_name)
        if spec is None:
            record.reason_code = "UNKNOWN_TOOL"
            return finish(record, ToolResult(False, {"error": f"Unknown tool {tool_name}"},
                                             "UNKNOWN_TOOL"), "tool_rejected")

        try:
            payload = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            args = spec.args_model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            record.reason_code = "SCHEMA_INVALID"
            return finish(record, ToolResult(False, {"error": "Invalid arguments.",
                                                     "detail": str(exc)[:300]},
                                             "SCHEMA_INVALID"), "tool_rejected")

        record.arguments = args.model_dump()
        handler: Callable[..., dict[str, Any]] = getattr(self.env, tool_name)
        try:
            content = handler(**record.arguments)
        except ToolError as exc:
            record.reason_code = exc.code
            return finish(record, ToolResult(False, {"error": exc.message}, exc.code),
                          "tool_rejected")
        record.outcome = "ok"
        return finish(record, ToolResult(True, content), "tool_finished")
