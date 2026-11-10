"""Evaluator contract (guide section 14). Evaluators see privileged ground truth; agents never do."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.schemas.task import Task
from app.tools.gateway import ToolCallRecord


@dataclass
class EvalContext:
    task: Task                       # full task, including hidden spec and fixture
    final_state: dict[str, Any]      # privileged environment snapshot
    tool_calls: list[ToolCallRecord]  # every attempted call, including rejected ones
    final_response: str
    stop_reason: str | None
    execution_status: str


@dataclass
class Score:
    evaluator: str
    version: str
    status: str                      # pass | fail | partial | not_applicable | evaluation_error
    dimension: str = "overall"
    applicable: bool = True
    score: float | None = None
    reason_code: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    name: str
    version: str

    def evaluate(self, ctx: EvalContext) -> list[Score]: ...
