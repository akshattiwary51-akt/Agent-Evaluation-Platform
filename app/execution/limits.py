"""Per-attempt limits. Durations use a monotonic clock; timestamps elsewhere are UTC."""
from __future__ import annotations

import time
from typing import Callable

from app.schemas.task import LimitsSpec


class LimitExceeded(Exception):
    """Raised when an attempt exceeds a configured limit. `kind` maps to an execution status."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        # "timed_out" | "budget_exceeded" | "cancelled" | "lease_lost"
        self.kind = kind
        self.detail = detail


class Limits:
    def __init__(self, spec: LimitsSpec, should_stop: Callable[[], str | None] | None = None):
        # should_stop returns None to continue, or "cancelled" / "lease_lost" to abort before the
        # next model call or tool execution (the guide: cancellation stops new tool execution).
        self._should_stop = should_stop
        self.max_model_turns = spec.max_model_turns
        self.max_tool_calls = spec.max_tool_calls
        self.timeout_seconds = spec.timeout_seconds
        self._deadline = time.monotonic() + spec.timeout_seconds
        self.tool_calls = 0
        self.model_turns = 0

    def remaining_seconds(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def check_deadline(self) -> None:
        if self._should_stop is not None:
            reason = self._should_stop()
            if reason:
                raise LimitExceeded(reason, f"stopped: {reason}")
        if self.remaining_seconds() <= 0:
            raise LimitExceeded("timed_out", f"exceeded {self.timeout_seconds}s")

    def register_tool_call(self) -> None:
        self.check_deadline()
        if self.tool_calls >= self.max_tool_calls:
            raise LimitExceeded("budget_exceeded", f"max_tool_calls={self.max_tool_calls}")
        self.tool_calls += 1

    def register_model_turn(self) -> None:
        self.check_deadline()
        if self.model_turns >= self.max_model_turns:
            raise LimitExceeded("budget_exceeded", f"max_model_turns={self.max_model_turns}")
        self.model_turns += 1
