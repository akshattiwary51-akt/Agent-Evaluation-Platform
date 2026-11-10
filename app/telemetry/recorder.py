"""Append-only trace recorder. Events carry the fields from guide section 19."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

REDACTED_KEYS = {"api_key", "authorization", "password", "secret", "token"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in REDACTED_KEYS else redact(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


@dataclass
class TraceEventData:
    event_id: str
    attempt_id: str
    sequence_number: int
    trace_id: str
    span_id: str
    parent_span_id: str | None
    event_type: str
    timestamp: datetime
    duration_ms: float | None
    status: str | None
    payload: dict[str, Any]


@dataclass
class TraceRecorder:
    attempt_id: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list[TraceEventData] = field(default_factory=list)
    # Chargeable model usage lives here so it survives aborted attempts (limits, outages).
    model_calls: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event_type: str, payload: dict[str, Any] | None = None, *,
             status: str | None = None, duration_ms: float | None = None,
             span_id: str | None = None, parent_span_id: str | None = None) -> TraceEventData:
        event = TraceEventData(
            event_id=uuid.uuid4().hex,
            attempt_id=self.attempt_id,
            sequence_number=len(self.events),
            trace_id=self.trace_id,
            span_id=span_id or uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            status=status,
            payload=redact(payload or {}),
        )
        self.events.append(event)
        return event
