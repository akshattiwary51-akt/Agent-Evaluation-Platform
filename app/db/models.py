"""SQLAlchemy models for the core tables in the guide (section 16).

JSONB is used on PostgreSQL; a plain JSON type is used elsewhere so unit tests can
run against SQLite without a database server.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JsonType = JSON().with_variant(JSONB(), "postgresql")


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    version: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict] = mapped_column(JsonType, default=dict)  # splits, counts
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TaskVersion(Base):
    __tablename__ = "task_versions"
    __table_args__ = (UniqueConstraint("dataset_version_id", "task_id", "version"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"))
    task_id: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(32), default="development")
    agent_input: Mapped[dict] = mapped_column(JsonType)  # agent-visible view
    environment_fixture: Mapped[dict] = mapped_column(JsonType)  # hidden
    evaluation_spec: Mapped[dict] = mapped_column(JsonType)  # hidden
    reviewer_behavior: Mapped[dict] = mapped_column(JsonType, default=dict)
    limits: Mapped[dict] = mapped_column(JsonType, default=dict)


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128))
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)


class AgentConfig(Base):
    __tablename__ = "agent_configs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    adapter: Mapped[str] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("prompt_versions.id"), nullable=True
    )
    settings: Mapped[dict] = mapped_column(JsonType, default=dict)
    config_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"))
    manifest: Mapped[dict] = mapped_column(JsonType)  # immutable plan
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    payload_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    planned_runs: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    # Derived on read from runs; kept only as a coarse creation-time marker.
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    runs: Mapped[list["Run"]] = relationship(back_populates="experiment")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint("experiment_id", "agent_config_id", "task_version_id", "repetition"),
        Index("ix_runs_experiment_status", "experiment_id", "execution_status"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"))
    agent_config_id: Mapped[str] = mapped_column(ForeignKey("agent_configs.id"))
    task_version_id: Mapped[str] = mapped_column(ForeignKey("task_versions.id"))
    repetition: Mapped[int] = mapped_column(Integer, default=0)
    execution_status: Mapped[str] = mapped_column(String(32), default="queued")
    evaluation_status: Mapped[str] = mapped_column(String(32), default="pending")
    # The single attempt whose result counts. Set exactly once, when the job is finalized.
    final_attempt_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="run")


class Attempt(Base):
    __tablename__ = "attempts"
    __table_args__ = (UniqueConstraint("run_id", "attempt_number"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    environment_id: Mapped[str] = mapped_column(String(64), default=_uuid)
    execution_status: Mapped[str] = mapped_column(String(32), default="queued")
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    final_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_state: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    run: Mapped[Run] = relationship(back_populates="attempts")


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_jobs_run_id"),  # exactly one job per run
        Index("ix_jobs_available", "status", "available_at"),
        Index("ix_jobs_lease", "status", "lease_expires_at"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=2)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TraceEvent(Base):
    __tablename__ = "trace_events"
    __table_args__ = (
        UniqueConstraint("attempt_id", "sequence_number"),
        Index("ix_trace_events_attempt_seq", "attempt_id", "sequence_number"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("attempts.id"))
    sequence_number: Mapped[int] = mapped_column(Integer)
    trace_id: Mapped[str] = mapped_column(String(64))
    span_id: Mapped[str] = mapped_column(String(64))
    parent_span_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Numeric(12, 3), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict] = mapped_column(JsonType, default=dict)  # redacted


class ModelCall(Base):
    __tablename__ = "model_calls"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("attempts.id"), index=True)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)  # None = unknown
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Numeric(12, 3), nullable=True)
    price_version_id: Mapped[str | None] = mapped_column(ForeignKey("price_versions.id"), nullable=True)
    payload: Mapped[dict] = mapped_column(JsonType, default=dict)


class ToolCall(Base):
    __tablename__ = "tool_calls"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("attempts.id"), index=True)
    tool_call_id: Mapped[str] = mapped_column(String(128))
    tool_name: Mapped[str] = mapped_column(String(64))
    arguments: Mapped[dict] = mapped_column(JsonType, default=dict)
    outcome: Mapped[str] = mapped_column(String(32))  # ok | rejected | error
    reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Numeric(12, 3), nullable=True)


class Score(Base):
    __tablename__ = "scores"
    __table_args__ = (
        UniqueConstraint("attempt_id", "evaluator_name", "evaluator_version", "dimension"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("attempts.id"), index=True)
    evaluator_name: Mapped[str] = mapped_column(String(64))
    evaluator_version: Mapped[str] = mapped_column(String(32))
    dimension: Mapped[str] = mapped_column(String(64), default="overall")
    applicable: Mapped[bool] = mapped_column(default=True)
    status: Mapped[str] = mapped_column(String(32))
    score: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence: Mapped[dict] = mapped_column(JsonType, default=dict)


class PriceVersion(Base):
    __tablename__ = "price_versions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    input_per_million: Mapped[float] = mapped_column(Numeric(12, 4))
    cached_input_per_million: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    output_per_million: Mapped[float] = mapped_column(Numeric(12, 4))
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[str | None] = mapped_column(String(256), nullable=True)


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    attempt_id: Mapped[str | None] = mapped_column(ForeignKey("attempts.id"), nullable=True)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))  # snapshot | transcript | report
    uri: Mapped[str] = mapped_column(String(512))
