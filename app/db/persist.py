"""Persist one attempt (task, config, run, trace, tool calls, scores) to the database.

Phase 1 helper used by the smoke script and integration test. In Phase 4 the worker will own
this path; experiment/run/job creation will move behind the API.
"""
from __future__ import annotations

import hashlib
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.execution.runner import AttemptOutcome
from app.schemas.task import Task


def _hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _get_or_create(session: Session, model, where: dict, **create):
    row = session.scalars(select(model).filter_by(**where)).first()
    if row is None:
        row = model(**where, **create)
        session.add(row)
        session.flush()
    return row


def add_attempt_children(session: Session, attempt_id: str, outcome: AttemptOutcome) -> None:
    """Persist trace events, model usage, tool calls and scores for one attempt."""
    for e in outcome.events:
        session.add(m.TraceEvent(
            id=e.event_id, attempt_id=attempt_id, sequence_number=e.sequence_number,
            trace_id=e.trace_id, span_id=e.span_id, parent_span_id=e.parent_span_id,
            event_type=e.event_type, timestamp=e.timestamp, duration_ms=e.duration_ms,
            status=e.status, payload=e.payload))
    for mc in outcome.model_calls:
        session.add(m.ModelCall(
            attempt_id=attempt_id, provider=mc["provider"], model=mc["model"],
            input_tokens=mc["input_tokens"], output_tokens=mc["output_tokens"],
            cached_input_tokens=mc["cached_input_tokens"], duration_ms=mc["duration_ms"],
            payload={"resolved_model": mc.get("resolved_model")}))
    for c in outcome.tool_calls:
        session.add(m.ToolCall(
            attempt_id=attempt_id, tool_call_id=c.tool_call_id, tool_name=c.tool_name,
            arguments=c.arguments or {}, outcome=c.outcome, reason_code=c.reason_code,
            result=c.result, duration_ms=c.duration_ms))
    for s in outcome.scores:
        session.add(m.Score(
            attempt_id=attempt_id, evaluator_name=s.evaluator, evaluator_version=s.version,
            dimension=s.dimension, applicable=s.applicable, status=s.status, score=s.score,
            reason_code=s.reason_code, evidence=s.evidence))


def persist_outcome(session: Session, task: Task, config_name: str,
                    outcome: AttemptOutcome, dataset_name: str = "support",
                    dataset_version: str = "v0") -> m.Attempt:
    dataset = _get_or_create(session, m.Dataset, {"name": dataset_name})
    dv = _get_or_create(session, m.DatasetVersion,
                        {"dataset_id": dataset.id, "version": dataset_version},
                        content_hash=_hash(task.model_dump(by_alias=True)), manifest={})
    tv = _get_or_create(
        session, m.TaskVersion,
        {"dataset_version_id": dv.id, "task_id": task.id, "version": task.version},
        category=task.category, split=task.split,
        agent_input=task.agent_input.model_dump(),
        environment_fixture=task.environment_fixture.model_dump(by_alias=True),
        evaluation_spec=task.evaluation.model_dump(),
        reviewer_behavior=task.reviewer_behavior.model_dump(),
        limits=task.evaluation.limits.model_dump())
    config = _get_or_create(session, m.AgentConfig, {"name": config_name},
                            adapter=config_name.split(":")[0], settings={},
                            config_hash=_hash(config_name))

    experiment = m.Experiment(dataset_version_id=dv.id, status="completed",
                              manifest={"configurations": [config_name], "tasks": [task.id]})
    session.add(experiment)
    session.flush()
    run = m.Run(experiment_id=experiment.id, agent_config_id=config.id,
                task_version_id=tv.id, repetition=0,
                execution_status=outcome.execution_status,
                evaluation_status=outcome.evaluation_status)
    session.add(run)
    session.flush()

    attempt = m.Attempt(id=outcome.attempt_id, run_id=run.id, attempt_number=1,
                        environment_id=outcome.environment_id,
                        execution_status=outcome.execution_status,
                        stop_reason=outcome.stop_reason, final_response=outcome.final_response,
                        final_state=outcome.final_state)
    session.add(attempt)
    session.flush()

    add_attempt_children(session, attempt.id, outcome)
    session.flush()
    return attempt
