from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.prompts import PROMPTS, prompt_hash
from app.agents.scripted import BEHAVIORS
from app.api.deps import get_session
from app.db import models as m
from app.execution import queue
from app.execution.planning import (ExperimentRequest, IdempotencyConflict, PlanningError,
                                    create_experiment, find_by_key)
from app.metrics import experiment_summary

router = APIRouter()
health = APIRouter()  # unauthenticated so orchestrators can probe it
SessionDep = Annotated[Session, Depends(get_session)]


def _or_404(obj: Any, what: str) -> Any:
    if obj is None:
        raise HTTPException(404, f"{what} not found")
    return obj


# ------------------------------------------------------------------ health
@health.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@health.get("/health/ready")
def ready(s: SessionDep) -> dict[str, str]:
    try:
        s.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(503, "database unavailable") from exc
    return {"status": "ok"}


# ------------------------------------------------------------------ datasets & configs
@router.get("/datasets")
def list_datasets(s: SessionDep) -> list[dict]:
    out = []
    for ds in s.scalars(select(m.Dataset).order_by(m.Dataset.name)):
        versions = s.scalars(select(m.DatasetVersion).where(m.DatasetVersion.dataset_id == ds.id)
                             .order_by(m.DatasetVersion.version)).all()
        out.append({"id": ds.id, "name": ds.name, "versions": [
            {"id": v.id, "version": v.version, "content_hash": v.content_hash,
             "task_count": (v.manifest or {}).get("task_count"),
             "categories": (v.manifest or {}).get("categories")} for v in versions]})
    return out


class AgentConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    adapter: str
    model: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


def _config_view(c: m.AgentConfig) -> dict:
    return {"id": c.id, "name": c.name, "adapter": c.adapter, "model": c.model,
            "settings": c.settings, "config_hash": c.config_hash}


@router.post("/agent-configs")
def register_agent_config(req: AgentConfigRequest, s: SessionDep, response: Response) -> dict:
    """Configurations are immutable: same name + same content is a no-op; a changed config needs
    a new name."""
    if req.adapter == "scripted":
        if req.settings.get("behavior") not in BEHAVIORS:
            raise HTTPException(422, f"settings.behavior must be one of {sorted(BEHAVIORS)}")
        identity: dict[str, Any] = {}
    elif req.adapter == "tool_loop":
        if req.settings.get("prompt") not in PROMPTS or req.settings.get("provider") != "anthropic":
            raise HTTPException(422, f"tool_loop needs settings.provider='anthropic' and "
                                     f"settings.prompt in {sorted(PROMPTS)}")
        if not req.model:
            raise HTTPException(422, "tool_loop needs a model")
        identity = {"prompt_hash": prompt_hash(req.settings["prompt"])}
    else:
        raise HTTPException(422, "adapter must be 'scripted' or 'tool_loop'")
    digest = hashlib.sha256(json.dumps(
        {"adapter": req.adapter, "model": req.model, "settings": req.settings, **identity},
        sort_keys=True).encode()).hexdigest()
    existing = s.scalars(select(m.AgentConfig).where(m.AgentConfig.name == req.name)).first()
    if existing:
        if existing.config_hash != digest:
            raise HTTPException(409, "configs are immutable; register changes under a new name")
        return _config_view(existing)
    cfg = m.AgentConfig(name=req.name, adapter=req.adapter, model=req.model,
                        settings=req.settings, config_hash=digest)
    s.add(cfg)
    try:
        s.commit()
    except IntegrityError:  # concurrent registration of the same name
        s.rollback()
        cfg = s.scalars(select(m.AgentConfig).where(m.AgentConfig.name == req.name)).one()
        if cfg.config_hash != digest:
            raise HTTPException(409, "configs are immutable; register changes under a new name")
    response.status_code = 201
    return _config_view(cfg)


@router.get("/agent-configs")
def list_agent_configs(s: SessionDep) -> list[dict]:
    return [_config_view(c) for c in s.scalars(select(m.AgentConfig).order_by(m.AgentConfig.name))]


# ------------------------------------------------------------------ experiments
@router.post("/experiments")
def post_experiment(req: ExperimentRequest, s: SessionDep, response: Response,
                    idempotency_key: Annotated[str | None, Header(max_length=128)] = None) -> dict:
    try:
        exp, was_new = create_experiment(s, req, idempotency_key)
        s.commit()  # experiment + runs + jobs become visible together, or not at all
    except IdempotencyConflict as exc:
        s.rollback()
        raise HTTPException(exc.status_code, exc.detail) from exc
    except PlanningError as exc:
        s.rollback()
        raise HTTPException(exc.status_code, exc.detail) from exc
    except IntegrityError:
        # Two requests raced on the same Idempotency-Key; the other one won. Return its result.
        s.rollback()
        try:
            exp = find_by_key(s, idempotency_key, req.payload_hash()) if idempotency_key else None
        except IdempotencyConflict as exc:
            raise HTTPException(409, exc.detail) from exc
        if exp is None:
            raise
        was_new = False
    response.status_code = 201 if was_new else 200
    return {"experiment_id": exp.id, "status": "queued" if was_new else experiment_summary(s, exp)["status"],
            "planned_runs": exp.planned_runs}


@router.get("/experiments")
def list_experiments(s: SessionDep, limit: int = Query(50, ge=1, le=200)) -> list[dict]:
    exps = s.scalars(select(m.Experiment).order_by(m.Experiment.created_at.desc()).limit(limit))
    return [{k: v for k, v in experiment_summary(s, e).items() if k != "manifest"} for e in exps]


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str, s: SessionDep) -> dict:
    return experiment_summary(s, _or_404(s.get(m.Experiment, experiment_id), "experiment"))


@router.get("/experiments/{experiment_id}/runs")
def list_runs(experiment_id: str, s: SessionDep,
              execution_status: str | None = None, evaluation_status: str | None = None,
              category: str | None = None, limit: int = Query(50, ge=1, le=500),
              offset: int = Query(0, ge=0)) -> dict:
    _or_404(s.get(m.Experiment, experiment_id), "experiment")
    q = (select(m.Run, m.TaskVersion, m.AgentConfig)
         .join(m.TaskVersion, m.TaskVersion.id == m.Run.task_version_id)
         .join(m.AgentConfig, m.AgentConfig.id == m.Run.agent_config_id)
         .where(m.Run.experiment_id == experiment_id))
    if execution_status:
        q = q.where(m.Run.execution_status == execution_status)
    if evaluation_status:
        q = q.where(m.Run.evaluation_status == evaluation_status)
    if category:
        q = q.where(m.TaskVersion.category == category)
    total = s.scalar(select(func.count()).select_from(q.subquery()))
    rows = s.execute(q.order_by(m.TaskVersion.task_id, m.AgentConfig.name, m.Run.repetition)
                     .limit(limit).offset(offset)).all()
    return {"total": total, "limit": limit, "offset": offset, "items": [
        {"run_id": r.id, "task_id": tv.task_id, "category": tv.category, "config": c.name,
         "repetition": r.repetition, "execution_status": r.execution_status,
         "evaluation_status": r.evaluation_status, "final_attempt_id": r.final_attempt_id}
        for r, tv, c in rows]}


@router.post("/experiments/{experiment_id}/cancel")
def cancel(experiment_id: str, s: SessionDep) -> dict:
    _or_404(s.get(m.Experiment, experiment_id), "experiment")
    result = queue.cancel_experiment(s, experiment_id)
    s.commit()
    return {"experiment_id": experiment_id, "cancel_requested": True, **result}


# ------------------------------------------------------------------ runs & traces
@router.get("/runs/{run_id}")
def get_run(run_id: str, s: SessionDep) -> dict:
    run = _or_404(s.get(m.Run, run_id), "run")
    tv, cfg = s.get(m.TaskVersion, run.task_version_id), s.get(m.AgentConfig, run.agent_config_id)
    attempts = s.scalars(select(m.Attempt).where(m.Attempt.run_id == run_id)
                         .order_by(m.Attempt.attempt_number)).all()
    final = next((a for a in attempts if a.id == run.final_attempt_id), None)
    scores = s.scalars(select(m.Score).where(m.Score.attempt_id == run.final_attempt_id)).all() \
        if final else []
    return {
        "run_id": run.id, "experiment_id": run.experiment_id, "config": cfg.name,
        "repetition": run.repetition, "execution_status": run.execution_status,
        "evaluation_status": run.evaluation_status,
        # Kept visually and structurally separate: what the agent saw vs. hidden ground truth.
        "agent_visible": {"task_id": tv.task_id, "category": tv.category, **tv.agent_input},
        "evaluator_visible": {"expected": tv.evaluation_spec, "fixture": tv.environment_fixture},
        "attempts": [{"attempt_id": a.id, "number": a.attempt_number,
                      "execution_status": a.execution_status, "stop_reason": a.stop_reason}
                     for a in attempts],
        "final_response": final.final_response if final else None,
        "scores": [{"evaluator": x.evaluator_name, "dimension": x.dimension, "status": x.status,
                    "reason_code": x.reason_code, "evidence": x.evidence} for x in scores]}


@router.get("/attempts/{attempt_id}/events")
def attempt_events(attempt_id: str, s: SessionDep, after: int = Query(-1, ge=-1),
                   limit: int = Query(200, ge=1, le=1000)) -> dict:
    _or_404(s.get(m.Attempt, attempt_id), "attempt")
    events = s.scalars(select(m.TraceEvent).where(
        m.TraceEvent.attempt_id == attempt_id, m.TraceEvent.sequence_number > after)
        .order_by(m.TraceEvent.sequence_number).limit(limit)).all()
    return {"attempt_id": attempt_id, "events": [
        {"sequence": e.sequence_number, "type": e.event_type, "timestamp": e.timestamp.isoformat(),
         "duration_ms": float(e.duration_ms) if e.duration_ms is not None else None,
         "status": e.status, "payload": e.payload} for e in events]}
