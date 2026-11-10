"""Create an experiment, its runs, and its jobs in ONE transaction (guide section 24)."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m

MAX_PLANNED_RUNS = 5000


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version_id: str
    agent_config_ids: list[str] = Field(min_length=1)
    split: Literal["development", "validation", "test", "all"] = "validation"
    repetitions: int = Field(default=1, ge=1, le=10)
    max_retries: int = Field(default=2, ge=0, le=5)  # infrastructure retries only
    task_ids: list[str] | None = Field(default=None, min_length=1)  # subset, for pilots/debugging

    def payload_hash(self) -> str:
        canonical = self.model_dump()
        canonical["agent_config_ids"] = sorted(set(canonical["agent_config_ids"]))
        if canonical["task_ids"]:
            canonical["task_ids"] = sorted(set(canonical["task_ids"]))
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


class PlanningError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


class IdempotencyConflict(PlanningError):
    pass


def find_by_key(s: Session, key: str, payload_hash: str) -> m.Experiment | None:
    existing = s.scalars(select(m.Experiment).where(m.Experiment.idempotency_key == key)).first()
    if existing is None:
        return None
    if existing.payload_hash != payload_hash:
        raise IdempotencyConflict(409, "Idempotency-Key was already used with a different payload")
    return existing


def create_experiment(s: Session, req: ExperimentRequest,
                      key: str | None) -> tuple[m.Experiment, bool]:
    """Returns (experiment, created). created=False means an idempotent replay."""
    payload_hash = req.payload_hash()
    if key and (existing := find_by_key(s, key, payload_hash)):
        return existing, False

    dv = s.get(m.DatasetVersion, req.dataset_version_id)
    if dv is None:
        raise PlanningError(404, f"dataset version {req.dataset_version_id} not found")
    config_ids = sorted(set(req.agent_config_ids))
    configs = s.scalars(select(m.AgentConfig).where(m.AgentConfig.id.in_(config_ids))).all()
    if len(configs) != len(config_ids):
        missing = set(config_ids) - {c.id for c in configs}
        raise PlanningError(404, f"agent configs not found: {sorted(missing)}")
    q = select(m.TaskVersion).where(m.TaskVersion.dataset_version_id == dv.id)
    if req.split != "all":
        q = q.where(m.TaskVersion.split == req.split)
    if req.task_ids:
        q = q.where(m.TaskVersion.task_id.in_(req.task_ids))
    tasks = s.scalars(q.order_by(m.TaskVersion.task_id)).all()
    if req.task_ids and {t.task_id for t in tasks} != set(req.task_ids):
        raise PlanningError(422, "unknown task ids for this split: "
                                 f"{sorted(set(req.task_ids) - {t.task_id for t in tasks})}")
    if not tasks:
        raise PlanningError(422, f"no tasks in split {req.split!r}")
    planned = len(tasks) * len(configs) * req.repetitions
    if planned > MAX_PLANNED_RUNS:
        raise PlanningError(422, f"{planned} planned runs exceeds the limit of {MAX_PLANNED_RUNS}")

    exp = m.Experiment(
        dataset_version_id=dv.id, idempotency_key=key, payload_hash=payload_hash,
        planned_runs=planned, status="queued",
        manifest={  # immutable plan: what was run, against exactly which versions
            "dataset_version_id": dv.id, "dataset_content_hash": dv.content_hash,
            "split": req.split, "repetitions": req.repetitions, "max_retries": req.max_retries,
            "configs": [{"id": c.id, "name": c.name, "config_hash": c.config_hash}
                        for c in sorted(configs, key=lambda c: c.id)],
            "task_ids": [t.task_id for t in tasks], "planned_runs": planned})
    s.add(exp)
    s.flush()

    # Interleave configurations on the same task, in a deterministic-per-experiment random
    # order, so slow drift (rate limits, provider load) does not systematically favor one config.
    rng = random.Random(exp.id)
    base = datetime.now(timezone.utc)
    position = 0
    for rep in range(req.repetitions):
        for tv in tasks:
            order = list(configs)
            rng.shuffle(order)
            for cfg in order:
                run = m.Run(experiment_id=exp.id, agent_config_id=cfg.id, task_version_id=tv.id,
                            repetition=rep)
                s.add(run)
                s.flush()
                s.add(m.Job(run_id=run.id, max_retries=req.max_retries,
                            created_at=base + timedelta(microseconds=position)))
                position += 1
    return exp, True
