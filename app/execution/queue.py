"""PostgreSQL-backed job queue with leases (guide section 17). Requires PostgreSQL.

Guarantees, and how each is enforced:
- Two workers never hold the same job: `FOR UPDATE SKIP LOCKED` claim in a short transaction.
- A lease must be kept alive: heartbeats extend it only if the caller still holds the token.
- A stale worker cannot finalize: every state change is guarded by `lease_token` and
  `status = 'running'`, and attempt results are written in the SAME transaction as the guard.
- A result is counted once: `runs.final_attempt_id` is set only by the guarded finalize.
- Execution is at-least-once: a lost worker's attempt is marked `interrupted` and the job is
  requeued (bounded by max_retries) with a fresh attempt and a fresh environment.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text, update
from sqlalchemy.orm import Session, sessionmaker

from app.db import models as m
from app.db.persist import add_attempt_children
from app.execution.runner import AttemptOutcome

CLAIM_SQL = text("""
WITH picked AS (
    SELECT id FROM jobs
    WHERE status = 'queued' AND available_at <= now()
    ORDER BY created_at, id
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE jobs AS j
SET status = 'running', worker_id = :worker_id, lease_token = :lease_token,
    lease_expires_at = now() + (:lease_seconds * interval '1 second')
FROM picked
WHERE j.id = picked.id
RETURNING j.id, j.run_id, j.retry_count, j.max_retries
""")

HEARTBEAT_SQL = text("""
UPDATE jobs SET lease_expires_at = now() + (:lease_seconds * interval '1 second')
WHERE id = :id AND lease_token = :token AND status = 'running'
""")

# Guarded transition used by finalize. A stale worker matches zero rows.
GUARDED_FINISH_SQL = text("""
UPDATE jobs
SET status = :status, lease_token = NULL, worker_id = NULL, lease_expires_at = NULL,
    retry_count = :retry_count, last_error = :last_error,
    available_at = now() + (:delay_seconds * interval '1 second'),
    finished_at = CASE WHEN :terminal THEN now() ELSE NULL END
WHERE id = :id AND lease_token = :token AND status = 'running'
""")

EXPIRED_SQL = text("""
SELECT j.id, j.run_id, j.retry_count, j.max_retries, e.cancel_requested
FROM jobs j JOIN runs r ON r.id = j.run_id JOIN experiments e ON e.id = r.experiment_id
WHERE j.status = 'running' AND j.lease_expires_at < now()
FOR UPDATE OF j SKIP LOCKED
""")


@dataclass(frozen=True)
class ClaimedJob:
    id: str
    run_id: str
    lease_token: str
    retry_count: int
    max_retries: int


def _now() -> datetime:
    return datetime.now(timezone.utc)


def claim_job(factory: sessionmaker, worker_id: str, lease_seconds: float) -> ClaimedJob | None:
    token = uuid.uuid4().hex
    with factory() as s, s.begin():  # short transaction: never held while a model is called
        row = s.execute(CLAIM_SQL, {"worker_id": worker_id, "lease_token": token,
                                    "lease_seconds": lease_seconds}).first()
    return ClaimedJob(row.id, row.run_id, token, row.retry_count, row.max_retries) if row else None


def heartbeat(s: Session, job: ClaimedJob, lease_seconds: float) -> bool:
    res = s.execute(HEARTBEAT_SQL, {"id": job.id, "token": job.lease_token,
                                    "lease_seconds": lease_seconds})
    return res.rowcount == 1


def run_cancel_requested(s: Session, run_id: str) -> bool:
    return bool(s.execute(text(
        "SELECT e.cancel_requested FROM runs r JOIN experiments e ON e.id = r.experiment_id "
        "WHERE r.id = :r"), {"r": run_id}).scalar())


def start_attempt(factory: sessionmaker, job: ClaimedJob) -> str | None:
    """Create the 'running' attempt row up front so a crash leaves a visible interrupted attempt.
    Returns None if this worker no longer holds the lease."""
    with factory() as s, s.begin():
        still_mine = s.execute(text(
            "SELECT 1 FROM jobs WHERE id = :id AND lease_token = :t AND status = 'running' "
            "FOR UPDATE"), {"id": job.id, "t": job.lease_token}).first()
        if not still_mine:
            return None
        n = s.execute(text("SELECT coalesce(max(attempt_number), 0) + 1 FROM attempts "
                           "WHERE run_id = :r"), {"r": job.run_id}).scalar()
        attempt = m.Attempt(run_id=job.run_id, attempt_number=n, execution_status="running",
                            started_at=_now())
        s.add(attempt)
        s.execute(update(m.Run).where(m.Run.id == job.run_id).values(execution_status="running"))
        s.flush()
        return attempt.id


def cancel_claimed_job(factory: sessionmaker, job: ClaimedJob) -> bool:
    """Cancel a job this worker claimed but has not started (cancel arrived in between)."""
    with factory() as s, s.begin():
        res = s.execute(GUARDED_FINISH_SQL, {
            "id": job.id, "token": job.lease_token, "status": "cancelled",
            "retry_count": job.retry_count, "last_error": None, "delay_seconds": 0,
            "terminal": True})
        if res.rowcount != 1:
            return False
        s.execute(update(m.Run).where(m.Run.id == job.run_id).values(execution_status="cancelled"))
        return True


def finalize(factory: sessionmaker, job: ClaimedJob, attempt_id: str, outcome: AttemptOutcome,
             retry_delay_seconds: float = 0.0) -> str:
    """Record an attempt's result and move the job on, atomically and only if the lease is still
    ours. Returns: finalized | retry_scheduled | failed | cancelled | stale."""
    status = outcome.execution_status
    will_retry = status == "infrastructure_error" and job.retry_count < job.max_retries
    if status == "cancelled":
        new_status, disposition = "cancelled", "cancelled"
    elif status == "infrastructure_error":
        new_status, disposition = ("queued", "retry_scheduled") if will_retry else ("failed", "failed")
    else:
        new_status, disposition = "succeeded", "finalized"  # agent outcomes are never retried

    with factory() as s, s.begin():
        res = s.execute(GUARDED_FINISH_SQL, {
            "id": job.id, "token": job.lease_token, "status": new_status,
            "retry_count": job.retry_count + (1 if will_retry else 0),
            "last_error": outcome.error if status == "infrastructure_error" else None,
            "delay_seconds": retry_delay_seconds if will_retry else 0,
            "terminal": new_status != "queued"})
        if res.rowcount != 1:
            s.rollback()  # stale worker: write nothing
            return "stale"

        attempt = s.get(m.Attempt, attempt_id)
        attempt.environment_id = outcome.environment_id
        attempt.execution_status = status
        attempt.stop_reason = outcome.stop_reason
        attempt.final_response = outcome.final_response
        attempt.final_state = outcome.final_state
        attempt.finished_at = _now()
        add_attempt_children(s, attempt_id, outcome)  # includes usage from failed/retried attempts

        run = s.get(m.Run, job.run_id)
        if disposition == "finalized":
            run.execution_status = status
            run.evaluation_status = outcome.evaluation_status
            run.final_attempt_id = attempt_id
        elif disposition == "retry_scheduled":
            run.execution_status = "queued"
        elif disposition == "failed":
            run.execution_status = "infrastructure_error"
        else:
            run.execution_status = "cancelled"
        return disposition


def recover_expired(factory: sessionmaker) -> list[tuple[str, str]]:
    """Sweep jobs whose lease expired (worker died or stalled). Safe to run from many workers."""
    actions: list[tuple[str, str]] = []
    with factory() as s, s.begin():
        for row in s.execute(EXPIRED_SQL).all():
            interrupted = s.query(m.Attempt).filter(
                m.Attempt.run_id == row.run_id, m.Attempt.execution_status == "running").all()
            for a in interrupted:
                a.execution_status, a.stop_reason, a.finished_at = (
                    "interrupted", "lease_expired", _now())
                s.add(m.TraceEvent(attempt_id=a.id, sequence_number=0, trace_id=a.id,
                                   span_id="recovery", event_type="attempt_interrupted",
                                   timestamp=_now(), status="interrupted",
                                   payload={"reason": "lease_expired", "job_id": row.id}))
            if row.cancel_requested:
                new, action, run_status, retry = "cancelled", "cancelled", "cancelled", row.retry_count
            elif row.retry_count < row.max_retries:
                new, action, run_status, retry = "queued", "requeued", "queued", row.retry_count + 1
            else:
                new, action, run_status, retry = ("failed", "failed", "infrastructure_error",
                                                  row.retry_count)
            s.execute(text(
                "UPDATE jobs SET status = :st, retry_count = :rc, lease_token = NULL, "
                "worker_id = NULL, lease_expires_at = NULL, available_at = now(), "
                "last_error = 'lease expired', "
                "finished_at = CASE WHEN :terminal THEN now() ELSE NULL END WHERE id = :id"),
                {"st": new, "rc": retry, "id": row.id, "terminal": new != "queued"})
            s.execute(update(m.Run).where(m.Run.id == row.run_id).values(
                execution_status=run_status))
            actions.append((row.id, action))
    return actions


def cancel_experiment(s: Session, experiment_id: str) -> dict[str, Any]:
    """Request cancellation. Queued jobs are cancelled now; running jobs stop before their next
    model call or tool execution (workers poll the flag via their heartbeat)."""
    s.execute(update(m.Experiment).where(m.Experiment.id == experiment_id)
              .values(cancel_requested=True))
    runs = text("SELECT id FROM runs WHERE experiment_id = :e")
    cancelled_jobs = s.execute(text(
        f"UPDATE jobs SET status = 'cancelled', finished_at = now() "
        f"WHERE status = 'queued' AND run_id IN ({runs.text})"), {"e": experiment_id}).rowcount
    s.execute(text(
        "UPDATE runs SET execution_status = 'cancelled' "
        "WHERE experiment_id = :e AND execution_status = 'queued'"), {"e": experiment_id})
    return {"cancelled_queued_jobs": cancelled_jobs}
