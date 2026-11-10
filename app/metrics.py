"""Experiment progress and headline metrics with explicit denominators (guide section 20)."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import models as m

TERMINAL = {"completed", "timed_out", "budget_exceeded", "infrastructure_error", "cancelled"}


def experiment_summary(s: Session, exp: m.Experiment) -> dict:
    rows = s.execute(select(m.Run.execution_status, m.Run.evaluation_status, func.count())
                     .where(m.Run.experiment_id == exp.id)
                     .group_by(m.Run.execution_status, m.Run.evaluation_status)).all()
    by_exec: dict[str, int] = {}
    by_eval: dict[str, int] = {}
    for ex, ev, n in rows:
        by_exec[ex] = by_exec.get(ex, 0) + n
        by_eval[ev] = by_eval.get(ev, 0) + n
    total = sum(by_exec.values())
    finished = sum(n for k, n in by_exec.items() if k in TERMINAL)
    if total and finished == total:
        status = "cancelled" if exp.cancel_requested and by_exec.get("cancelled") else "completed"
    elif exp.cancel_requested:
        status = "cancelling"
    elif by_exec.get("running") or finished:
        status = "running"
    else:
        status = "queued"
    passed, failed = by_eval.get("pass", 0), by_eval.get("fail", 0)
    scorable = passed + failed
    planned = exp.planned_runs
    return {
        "experiment_id": exp.id, "status": status, "planned_runs": planned,
        "execution": by_exec, "evaluation": by_eval,
        "metrics": {
            "scorable_runs": scorable,
            # Verified success: passed / scorable. Strict: passed / ALL planned (missing = fail).
            "verified_success_rate": passed / scorable if scorable else None,
            "strict_end_to_end_success": passed / planned if planned else None,
            "evaluation_coverage": scorable / planned if planned else None},
        "manifest": exp.manifest}
