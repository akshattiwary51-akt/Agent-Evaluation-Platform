"""Per-configuration scorecards with explicit denominators (guide sections 20-22).

Denominators, stated once:
- planned: every run created for this config in the experiment.
- scorable: runs whose final attempt reached an evaluation of `pass` or `fail`.
- verified success   = passed / scorable
- strict success     = passed / planned   (anything unscored counts as unsuccessful)
- coverage           = scorable / planned
- unsafe-attempt and executed-violation rates use scorable runs as the denominator.
- schema validity    = schema-valid calls / attempted calls (final attempts)
- required-call recall = satisfied / (satisfied + missing) over applicable requirements
- useful-call rate   = useful / attempted, where attempted excludes environment-injected
  transient faults (SERVICE_UNAVAILABLE, TIMEOUT) and useful = succeeded and not an exact repeat.
  (This is a documented stand-in for "call precision"; there is no per-call relevance label.)
- cost covers ALL attempts, including infrastructure retries; cost per successful run =
  total candidate cost / passed runs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis import costing, stats
from app.db import models as m

TRANSIENT = {"SERVICE_UNAVAILABLE", "TIMEOUT"}


@dataclass
class RunFact:
    run_id: str
    task_id: str
    category: str
    split: str
    repetition: int
    execution_status: str
    evaluation_status: str
    final_attempt_id: str | None
    attempts: int = 0
    passed: bool | None = None            # None = not scorable
    unsafe_attempt: bool = False
    executed_violation: bool = False
    failing: list[str] = field(default_factory=list)
    calls_total: int = 0
    calls_ok: int = 0
    calls_schema_invalid: int = 0
    calls_transient: int = 0
    calls_redundant: int = 0
    required_satisfied: int = 0
    required_missing: int = 0
    queue_seconds: float | None = None
    active_seconds: float | None = None
    model_calls: list[costing.CallRow] = field(default_factory=list)


def load_run_facts(s: Session, experiment_id: str, config_id: str) -> list[RunFact]:
    rows = s.execute(
        select(m.Run, m.TaskVersion.task_id, m.TaskVersion.category, m.TaskVersion.split,
               m.Job.created_at)
        .join(m.TaskVersion, m.TaskVersion.id == m.Run.task_version_id)
        .outerjoin(m.Job, m.Job.run_id == m.Run.id)
        .where(m.Run.experiment_id == experiment_id, m.Run.agent_config_id == config_id)
        .order_by(m.TaskVersion.task_id, m.Run.repetition)).all()
    facts = {r.id: RunFact(r.id, task_id, cat, split, r.repetition, r.execution_status,
                           r.evaluation_status, r.final_attempt_id)
             for r, task_id, cat, split, _ in rows}
    job_created = {r.id: created for r, _, _, _, created in rows}
    by_attempt = {f.final_attempt_id: f for f in facts.values() if f.final_attempt_id}

    for a, run_id in s.execute(
            select(m.Attempt, m.Attempt.run_id).join(m.Run, m.Run.id == m.Attempt.run_id)
            .where(m.Run.experiment_id == experiment_id, m.Run.agent_config_id == config_id)):
        f = facts[run_id]
        f.attempts += 1
        if a.id == f.final_attempt_id and a.started_at and a.finished_at:
            f.active_seconds = (a.finished_at - a.started_at).total_seconds()
            if job_created.get(run_id):
                f.queue_seconds = max(0.0, (a.started_at - job_created[run_id]).total_seconds())

    for sc in s.scalars(select(m.Score).join(m.Run, m.Run.final_attempt_id == m.Score.attempt_id)
                        .where(m.Run.experiment_id == experiment_id,
                               m.Run.agent_config_id == config_id)):
        f = by_attempt[sc.attempt_id]
        if sc.status == "fail":
            f.failing.append(f"{sc.evaluator_name}:{sc.dimension}:{sc.reason_code}")
        if sc.evaluator_name == "policy":
            if sc.status == "fail":
                f.executed_violation = True
            if (sc.evidence or {}).get("attempted"):
                f.unsafe_attempt = True
        elif sc.evaluator_name == "trajectory":
            ev = sc.evidence or {}
            if sc.dimension == "required_calls":
                f.required_satisfied += len(ev.get("satisfied", []))
                f.required_missing += len(ev.get("missing", []))
            elif sc.dimension == "redundancy":
                f.calls_redundant += ev.get("redundant_calls", 0)

    for tc in s.scalars(select(m.ToolCall).join(m.Run, m.Run.final_attempt_id == m.ToolCall.attempt_id)
                        .where(m.Run.experiment_id == experiment_id,
                               m.Run.agent_config_id == config_id)):
        f = by_attempt[tc.attempt_id]
        f.calls_total += 1
        f.calls_ok += tc.outcome == "ok"
        f.calls_schema_invalid += tc.reason_code == "SCHEMA_INVALID"
        f.calls_transient += tc.reason_code in TRANSIENT

    # Usage from EVERY attempt (retries and interrupted attempts included): they cost money too.
    for mc, started, run_id in s.execute(
            select(m.ModelCall, m.Attempt.started_at, m.Attempt.run_id)
            .join(m.Attempt, m.Attempt.id == m.ModelCall.attempt_id)
            .join(m.Run, m.Run.id == m.Attempt.run_id)
            .where(m.Run.experiment_id == experiment_id, m.Run.agent_config_id == config_id)):
        facts[run_id].model_calls.append(costing.CallRow(
            mc.provider, mc.model, mc.input_tokens, mc.output_tokens, mc.cached_input_tokens, started))

    for f in facts.values():
        if f.evaluation_status in ("pass", "fail"):
            f.passed = f.evaluation_status == "pass"
    return sorted(facts.values(), key=lambda f: (f.task_id, f.repetition))


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def _group_rates(facts: list[RunFact]) -> dict[str, Any]:
    scorable = [f for f in facts if f.passed is not None]
    return {"planned": len(facts), "scorable": len(scorable),
            "passed": sum(1 for f in scorable if f.passed),
            "verified_success": _safe_div(sum(1 for f in scorable if f.passed), len(scorable)),
            "coverage": _safe_div(len(scorable), len(facts)),
            "unsafe_attempt_rate": _safe_div(sum(f.unsafe_attempt for f in scorable), len(scorable)),
            "executed_violation_rate": _safe_div(sum(f.executed_violation for f in scorable),
                                                 len(scorable))}


def task_units(facts: list[RunFact]) -> dict[str, stats.TaskUnit]:
    units: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for f in facts:
        if f.passed is not None:
            units[f.task_id][1] += 1
            units[f.task_id][0] += f.passed
    return {t: (p, n) for t, (p, n) in units.items()}


def build_scorecard(facts: list[RunFact], prices: dict, *, seed: str = "") -> dict[str, Any]:
    planned = len(facts)
    scorable = [f for f in facts if f.passed is not None]
    passed = sum(1 for f in scorable if f.passed)
    units = task_units(facts)

    ok_calls = sum(f.calls_ok for f in facts)
    redundant = sum(f.calls_redundant for f in facts)
    attempted = sum(f.calls_total for f in facts)
    attempted_nontransient = attempted - sum(f.calls_transient for f in facts)
    req_sat = sum(f.required_satisfied for f in facts)
    req_miss = sum(f.required_missing for f in facts)

    all_calls = [c for f in facts for c in f.model_calls]
    cost = costing.cost_calls(all_calls, prices)
    unknown_usage = sum(1 for c in all_calls if c.input_tokens is None or c.output_tokens is None)

    per_task_reps: dict[str, list[bool | None]] = defaultdict(list)
    for f in facts:
        per_task_reps[f.task_id].append(f.passed)
    by_cat: dict[str, list[RunFact]] = defaultdict(list)
    for f in facts:
        by_cat[f.category].append(f)
    reasons = Counter(code for f in scorable for code in set(f.failing))

    return {
        "headline": {
            "planned_runs": planned, "scorable_runs": len(scorable), "passed_runs": passed,
            "verified_success_rate": _safe_div(passed, len(scorable)),
            "strict_end_to_end_success": _safe_div(passed, planned),
            "evaluation_coverage": _safe_div(len(scorable), planned),
            "verified_success_ci95": stats.bootstrap_ci(list(units.values()), seed=seed),
        },
        "execution": dict(Counter(f.execution_status for f in facts)),
        "evaluation": dict(Counter(f.evaluation_status for f in facts)),
        "hard_violations": {
            "unsafe_attempt_rate": _safe_div(sum(f.unsafe_attempt for f in scorable), len(scorable)),
            "executed_violation_rate": _safe_div(sum(f.executed_violation for f in scorable),
                                                 len(scorable)),
            "runs_with_unsafe_attempt": sum(f.unsafe_attempt for f in scorable),
            "runs_with_executed_violation": sum(f.executed_violation for f in scorable)},
        "tool_behavior": {
            "attempted_calls": attempted, "mean_calls_per_run": _safe_div(attempted, planned),
            "schema_validity": _safe_div(attempted - sum(f.calls_schema_invalid for f in facts),
                                         attempted),
            "required_call_recall": _safe_div(req_sat, req_sat + req_miss),
            "useful_call_rate": _safe_div(ok_calls - redundant, attempted_nontransient),
            "redundant_calls": redundant},
        "latency_seconds": {
            "queue": stats.summarize([f.queue_seconds for f in facts if f.queue_seconds is not None]),
            "active_execution": stats.summarize([f.active_seconds for f in facts
                                                 if f.active_seconds is not None])},
        "usage": {"model_calls": len(all_calls), "unknown_usage_calls": unknown_usage,
                  "input_tokens": sum(c.input_tokens or 0 for c in all_calls),
                  "output_tokens": sum(c.output_tokens or 0 for c in all_calls),
                  "cached_input_tokens": sum(c.cached_input_tokens or 0 for c in all_calls),
                  "attempts": sum(f.attempts for f in facts),
                  "retried_or_interrupted_attempts": sum(max(0, f.attempts - 1) for f in facts)},
        "cost": {"status": cost.status, "total": cost.total, "currency": cost.currency,
                 "unknown_calls": cost.unknown_calls, "known_lower_bound": cost.known_lower_bound,
                 "per_successful_run": (cost.total / passed if cost.total is not None and passed
                                        else (0.0 if cost.status == "none_incurred" and passed else None)),
                 "pricing_basis": cost.pricing_basis},
        "repeated_trials": stats.repetition_metrics(dict(per_task_reps)),
        "by_category": {cat: _group_rates(fs) for cat, fs in sorted(by_cat.items())},
        "failure_reasons": [{"reason": r, "runs": n} for r, n in reasons.most_common()],
    }


def experiment_scorecards(s: Session, exp: m.Experiment) -> dict[str, dict[str, Any]]:
    """Scorecard per configuration of an experiment, keyed by config name."""
    from app.analysis import costing as _costing
    prices = _costing.load_prices(s)
    return {c["name"]: build_scorecard(load_run_facts(s, exp.id, c["id"]), prices,
                                       seed=f"{exp.id}:{c['id']}")
            for c in exp.manifest["configs"]}
