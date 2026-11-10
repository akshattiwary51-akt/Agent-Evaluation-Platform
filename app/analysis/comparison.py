"""Compare two configurations on the same tasks (guide section 22).

Each side is (experiment_id, config_id). Sides may live in different experiments, but then the
comparison is only valid if they ran the same tasks from the same immutable dataset content,
in the same split, scored by the same evaluator versions. Otherwise it is refused (422) rather
than silently comparing unlike things.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis import costing, scorecard, stats
from app.db import models as m


class ComparisonError(Exception):
    def __init__(self, status_code: int, problems: list[str]):
        super().__init__("; ".join(problems))
        self.status_code, self.problems = status_code, problems


@dataclass(frozen=True)
class Side:
    experiment_id: str
    config_id: str


# (label, path in scorecard, higher_is_better)
METRICS: list[tuple[str, tuple[str, ...], bool | None]] = [
    ("verified_success_rate", ("headline", "verified_success_rate"), True),
    ("strict_end_to_end_success", ("headline", "strict_end_to_end_success"), True),
    ("evaluation_coverage", ("headline", "evaluation_coverage"), True),
    ("executed_violation_rate", ("hard_violations", "executed_violation_rate"), False),
    ("unsafe_attempt_rate", ("hard_violations", "unsafe_attempt_rate"), False),
    ("required_call_recall", ("tool_behavior", "required_call_recall"), True),
    ("useful_call_rate", ("tool_behavior", "useful_call_rate"), True),
    ("schema_validity", ("tool_behavior", "schema_validity"), True),
    ("mean_calls_per_run", ("tool_behavior", "mean_calls_per_run"), False),
    ("active_latency_mean_s", ("latency_seconds", "active_execution", "mean"), False),
    ("active_latency_p95_s", ("latency_seconds", "active_execution", "p95"), False),
    ("input_tokens", ("usage", "input_tokens"), False),
    ("output_tokens", ("usage", "output_tokens"), False),
    ("total_cost", ("cost", "total"), False),
    ("cost_per_successful_run", ("cost", "per_successful_run"), False),
]


def _dig(d: dict, path: tuple[str, ...]) -> Any:
    for p in path:
        d = d.get(p) if isinstance(d, dict) else None
        if d is None:
            return None
    return d


def _evaluator_versions(s: Session, side: Side) -> set[tuple[str, str]]:
    rows = s.execute(select(m.Score.evaluator_name, m.Score.evaluator_version).distinct()
                     .join(m.Run, m.Run.final_attempt_id == m.Score.attempt_id)
                     .where(m.Run.experiment_id == side.experiment_id,
                            m.Run.agent_config_id == side.config_id)).all()
    return {(a, b) for a, b in rows}


def check_compatibility(s: Session, base: Side, cand: Side) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    if base == cand:
        problems.append("baseline and candidate are the same configuration in the same experiment")
    ea, eb = s.get(m.Experiment, base.experiment_id), s.get(m.Experiment, cand.experiment_id)
    if ea is None or eb is None:
        raise ComparisonError(404, ["experiment not found"])
    for side, exp in ((base, ea), (cand, eb)):
        ids = {c["id"] for c in exp.manifest["configs"]}
        if side.config_id not in ids:
            raise ComparisonError(404, [f"config {side.config_id} is not part of experiment {exp.id}"])
    ma, mb = ea.manifest, eb.manifest
    if ma["dataset_content_hash"] != mb["dataset_content_hash"]:
        problems.append("experiments used different dataset content (hash mismatch)")
    if ma["split"] != mb["split"]:
        problems.append(f"different splits: {ma['split']!r} vs {mb['split']!r}")
    if sorted(ma["task_ids"]) != sorted(mb["task_ids"]):
        problems.append("experiments ran different task sets")
    va, vb = _evaluator_versions(s, base), _evaluator_versions(s, cand)
    if va and vb and va != vb:
        problems.append(f"scored by different evaluator versions: {sorted(va ^ vb)}")
    if ma["repetitions"] != mb["repetitions"]:
        warnings.append(f"repetitions differ ({ma['repetitions']} vs {mb['repetitions']}); "
                        "per-task rates are comparable but variance is not")
    return problems, warnings


def compare(s: Session, base: Side, cand: Side) -> dict[str, Any]:
    problems, warnings = check_compatibility(s, base, cand)
    if problems:
        raise ComparisonError(422, problems)

    prices = costing.load_prices(s)
    bf = scorecard.load_run_facts(s, base.experiment_id, base.config_id)
    cf = scorecard.load_run_facts(s, cand.experiment_id, cand.config_id)
    seed = f"{base.experiment_id}:{base.config_id}:{cand.experiment_id}:{cand.config_id}"
    bs, cs = scorecard.build_scorecard(bf, prices, seed=seed + ":b"), \
        scorecard.build_scorecard(cf, prices, seed=seed + ":c")
    bu, cu = scorecard.task_units(bf), scorecard.task_units(cf)

    cats = {f.task_id: f.category for f in bf + cf}
    codes: dict[str, dict[str, set[str]]] = {"base": {}, "cand": {}}
    for tag, facts in (("base", bf), ("cand", cf)):
        for f in facts:
            codes[tag].setdefault(f.task_id, set()).update(f.failing)

    rows, tally = [], {"improved": 0, "regressed": 0, "same_pass": 0, "same_fail": 0,
                       "not_comparable": 0}
    for task_id in sorted(cats):
        (bp, bn), (cp, cn) = bu.get(task_id, (0, 0)), cu.get(task_id, (0, 0))
        if bn == 0 or cn == 0:
            verdict = "not_comparable"       # unscored on a side: cannot call a winner
        else:
            br, cr = bp / bn, cp / cn
            verdict = ("improved" if cr > br else "regressed" if cr < br else
                       "same_pass" if cr == 1 else "same_fail")
        tally[verdict] += 1
        rows.append({"task_id": task_id, "category": cats[task_id], "verdict": verdict,
                     "baseline": {"passed": bp, "scorable": bn},
                     "candidate": {"passed": cp, "scorable": cn},
                     "baseline_failing": sorted(codes["base"].get(task_id, [])),
                     "candidate_failing": sorted(codes["cand"].get(task_id, []))})

    metrics = {}
    better, worse = [], []
    for label, path, higher_better in METRICS:
        b, c = _dig(bs, path), _dig(cs, path)
        entry: dict[str, Any] = {"baseline": b, "candidate": c, "delta": None}
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            entry["delta"] = c - b
            if higher_better is not None and c != b:
                (better if (c > b) == higher_better else worse).append(label)
        metrics[label] = entry

    if bs["headline"]["evaluation_coverage"] != cs["headline"]["evaluation_coverage"]:
        warnings.append("evaluation coverage differs, so verified success rates are computed over "
                        "different sets of runs; prefer strict end-to-end success")
    if tally["not_comparable"]:
        warnings.append(f"{tally['not_comparable']} task(s) were unscored on at least one side "
                        "and are excluded from wins/losses and the confidence interval")
    diff = stats.paired_bootstrap_diff(bu, cu, seed=seed)
    if diff is None:
        warnings.append("too few commonly-scored tasks for a confidence interval")

    return {
        "baseline": _side_view(s, base, bs), "candidate": _side_view(s, cand, cs),
        "compatible": True, "warnings": warnings,
        "task_outcomes": tally,
        "success_difference": diff,
        "metrics": metrics, "metrics_better": better, "metrics_worse": worse,
        "regressed_cases": [r for r in rows if r["verdict"] == "regressed"],
        "improved_cases": [r for r in rows if r["verdict"] == "improved"],
        "tasks": rows}


def _side_view(s: Session, side: Side, card: dict) -> dict[str, Any]:
    cfg = s.get(m.AgentConfig, side.config_id)
    return {"experiment_id": side.experiment_id, "config_id": side.config_id,
            "config_name": cfg.name, "headline": card["headline"]}
