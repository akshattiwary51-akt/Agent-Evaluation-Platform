"""Pure statistics helpers (no database). Task-level units, so repetitions of one task stay together.

A "task unit" is (passed, scorable): how many of a task's repetitions passed and how many were
scorable. Treating the TASK as the resampling unit matters: repetitions of one task are not
independent evidence, so bootstrapping individual runs would overstate precision.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

TaskUnit = tuple[int, int]  # (passed, scorable)


def rate(units: list[TaskUnit]) -> float | None:
    scorable = sum(s for _, s in units)
    return sum(p for p, _ in units) / scorable if scorable else None


def bootstrap_ci(units: list[TaskUnit], *, iterations: int = 2000, seed: str = "",
                 confidence: float = 0.95) -> tuple[float, float] | None:
    """Percentile CI for the pooled success rate, resampling tasks with replacement."""
    units = [u for u in units if u[1] > 0]
    if len(units) < 2:
        return None  # one task cannot say anything about task-to-task variation
    rng = random.Random(seed)
    n = len(units)
    stats = []
    for _ in range(iterations):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        stats.append(rate(sample))
    stats.sort()
    lo = stats[int((1 - confidence) / 2 * iterations)]
    hi = stats[min(iterations - 1, int((1 + confidence) / 2 * iterations))]
    return lo, hi


def paired_bootstrap_diff(base: dict[str, TaskUnit], cand: dict[str, TaskUnit], *,
                          iterations: int = 2000, seed: str = "",
                          confidence: float = 0.95) -> dict | None:
    """CI for (candidate rate - baseline rate), resampling the SAME tasks for both sides."""
    ids = sorted(t for t in base if t in cand and base[t][1] > 0 and cand[t][1] > 0)
    if len(ids) < 2:
        return None
    rng = random.Random(seed)
    n = len(ids)
    diffs = []
    for _ in range(iterations):
        pick = [ids[rng.randrange(n)] for _ in range(n)]
        diffs.append(rate([cand[t] for t in pick]) - rate([base[t] for t in pick]))
    diffs.sort()
    lo = diffs[int((1 - confidence) / 2 * iterations)]
    hi = diffs[min(iterations - 1, int((1 + confidence) / 2 * iterations))]
    point = rate([cand[t] for t in ids]) - rate([base[t] for t in ids])
    return {"difference": point, "ci_low": lo, "ci_high": hi, "tasks": n,
            "confidence": confidence, "excludes_zero": lo > 0 or hi < 0}


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"mean": sum(values) / len(values), "p50": percentile(values, 50),
            "p95": percentile(values, 95), "max": max(values), "n": len(values)}


def repetition_metrics(per_task: dict[str, list[bool | None]]) -> dict:
    """per_task maps task -> outcome per repetition (True/False, None = not scorable).

    passed_all_repetitions is a plain count under that name, NOT a named benchmark's reliability
    estimator (e.g. pass^k). Unscored repetitions count as not passed (strict).
    """
    reps = max((len(v) for v in per_task.values()), default=0)
    per_rep = []
    for i in range(reps):
        outcomes = [v[i] for v in per_task.values() if i < len(v) and v[i] is not None]
        per_rep.append(sum(outcomes) / len(outcomes) if outcomes else None)
    valid = [r for r in per_rep if r is not None]
    mean = sum(valid) / len(valid) if valid else None
    std = (math.sqrt(sum((r - mean) ** 2 for r in valid) / (len(valid) - 1))
           if len(valid) > 1 else None)
    flaky = [t for t, v in per_task.items()
             if (scored := [x for x in v if x is not None]) and 0 < sum(scored) < len(scored)]
    return {"repetitions": reps, "success_by_repetition": per_rep, "mean_success": mean,
            "std_across_repetitions": std,
            "passed_all_repetitions": sum(1 for v in per_task.values()
                                          if len(v) == reps and all(x is True for x in v)),
            "tasks": len(per_task), "inconsistent_tasks": sorted(flaky)}
