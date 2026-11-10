import pytest

from app.analysis.stats import (bootstrap_ci, paired_bootstrap_diff, percentile, rate,
                                repetition_metrics, summarize)


def test_rate_is_pooled_over_scorable_runs_not_averaged_per_task():
    assert rate([(1, 1), (1, 3)]) == pytest.approx(2 / 4)   # not mean(1.0, 0.33)
    assert rate([(0, 0)]) is None


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    units = [(1, 1)] * 30 + [(0, 1)] * 10
    a, b = bootstrap_ci(units, seed="x"), bootstrap_ci(units, seed="x")
    assert a == b and a[0] < rate(units) < a[1]
    assert bootstrap_ci(units, seed="y") != a                  # seed matters


def test_bootstrap_uses_tasks_as_units_so_repetitions_do_not_shrink_the_interval():
    # Same 10 tasks, 1 rep vs 5 reps each (identical per-task behavior): CI must NOT tighten.
    one = [(1, 1)] * 5 + [(0, 1)] * 5
    five = [(5, 5)] * 5 + [(0, 5)] * 5
    lo1, hi1 = bootstrap_ci(one, seed="s")
    lo5, hi5 = bootstrap_ci(five, seed="s")
    assert (hi5 - lo5) >= (hi1 - lo1) - 0.02


def test_bootstrap_refuses_to_invent_precision_from_one_task():
    assert bootstrap_ci([(1, 1)]) is None and bootstrap_ci([(1, 1), (0, 0)]) is None


def test_paired_difference_detects_a_real_gap_and_not_noise():
    base = {f"t{i}": (1 if i < 10 else 0, 1) for i in range(40)}
    cand = {f"t{i}": (1 if i < 35 else 0, 1) for i in range(40)}
    d = paired_bootstrap_diff(base, cand, seed="s")
    assert d["difference"] == pytest.approx(25 / 40) and d["excludes_zero"] and d["tasks"] == 40
    same = paired_bootstrap_diff(base, base, seed="s")
    assert same["difference"] == 0 and not same["excludes_zero"]


def test_paired_difference_only_uses_tasks_scored_on_both_sides():
    base = {"a": (1, 1), "b": (0, 1), "c": (0, 0)}
    cand = {"a": (1, 1), "b": (1, 1), "c": (1, 1)}
    assert paired_bootstrap_diff(base, cand, seed="s")["tasks"] == 2


def test_percentiles():
    xs = [1, 2, 3, 4, 100]
    assert percentile(xs, 50) == 3 and percentile(xs, 0) == 1 and percentile(xs, 100) == 100
    assert percentile(xs, 95) == pytest.approx(80.8)
    assert percentile([], 50) is None and summarize([]) is None


def test_repetition_metrics_distinguish_flaky_from_stable_tasks():
    m = repetition_metrics({"stable_pass": [True] * 3, "stable_fail": [False] * 3,
                            "flaky": [True, False, True], "missing_rep": [True, True, None]})
    assert m["repetitions"] == 3 and m["tasks"] == 4
    assert m["passed_all_repetitions"] == 1          # unscored repetition is NOT a pass
    assert m["inconsistent_tasks"] == ["flaky"]
    assert m["success_by_repetition"][0] == pytest.approx(3 / 4)
    assert m["std_across_repetitions"] is not None


def test_repetition_metrics_single_repetition_has_no_spread():
    m = repetition_metrics({"a": [True], "b": [False]})
    assert m["std_across_repetitions"] is None and m["mean_success"] == 0.5
