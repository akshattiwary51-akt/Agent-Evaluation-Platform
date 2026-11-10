import asyncio
import json

import pytest
from sqlalchemy import select, text

from app.agents.base import AgentResult
from app.agents.scripted import ScriptedAdapter
from app.analysis import comparison, costing, scorecard
from app.db import models as m
from app.execution.adapters import build_adapter
from tests.pg.conftest import new_experiment, run_to_completion

REF, CARELESS, CAREFUL = ("scripted:reference_solver", "scripted:careless_agent",
                          "scripted:careful_verbose_agent")
ONE = ["refund_eligible_004"]


def card(factory, exp_id, config_name, ids):
    with factory() as s:
        facts = scorecard.load_run_facts(s, exp_id, ids["cfg"][config_name])
        return scorecard.build_scorecard(facts, costing.load_prices(s), seed="t")


def cmp(factory, exp_id, base, cand, ids):
    with factory() as s:
        return comparison.compare(s, comparison.Side(exp_id, ids["cfg"][base]),
                                  comparison.Side(exp_id, ids["cfg"][cand]))


@pytest.fixture
def full(factory, ids):
    return run_to_completion(factory, ids, [REF, CARELESS, CAREFUL])


# ------------------------------------------------------------------ denominators
def test_headline_metrics_and_denominators(factory, ids, full):
    ref, bad = card(factory, full, REF, ids), card(factory, full, CARELESS, ids)
    h = ref["headline"]
    assert (h["planned_runs"], h["scorable_runs"], h["passed_runs"]) == (50, 50, 50)
    assert h["verified_success_rate"] == h["strict_end_to_end_success"] == 1.0
    assert h["evaluation_coverage"] == 1.0
    b = bad["headline"]
    assert b["passed_runs"] == 26 and b["verified_success_rate"] == pytest.approx(26 / 50)
    assert b["verified_success_ci95"][0] < 26 / 50 < b["verified_success_ci95"][1]


def test_unscored_runs_lower_strict_success_and_coverage_but_not_verified_success(factory, ids, full):
    with factory() as s, s.begin():                       # 10 careless runs lose their result
        run_ids = [r.id for r in s.scalars(select(m.Run).where(
            m.Run.agent_config_id == ids["cfg"][CARELESS])
            .order_by(m.Run.id).limit(10))]
        s.execute(text("UPDATE runs SET execution_status='infrastructure_error', "
                       "evaluation_status='pending', final_attempt_id=NULL WHERE id = ANY(:ids)"),
                  {"ids": run_ids})
    h = card(factory, full, CARELESS, ids)["headline"]
    assert h["planned_runs"] == 50 and h["scorable_runs"] == 40 and h["evaluation_coverage"] == 0.8
    assert h["strict_end_to_end_success"] == pytest.approx(h["passed_runs"] / 50)
    assert h["verified_success_rate"] == pytest.approx(h["passed_runs"] / 40)
    assert h["strict_end_to_end_success"] < h["verified_success_rate"]  # missing != success


def test_unsafe_attempts_and_executed_violations_are_separate_rates(factory, ids, full):
    bad = card(factory, full, CARELESS, ids)["hard_violations"]
    assert bad["runs_with_unsafe_attempt"] == 15 and bad["unsafe_attempt_rate"] == pytest.approx(0.3)
    assert bad["executed_violation_rate"] == 0.0          # the gateway blocked every one
    ref = card(factory, full, REF, ids)["hard_violations"]
    assert ref["executed_violation_rate"] == 0.0 and ref["unsafe_attempt_rate"] < bad["unsafe_attempt_rate"]


def test_tool_behavior_metrics_separate_recall_validity_and_usefulness(factory, ids, full):
    ref, bad, careful = (card(factory, full, c, ids)["tool_behavior"] for c in (REF, CARELESS, CAREFUL))
    assert ref["required_call_recall"] == 1.0 and bad["required_call_recall"] < 1.0
    assert ref["schema_validity"] == 1.0
    assert careful["mean_calls_per_run"] > ref["mean_calls_per_run"]
    assert careful["useful_call_rate"] < ref["useful_call_rate"]   # redundant lookups are not useful
    assert careful["redundant_calls"] > ref["redundant_calls"]


def test_failure_reasons_and_category_breakdown(factory, ids, full):
    bad = card(factory, full, CARELESS, ids)
    reasons = {r["reason"]: r["runs"] for r in bad["failure_reasons"]}
    assert reasons["response:no_false_refund_claim:FALSE_REFUND_CLAIM"] > 0
    assert bad["by_category"]["refund_ineligible"]["passed"] == 0
    assert bad["by_category"]["refund_eligible"]["verified_success"] == 1.0
    assert sum(c["planned"] for c in bad["by_category"].values()) == 50


# ------------------------------------------------------------------ cost
class Priced:
    """Wraps the reference solver and reports model usage. First call to an instance with
    fail_first=True emits usage then dies, like a provider outage mid-attempt."""
    name = "priced"

    def __init__(self, calls, fail=False, tokens=(1000, 100, 0)):
        self.calls, self.fail, self.tokens = calls, fail, tokens

    async def run(self, task_input, tools, events, limits):
        for _ in range(self.calls):
            i, o, c = self.tokens
            events.model_calls.append({"provider": "test", "model": "tm", "input_tokens": i,
                                       "output_tokens": o, "cached_input_tokens": c,
                                       "duration_ms": 1.0})
        if self.fail:
            raise ConnectionError("provider unreachable")
        return await ScriptedAdapter("reference_solver").run(task_input, tools, events, limits)


def add_price(factory, rate_in=3.0, rate_out=15.0, effective="2020-01-01T00:00:00+00:00"):
    from datetime import datetime
    with factory() as s:
        s.add(m.PriceVersion(provider="test", model="tm", currency="USD", input_per_million=rate_in,
                             output_per_million=rate_out, cached_input_per_million=0.3,
                             effective_date=datetime.fromisoformat(effective),
                             source="unit-test rate card"))
        s.commit()


def priced_experiment(factory, ids, **kw):
    state = {"n": 0}

    def make(cfg):
        state["n"] += 1
        return Priced(calls=1 if state["n"] == 1 else 2, fail=state["n"] == 1, **kw)

    return run_to_completion(factory, ids, [REF], task_ids=ONE, adapter_factory=make)


def test_cost_includes_retried_attempts_and_uses_the_stated_pricing_basis(factory, ids):
    add_price(factory)
    exp = priced_experiment(factory, ids)
    c = card(factory, exp, REF, ids)
    per_call = 1000 * 3.0 / 1e6 + 100 * 15.0 / 1e6                   # 0.0045
    assert c["usage"]["attempts"] == 2 and c["usage"]["retried_or_interrupted_attempts"] == 1
    assert c["usage"]["model_calls"] == 3                             # 1 (failed attempt) + 2
    assert c["cost"]["status"] == "estimated"
    assert c["cost"]["total"] == pytest.approx(3 * per_call)
    assert c["cost"]["per_successful_run"] == pytest.approx(3 * per_call)
    (basis,) = c["cost"]["pricing_basis"]
    assert basis["currency"] == "USD" and basis["source"] == "unit-test rate card"
    assert basis["effective_date"] == "2020-01-01"


def test_missing_price_or_usage_means_unknown_cost_never_free(factory, ids):
    exp = priced_experiment(factory, ids)                              # no price registered
    c = card(factory, exp, REF, ids)["cost"]
    assert c["status"] == "unknown" and c["total"] is None and c["per_successful_run"] is None
    assert c["unknown_calls"] == 3

    add_price(factory)
    exp2 = run_to_completion(factory, ids, [CAREFUL], task_ids=ONE, adapter_factory=lambda c: Priced(
        2, tokens=(None, None, None)))
    c2 = card(factory, exp2, CAREFUL, ids)
    assert c2["cost"]["status"] == "unknown" and c2["usage"]["unknown_usage_calls"] == 2


def test_price_in_effect_at_attempt_time_is_used(factory, ids):
    add_price(factory, effective="2999-01-01T00:00:00+00:00")           # not yet in effect
    exp = priced_experiment(factory, ids)
    assert card(factory, exp, REF, ids)["cost"]["status"] == "unknown"
    add_price(factory, rate_in=6.0, rate_out=30.0, effective="2020-06-01T00:00:00+00:00")
    add_price(factory, rate_in=999.0, rate_out=999.0, effective="2999-06-01T00:00:00+00:00")
    exp2 = priced_experiment(factory, ids)
    expected = 3 * (1000 * 6.0 / 1e6 + 100 * 30.0 / 1e6)
    assert card(factory, exp2, REF, ids)["cost"]["total"] == pytest.approx(expected)


def test_scripted_agents_incur_no_model_cost(factory, ids, full):
    c = card(factory, full, REF, ids)["cost"]
    assert c["status"] == "none_incurred" and c["total"] == 0.0


def test_projection_scales_pilot_cost_with_headroom(factory, ids):
    from tests.pg.test_api import client_for
    from app.main import app
    add_price(factory)
    exp = priced_experiment(factory, ids)
    try:
        r = client_for(factory).get(f"/experiments/{exp}/projection",
                                    params={"planned_runs": 100, "headroom": 0.2}).json()
    finally:
        app.dependency_overrides.clear()
    out = r[REF]
    assert out["mean_cost_per_run"] == pytest.approx(0.0135)       # whole pilot cost / 1 run
    assert out["estimate"] == pytest.approx(0.0135 * 100 * 1.2)
    assert "not a billing guarantee" in out["note"]


# ------------------------------------------------------------------ repeated trials
def test_repeated_trials_report_variability_not_just_a_mean(factory, ids):
    state = {"n": 0}

    def alternate(cfg):          # passes, fails, passes, ...: the same task is inconsistent
        state["n"] += 1
        return ScriptedAdapter("reference_solver" if state["n"] % 2 else "do_nothing")

    # An ODD number of tasks per repetition makes the alternation shift between repetitions, so
    # every task sees both outcomes (with an even number, each task would be consistently
    # passing or failing and the test would prove nothing).
    tasks = ["lookup_001", "lookup_002", "lookup_003"]
    exp = run_to_completion(factory, ids, [REF], task_ids=tasks, reps=3, adapter_factory=alternate)
    c = card(factory, exp, REF, ids)
    rt = c["repeated_trials"]
    assert rt["repetitions"] == 3 and rt["tasks"] == 3
    assert rt["passed_all_repetitions"] == 0 and len(rt["inconsistent_tasks"]) == 3
    assert rt["std_across_repetitions"] is not None and rt["mean_success"] == pytest.approx(5 / 9, abs=0.12)
    assert c["headline"]["passed_runs"] == 5 and c["headline"]["verified_success_ci95"] is not None


def test_single_task_gets_no_confidence_interval(factory, ids):
    exp = run_to_completion(factory, ids, [REF], task_ids=ONE, reps=3)
    h = card(factory, exp, REF, ids)["headline"]
    assert h["verified_success_ci95"] is None and h["passed_runs"] == 3


# ------------------------------------------------------------------ comparison
def test_comparison_names_the_exact_tasks_that_improved_and_regressed(factory, ids, full):
    up = cmp(factory, full, CARELESS, REF, ids)                      # baseline -> better candidate
    assert up["task_outcomes"] == {"improved": 24, "regressed": 0, "same_pass": 26,
                                   "same_fail": 0, "not_comparable": 0}
    assert up["success_difference"]["excludes_zero"] and up["success_difference"]["difference"] == pytest.approx(0.48)
    assert "verified_success_rate" in up["metrics_better"]
    down = cmp(factory, full, REF, CARELESS, ids)                    # same pair, reversed
    assert down["task_outcomes"]["regressed"] == 24 and down["task_outcomes"]["improved"] == 0
    cases = down["regressed_cases"]
    assert {c["task_id"] for c in cases} == {c["task_id"] for c in up["improved_cases"]}
    assert all(c["candidate_failing"] and not c["baseline_failing"] for c in cases)
    assert {c["category"] for c in cases} >= {"refund_ineligible", "duplicate", "tool_failure"}


def test_equal_success_but_costlier_behavior_shows_as_a_tradeoff(factory, ids, full):
    r = cmp(factory, full, REF, CAREFUL, ids)
    assert r["task_outcomes"]["same_pass"] == 50
    assert not r["success_difference"]["excludes_zero"] and r["success_difference"]["difference"] == 0
    assert "mean_calls_per_run" in r["metrics_worse"] and "useful_call_rate" in r["metrics_worse"]
    assert "verified_success_rate" not in r["metrics_better"] + r["metrics_worse"]


def test_tasks_unscored_on_a_side_are_not_called_wins_or_losses(factory, ids, full):
    with factory() as s, s.begin():
        tv = s.scalars(select(m.TaskVersion).where(m.TaskVersion.task_id == "lookup_001")).one()
        s.execute(text("UPDATE runs SET execution_status='infrastructure_error', "
                       "evaluation_status='pending', final_attempt_id=NULL "
                       "WHERE agent_config_id=:c AND task_version_id=:t"),
                  {"c": ids["cfg"][CARELESS], "t": tv.id})
    r = cmp(factory, full, REF, CARELESS, ids)
    assert r["task_outcomes"]["not_comparable"] == 1
    assert r["success_difference"]["tasks"] == 49
    assert any("unscored" in w for w in r["warnings"]) and any("coverage differs" in w for w in r["warnings"])
    row = next(t for t in r["tasks"] if t["task_id"] == "lookup_001")
    assert row["verdict"] == "not_comparable"


# ------------------------------------------------------------------ compatibility
def side(exp, cfg, ids):
    return comparison.Side(exp, ids["cfg"][cfg])


def test_cross_experiment_comparison_works_when_compatible(factory, ids):
    a = run_to_completion(factory, ids, [REF])
    b = run_to_completion(factory, ids, [CARELESS])
    with factory() as s:
        r = comparison.compare(s, side(a, REF, ids), side(b, CARELESS, ids))
    assert r["compatible"] and r["task_outcomes"]["regressed"] == 24


def expect_422(factory, a, b, ids, needle, cfgs=(REF, CARELESS)):
    with factory() as s, pytest.raises(comparison.ComparisonError) as e:
        comparison.compare(s, side(a, cfgs[0], ids), side(b, cfgs[1], ids))
    assert e.value.status_code == 422 and any(needle in p for p in e.value.problems), e.value.problems


def test_incompatible_comparisons_are_refused_not_silently_made(factory, ids):
    a = run_to_completion(factory, ids, [REF], split="all")
    other_split = run_to_completion(factory, ids, [CARELESS], split="test")
    expect_422(factory, a, other_split, ids, "different splits")

    b = run_to_completion(factory, ids, [CARELESS], split="all")
    with factory() as s, s.begin():                                  # tamper: different dataset content
        e = s.get(m.Experiment, b)
        e.manifest = {**e.manifest, "dataset_content_hash": "0" * 64}
    expect_422(factory, a, b, ids, "different dataset content")

    c = run_to_completion(factory, ids, [CARELESS], split="all")
    with factory() as s, s.begin():                                  # tamper: newer evaluator
        s.execute(text("UPDATE scores SET evaluator_version='2.0.0' WHERE evaluator_name='outcome' "
                       "AND attempt_id IN (SELECT final_attempt_id FROM runs WHERE experiment_id=:e)"),
                  {"e": c})
    expect_422(factory, a, c, ids, "evaluator versions")

    expect_422(factory, a, a, ids, "same configuration", cfgs=(REF, REF))
    with factory() as s, pytest.raises(comparison.ComparisonError) as e:
        comparison.compare(s, side(a, REF, ids), side(a, "scripted:eager_refund", ids))
    assert e.value.status_code == 404                                 # config not in that experiment


# ------------------------------------------------------------------ export
def test_export_report_is_reproducible_and_complete(factory, ids):
    from tests.pg.test_api import client_for
    from app.main import app
    exp = run_to_completion(factory, ids, [REF, CARELESS], split="test")
    try:
        c = client_for(factory)
        rep = c.get(f"/experiments/{exp}/export").json()
        csv_text = c.get(f"/experiments/{exp}/export", params={"format": "csv"}).text
        bad = c.get(f"/experiments/{exp}/export", params={"format": "xml"})
    finally:
        app.dependency_overrides.clear()
    r = rep["reproducibility"]
    assert len(r["dataset_content_hash"]) == 64 and r["environment_version"] == "support-env-v1"
    assert {"outcome@1.0.0", "policy@1.0.0", "trajectory@1.0.0", "response@1.0.0"} <= set(r["evaluator_versions"])
    assert all(len(cfg["config_hash"]) == 64 for cfg in r["configs"])
    assert r["dependencies"]["installed_distributions_sha256"] and r["agent_code_commit"]
    assert set(rep["scorecards"]) == {REF, CARELESS} and len(rep["runs"]) == 24
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("experiment_id,task_id,category") and len(lines) == 25
    assert bad.status_code == 422
