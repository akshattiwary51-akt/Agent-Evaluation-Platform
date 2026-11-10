import base64

import pytest
from sqlalchemy import select

from app.agents.base import AgentResult
from app.db import models as m
from app.main import app
from tests.pg.conftest import run_to_completion
from tests.pg.test_api import client_for

REF, CARELESS = "scripted:reference_solver", "scripted:careless_agent"


@pytest.fixture
def ui(factory):
    c = client_for(factory)
    yield c
    app.dependency_overrides.clear()


@pytest.fixture
def exp(factory, ids):
    return run_to_completion(factory, ids, [REF, CARELESS], split="test")


def first_run(factory, exp, config_name, ids, task_id=None):
    with factory() as s:
        q = select(m.Run).where(m.Run.experiment_id == exp, m.Run.agent_config_id == ids["cfg"][config_name])
        if task_id:
            q = q.join(m.TaskVersion, m.TaskVersion.id == m.Run.task_version_id).where(
                m.TaskVersion.task_id == task_id)
        return s.scalars(q.order_by(m.Run.id)).first().id


def test_every_screen_renders(ui, exp, ids, factory):
    page = ui.get("/ui")
    assert page.status_code == 200 and exp[:10] in page.text and REF in page.text
    detail = ui.get(f"/ui/experiments/{exp}")
    assert detail.status_code == 200 and "Scorecard" in detail.text and "Success by category" in detail.text
    assert "Failure reasons" in detail.text and f"/ui/compare?experiment_id={exp}" in detail.text
    run = ui.get(f"/ui/runs/{first_run(factory, exp, REF, ids)}")
    assert run.status_code == 200
    cmp_page = ui.get("/ui/compare", params={"experiment_id": exp, "baseline": REF, "candidate": CARELESS})
    assert cmp_page.status_code == 200 and "Regressed cases" in cmp_page.text
    assert ui.get("/ui/benchmarks").status_code == 200
    assert "No public-benchmark results" in ui.get("/ui/benchmarks").text
    for bad in ("/ui/experiments/nope", "/ui/runs/nope"):
        assert ui.get(bad).status_code == 404


def test_run_page_keeps_hidden_ground_truth_visibly_separate(ui, exp, ids, factory):
    html = ui.get(f"/ui/runs/{first_run(factory, exp, REF, ids)}").text
    seen, hidden = html.index("What the agent saw"), html.index("Hidden ground truth")
    assert seen < hidden
    visible_panel = html[seen:hidden]
    assert "paid_amount_minor" not in visible_panel and "refund_total_minor" not in visible_panel
    assert "never shown to the agent" in html
    assert "Evaluator findings" in html and "Final-state differences" in html and "Trace" in html


def test_failed_run_shows_what_went_wrong_and_which_call_caused_it(ui, exp, ids, factory):
    rid = first_run(factory, exp, CARELESS, ids, task_id="refund_ineligible_001")
    html = ui.get(f"/ui/runs/{rid}").text
    assert "FALSE_REFUND_CLAIM" in html and "evaluation: fail" in html
    assert "APPROVAL_INVALID" in html or "issue_refund" in html           # the blocked call is in the trace
    assert "tool_rejected" in html


def test_comparison_page_lists_regressed_tasks_and_ci(ui, exp):
    html = ui.get("/ui/compare", params={"experiment_id": exp, "baseline": CARELESS,
                                         "candidate": REF}).text
    assert "Improved cases" in html and "refund_ineligible_" in html
    assert "95% CI" in html and "Same-task outcomes" in html


def test_incompatible_comparison_page_explains_instead_of_comparing(ui, exp, factory, ids):
    other = run_to_completion(factory, ids, [CARELESS], split="validation")
    html = ui.get("/ui/compare", params={"experiment_id": exp, "baseline": REF, "candidate": CARELESS,
                                         "candidate_experiment_id": other}).text
    assert "can't be compared fairly" in html and "different splits" in html
    assert "Regressed cases" not in html


def test_run_list_filters_work(ui, exp, ids):
    failed = ui.get(f"/ui/experiments/{exp}", params={"evaluation_status": "fail"}).text
    assert "careless" in failed and 'class="badge fail"' in failed
    only_ref = ui.get(f"/ui/experiments/{exp}", params={"config": REF, "category": "lookup"}).text
    assert "lookup_" in only_ref and "refund_ineligible_" not in only_ref.split('id="runs"')[1]


def test_model_output_is_escaped_not_executed(ui, factory, ids):
    payload = "<script>alert('x')</script><img src=x onerror=alert(1)>"

    async def evil(task_input, tools, events, limits):
        return AgentResult(payload, "model_final")

    class Evil:
        name = "evil"
        async def run(self, *a):
            return await evil(*a)

    e = run_to_completion(factory, ids, [REF], task_ids=["lookup_001"], adapter_factory=lambda c: Evil())
    rid = first_run(factory, e, REF, ids)
    html = ui.get(f"/ui/runs/{rid}").text
    assert "<script>alert" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_experiment_with_unknown_cost_shows_unknown_not_zero(ui, factory, ids):
    from tests.pg.test_analysis import Priced
    e = run_to_completion(factory, ids, [REF], task_ids=["refund_eligible_004"],
                          adapter_factory=lambda c: Priced(2))        # usage but no price registered
    page = ui.get("/ui").text
    assert ">unknown<" in page
    assert ">unknown<" in ui.get(f"/ui/experiments/{e}").text


def test_empty_state(ui):
    assert "No experiments yet" in ui.get("/ui").text


# ------------------------------------------------------------------ auth
def basic(user, pw):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


def test_password_protects_api_and_dashboard_but_not_health(ui, exp, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    for path in ("/ui", f"/experiments/{exp}", "/datasets", f"/experiments/{exp}/export"):
        r = ui.get(path)
        assert r.status_code == 401 and "Basic" in r.headers["www-authenticate"], path
    assert ui.get("/ui", headers=basic("admin", "wrong")).status_code == 401
    assert ui.get("/ui", headers=basic("root", "s3cret")).status_code == 401
    assert ui.get("/ui", headers=basic("admin", "s3cret")).status_code == 200
    assert ui.get("/datasets", headers=basic("admin", "s3cret")).status_code == 200
    assert ui.get("/health/live").status_code == 200 and ui.get("/health/ready").status_code == 200
    assert ui.post("/experiments", json={}).status_code == 401           # writes protected too


def test_no_password_means_open(ui, monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert ui.get("/ui").status_code == 200
