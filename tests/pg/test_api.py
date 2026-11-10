import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api.deps import get_session
from app.db import models as m
from app.execution.worker import Worker
from app.main import app


def client_for(factory) -> TestClient:
    def override():
        with factory() as s:
            yield s
    app.dependency_overrides[get_session] = override
    return TestClient(app)


@pytest.fixture
def api(factory):
    c = client_for(factory)
    yield c
    app.dependency_overrides.clear()


def body(ids, **kw):
    return {"dataset_version_id": ids["dv"], "agent_config_ids": [ids["cfg"]["scripted:reference_solver"]],
            "split": "test", **kw}


def drain(factory):
    w = Worker(factory, concurrency=1, lease_seconds=5)
    while w.run_once():
        pass


def test_health_ready_checks_the_database(api):
    assert api.get("/health/ready").json() == {"status": "ok"}


def test_datasets_and_configs_are_listed(api, ids):
    (ds,) = api.get("/datasets").json()
    assert ds["name"] == "support" and ds["versions"][0]["task_count"] == 50
    from app.agents.scripted import BEHAVIORS
    assert len(api.get("/agent-configs").json()) == len(BEHAVIORS)


def test_create_experiment_returns_planned_runs_and_queues_jobs(api, factory, ids):
    r = api.post("/experiments", json=body(ids, repetitions=2))
    assert r.status_code == 201
    out = r.json()
    assert out["status"] == "queued" and out["planned_runs"] == 24      # 12 tasks x 1 config x 2
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(m.Job).where(m.Job.status == "queued")) == 24
    exp = api.get(f"/experiments/{out['experiment_id']}").json()
    assert exp["status"] == "queued" and exp["manifest"]["dataset_content_hash"]


def test_queued_work_survives_an_api_restart(api, factory, ids):
    exp_id = api.post("/experiments", json=body(ids)).json()["experiment_id"]
    app.dependency_overrides.clear()
    fresh = client_for(factory)                                          # a brand-new "process"
    assert fresh.get(f"/experiments/{exp_id}").json()["execution"] == {"queued": 12}
    drain(factory)
    done = fresh.get(f"/experiments/{exp_id}").json()
    assert done["status"] == "completed" and done["metrics"]["verified_success_rate"] == 1.0


def test_idempotency_key_replays_and_rejects_conflicting_reuse(api, factory, ids):
    h = {"Idempotency-Key": "client-req-1"}
    a = api.post("/experiments", json=body(ids), headers=h)
    b = api.post("/experiments", json=body(ids), headers=h)
    assert (a.status_code, b.status_code) == (201, 200)
    assert a.json()["experiment_id"] == b.json()["experiment_id"]
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(m.Experiment)) == 1
        assert s.scalar(select(func.count()).select_from(m.Run)) == 12   # no duplicated runs
    assert api.post("/experiments", json=body(ids, repetitions=2), headers=h).status_code == 409
    # Equivalent payloads (e.g. reordered/duplicated config ids) are recognised as the same.
    cfg = ids["cfg"]["scripted:reference_solver"]
    same = api.post("/experiments", json=body(ids) | {"agent_config_ids": [cfg, cfg]}, headers=h)
    assert same.status_code == 200
    assert api.post("/experiments", json=body(ids)).status_code == 201   # no key: new experiment


def test_concurrent_requests_with_one_key_create_exactly_one_experiment(factory, ids):
    app.dependency_overrides.clear()
    results = []

    def go():
        c = client_for(factory)
        results.append(c.post("/experiments", json=body(ids), headers={"Idempotency-Key": "race"}))

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    app.dependency_overrides.clear()
    assert {r.status_code for r in results} <= {200, 201} and \
        sum(r.status_code == 201 for r in results) == 1
    assert len({r.json()["experiment_id"] for r in results}) == 1
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(m.Experiment)) == 1
        assert s.scalar(select(func.count()).select_from(m.Job)) == 12


@pytest.mark.parametrize("patch,code", [
    ({"dataset_version_id": "nope"}, 404),
    ({"agent_config_ids": ["nope"]}, 404),
    ({"task_ids": ["not_a_task"]}, 422),
    ({"repetitions": 0}, 422),
    ({"repetitions": 11}, 422),
    ({"split": "bogus"}, 422),
    ({"extra_field": 1}, 422),
])
def test_invalid_experiment_requests_are_rejected_and_create_nothing(api, factory, ids, patch, code):
    assert api.post("/experiments", json=body(ids) | patch).status_code == code
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(m.Experiment)) == 0


def test_runs_listing_filters_and_paginates(api, factory, ids):
    exp_id = api.post("/experiments", json=body(ids)).json()["experiment_id"]
    drain(factory)
    page = api.get(f"/experiments/{exp_id}/runs", params={"limit": 5, "offset": 0}).json()
    assert page["total"] == 12 and len(page["items"]) == 5
    rest = api.get(f"/experiments/{exp_id}/runs", params={"limit": 5, "offset": 10}).json()
    assert len(rest["items"]) == 2
    only = api.get(f"/experiments/{exp_id}/runs", params={"category": "lookup"}).json()
    assert only["total"] >= 1 and all(i["category"] == "lookup" for i in only["items"])
    assert api.get(f"/experiments/{exp_id}/runs", params={"evaluation_status": "fail"}).json()["total"] == 0


def test_run_detail_separates_agent_visible_from_hidden_ground_truth(api, factory, ids):
    exp_id = api.post("/experiments", json=body(ids, task_ids=["refund_eligible_004"])).json()["experiment_id"]
    drain(factory)
    (item,) = api.get(f"/experiments/{exp_id}/runs").json()["items"]
    run = api.get(f"/runs/{item['run_id']}").json()
    assert set(run["agent_visible"]) == {"task_id", "category", "user_message",
                                         "authenticated_customer_id", "ticket_id"}
    assert "expected" in run["evaluator_visible"] and "fixture" in run["evaluator_visible"]
    assert "249900" not in str(run["agent_visible"])
    assert run["attempts"][0]["execution_status"] == "completed" and run["final_response"]
    assert any(s["evaluator"] == "outcome" and s["status"] == "pass" for s in run["scores"])


def test_trace_endpoint_is_ordered_and_pageable(api, factory, ids):
    exp_id = api.post("/experiments", json=body(ids, task_ids=["refund_eligible_004"])).json()["experiment_id"]
    drain(factory)
    (item,) = api.get(f"/experiments/{exp_id}/runs").json()["items"]
    aid = item["final_attempt_id"]
    events = api.get(f"/attempts/{aid}/events").json()["events"]
    assert [e["sequence"] for e in events] == list(range(len(events)))
    assert events[0]["type"] == "attempt_started" and events[-1]["type"] == "attempt_finished"
    tail = api.get(f"/attempts/{aid}/events", params={"after": 3, "limit": 2}).json()["events"]
    assert [e["sequence"] for e in tail] == [4, 5]


def test_cancel_endpoint(api, factory, ids):
    exp_id = api.post("/experiments", json=body(ids)).json()["experiment_id"]
    out = api.post(f"/experiments/{exp_id}/cancel").json()
    assert out["cancelled_queued_jobs"] == 12
    assert api.get(f"/experiments/{exp_id}").json()["status"] == "cancelled"
    assert api.post("/experiments/nope/cancel").status_code == 404


def test_agent_configs_are_immutable(api):
    spec = {"name": "my-cfg", "adapter": "scripted", "settings": {"behavior": "correct_refund"}}
    assert api.post("/agent-configs", json=spec).status_code == 201
    assert api.post("/agent-configs", json=spec).status_code == 200            # same content: no-op
    changed = spec | {"settings": {"behavior": "false_success"}}
    assert api.post("/agent-configs", json=changed).status_code == 409         # edit needs a new name
    assert api.post("/agent-configs", json=spec | {"settings": {"behavior": "nope"}}).status_code == 422
    assert api.post("/agent-configs", json={"name": "t", "adapter": "tool_loop",
                    "model": "m", "settings": {"provider": "anthropic", "prompt": "support-v2"}}
                    ).status_code == 201
    assert api.post("/agent-configs", json={"name": "t2", "adapter": "tool_loop", "model": "m",
                    "settings": {"provider": "anthropic", "prompt": "missing"}}).status_code == 422
