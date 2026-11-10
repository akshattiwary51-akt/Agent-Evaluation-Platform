import asyncio
import threading
import time
from collections import Counter

import pytest
from sqlalchemy import func, select, text

from app.agents.base import AgentResult
from app.db import models as m
from app.execution import queue
from app.execution.worker import Worker, load_task
from app.metrics import experiment_summary
from tests.pg.conftest import new_experiment

ONE = ["refund_eligible_004"]  # a single, simple task for focused lifecycle tests


def worker(factory, **kw):
    kw.setdefault("lease_seconds", 5.0)
    kw.setdefault("heartbeat_seconds", 0.1)
    kw.setdefault("concurrency", 1)
    return Worker(factory, **kw)


def drain(w):
    while w.run_once():
        pass


def runs(factory, exp_id):
    with factory() as s:
        return s.scalars(select(m.Run).where(m.Run.experiment_id == exp_id)).all()


def attempts(factory, run_id):
    with factory() as s:
        return s.scalars(select(m.Attempt).where(m.Attempt.run_id == run_id)
                         .order_by(m.Attempt.attempt_number)).all()


class Custom:
    """Adapter that runs a coroutine function; lets tests script timing/failure precisely."""
    name = "custom"

    def __init__(self, fn):
        self.fn = fn

    async def run(self, task_input, tools, events, limits):
        return await self.fn(task_input, tools, events, limits)


def factory_of(fn):
    return lambda config: Custom(fn)


# ------------------------------------------------------------------ queue semantics
def test_concurrent_claims_never_hand_out_a_job_twice(factory, ids):
    new_experiment(factory, ids, split="all")  # 50 jobs
    claimed, lock = [], threading.Lock()

    def grab(i):
        while (job := queue.claim_job(factory, f"w{i}", 30)) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(claimed) == 50 and len(set(claimed)) == 50


def test_claims_follow_creation_order(factory, ids):
    exp = new_experiment(factory, ids, split="test")
    first = queue.claim_job(factory, "w", 30)
    with factory() as s:
        earliest = s.scalars(select(m.Job).order_by(m.Job.created_at, m.Job.id)).first()
    assert first.id == earliest.id


def test_configs_are_interleaved_on_each_task(factory, ids):
    exp = new_experiment(factory, ids, configs=("scripted:reference_solver", "scripted:do_nothing"),
                         split="test")
    with factory() as s:
        rows = s.execute(select(m.TaskVersion.task_id, m.AgentConfig.name)
                         .select_from(m.Job).join(m.Run, m.Run.id == m.Job.run_id)
                         .join(m.TaskVersion, m.TaskVersion.id == m.Run.task_version_id)
                         .join(m.AgentConfig, m.AgentConfig.id == m.Run.agent_config_id)
                         .order_by(m.Job.created_at, m.Job.id)).all()
    pairs = [rows[i:i + 2] for i in range(0, len(rows), 2)]
    assert all(a[0] == b[0] and a[1] != b[1] for a, b in pairs)  # same task, both configs


def test_db_rejects_duplicate_runs_and_second_job_per_run(factory, ids):
    from sqlalchemy.exc import IntegrityError
    exp = new_experiment(factory, ids, task_ids=ONE)
    with factory() as s:
        r = s.scalars(select(m.Run)).one()
        s.add(m.Run(experiment_id=r.experiment_id, agent_config_id=r.agent_config_id,
                    task_version_id=r.task_version_id, repetition=r.repetition))
        with pytest.raises(IntegrityError):
            s.flush()
        s.rollback()
        s.add(m.Job(run_id=r.id))
        with pytest.raises(IntegrityError):
            s.flush()


# ------------------------------------------------------------------ happy path
def test_worker_completes_an_experiment_with_scores_and_traces(factory, ids):
    exp = new_experiment(factory, ids, split="test")
    drain(worker(factory))
    with factory() as s:
        summary = experiment_summary(s, s.get(m.Experiment, exp))
    assert summary["status"] == "completed"
    assert summary["execution"] == {"completed": summary["planned_runs"]}
    assert summary["evaluation"] == {"pass": summary["planned_runs"]}  # reference solver
    assert summary["metrics"]["evaluation_coverage"] == 1.0
    for r in runs(factory, exp):
        a = attempts(factory, r.id)
        assert len(a) == 1 and r.final_attempt_id == a[0].id and a[0].execution_status == "completed"
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(m.Score)) > 0
        assert s.scalar(select(func.count()).select_from(m.TraceEvent)) > 0
        assert s.scalar(select(func.count()).select_from(m.Job).where(m.Job.status != "succeeded")) == 0


def test_agent_failures_are_preserved_and_never_retried(factory, ids):
    exp = new_experiment(factory, ids, configs=("scripted:false_success",), split="test")
    drain(worker(factory))
    for r in runs(factory, exp):
        assert len(attempts(factory, r.id)) == 1           # not rerun until it passes
        assert r.execution_status == "completed" and r.evaluation_status == "fail"
    with factory() as s:
        assert s.scalar(select(func.sum(m.Job.retry_count))) == 0
        summary = experiment_summary(s, s.get(m.Experiment, exp))
    assert summary["metrics"]["verified_success_rate"] == 0.0


def test_concurrency_is_bounded(factory, ids):
    new_experiment(factory, ids, split="test")
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    async def slow(task_input, tools, events, limits):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.15)
        with lock:
            state["now"] -= 1
        return AgentResult("Order is delivered.", "model_final")

    w = worker(factory, concurrency=3, poll_seconds=0.05, adapter_factory=factory_of(slow))
    stop = threading.Event()
    t = threading.Thread(target=w.run_forever, args=(stop,))
    t.start()
    deadline = time.time() + 30
    while time.time() < deadline:
        with factory() as s:
            left = s.scalar(select(func.count()).select_from(m.Job).where(
                m.Job.status.in_(["queued", "running"])))
        if left == 0:
            break
        time.sleep(0.1)
    stop.set()
    t.join(10)
    assert left == 0 and 2 <= state["max"] <= 3


def test_graceful_shutdown_finishes_in_flight_work(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE)

    async def slow(task_input, tools, events, limits):
        await asyncio.sleep(0.6)
        return AgentResult("done", "model_final")

    w = worker(factory, poll_seconds=0.05, adapter_factory=factory_of(slow))
    stop = threading.Event()
    t = threading.Thread(target=w.run_forever, args=(stop,))
    t.start()
    time.sleep(0.3)           # job is in flight
    stop.set()
    t.join(10)
    assert not t.is_alive()
    (r,) = runs(factory, exp)
    assert r.execution_status == "completed"  # finished, not abandoned mid-run


# ------------------------------------------------------------------ infrastructure retries
def test_infrastructure_failure_is_retried_with_a_fresh_attempt_and_usage_is_kept(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE)
    calls = {"n": 0}

    async def flaky(task_input, tools, events, limits):
        calls["n"] += 1
        if calls["n"] == 1:
            events.model_calls.append({"provider": "p", "model": "m", "input_tokens": 500,
                                       "output_tokens": 50, "cached_input_tokens": 0,
                                       "duration_ms": 1.0})
            raise ConnectionError("provider unreachable")
        return AgentResult("Order is delivered.", "model_final")

    drain(worker(factory, adapter_factory=factory_of(flaky)))
    (r,) = runs(factory, exp)
    a = attempts(factory, r.id)
    assert [x.execution_status for x in a] == ["infrastructure_error", "completed"]
    assert a[0].environment_id != a[1].environment_id            # clean state on retry
    assert r.final_attempt_id == a[1].id and r.execution_status == "completed"
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(m.ModelCall).where(
            m.ModelCall.attempt_id == a[0].id)) == 1             # the failed attempt still cost money
        assert s.scalar(select(func.count()).select_from(m.Score).where(
            m.Score.attempt_id == a[0].id)) == 0                 # and was never scored


def test_retries_are_bounded_then_the_run_fails_unscored(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE, max_retries=1)

    async def down(*a):
        raise ConnectionError("provider unreachable")

    drain(worker(factory, adapter_factory=factory_of(down)))
    (r,) = runs(factory, exp)
    assert [x.execution_status for x in attempts(factory, r.id)] == ["infrastructure_error"] * 2
    assert r.execution_status == "infrastructure_error" and r.evaluation_status == "pending"
    assert r.final_attempt_id is None
    with factory() as s:
        job = s.scalars(select(m.Job)).one()
        assert job.status == "failed" and "unreachable" in job.last_error
        summary = experiment_summary(s, s.get(m.Experiment, exp))
    assert summary["metrics"]["evaluation_coverage"] == 0.0     # missing results are visible


def test_unbuildable_adapter_is_an_infrastructure_error_not_an_agent_failure(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE, max_retries=0)

    def boom(config):
        raise RuntimeError("ANTHROPIC_API_KEY is not set for this worker")

    drain(worker(factory, adapter_factory=boom))
    (r,) = runs(factory, exp)
    assert r.execution_status == "infrastructure_error" and r.evaluation_status == "pending"


# ------------------------------------------------------------------ leases and recovery
def claim_and_start(factory, lease=0.2):
    job = queue.claim_job(factory, "dead_worker", lease)
    attempt_id = queue.start_attempt(factory, job)
    return job, attempt_id


def test_expired_lease_interrupts_the_attempt_and_requeues(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE)
    job, attempt_id = claim_and_start(factory)
    time.sleep(0.4)
    assert queue.recover_expired(factory) == [(job.id, "requeued")]
    assert queue.recover_expired(factory) == []                  # idempotent
    (r,) = runs(factory, exp)
    assert r.execution_status == "queued"
    with factory() as s:
        a = s.get(m.Attempt, attempt_id)
        assert a.execution_status == "interrupted" and a.stop_reason == "lease_expired"
        ev = s.scalars(select(m.TraceEvent).where(m.TraceEvent.attempt_id == attempt_id)).one()
        assert ev.event_type == "attempt_interrupted"
        j = s.get(m.Job, job.id)
        assert j.status == "queued" and j.retry_count == 1 and j.lease_token is None

    drain(worker(factory))                                       # replacement runs cleanly
    (r,) = runs(factory, exp)
    a = attempts(factory, r.id)
    assert [x.execution_status for x in a] == ["interrupted", "completed"]
    assert r.final_attempt_id == a[1].id and r.execution_status == "completed"


def test_lease_loss_beyond_max_retries_fails_the_run(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE, max_retries=0)
    claim_and_start(factory)
    time.sleep(0.4)
    assert [a for _, a in queue.recover_expired(factory)] == ["failed"]
    (r,) = runs(factory, exp)
    assert r.execution_status == "infrastructure_error" and r.final_attempt_id is None


def test_healthy_worker_is_not_reclaimed_while_heartbeating(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE)

    async def slow(task_input, tools, events, limits):
        await asyncio.sleep(1.5)                    # far longer than the 0.5s lease
        return AgentResult("Order is delivered.", "model_final")

    w = worker(factory, lease_seconds=0.5, heartbeat_seconds=0.1, adapter_factory=factory_of(slow))
    job = queue.claim_job(factory, w.worker_id, 0.5)
    result = {}
    t = threading.Thread(target=lambda: result.update(r=w.process(job)))
    t.start()
    recovered = []
    while t.is_alive():
        recovered += queue.recover_expired(factory)
        time.sleep(0.1)
    t.join()
    assert recovered == [] and result["r"] == "finalized"
    (r,) = runs(factory, exp)
    assert len(attempts(factory, r.id)) == 1


def test_stale_worker_cannot_finalize_and_results_count_once(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE)
    from app.agents.scripted import ScriptedAdapter
    from app.execution.runner import run_attempt

    async def run_for(job, attempt_id):
        with factory() as s:
            task = load_task(s.get(m.TaskVersion, s.get(m.Run, job.run_id).task_version_id))
        return await run_attempt(task, ScriptedAdapter("reference_solver"), attempt_id=attempt_id)

    job_a, att_a = claim_and_start(factory, lease=0.2)
    out_a = asyncio.run(run_for(job_a, att_a))                   # A finishes its work...
    time.sleep(0.4)
    queue.recover_expired(factory)                               # ...but its lease expired
    job_b = queue.claim_job(factory, "worker_b", 30)
    att_b = queue.start_attempt(factory, job_b)
    # A tries to finalize while B holds the job: refused, nothing written.
    assert queue.finalize(factory, job_a, att_a, out_a) == "stale"
    out_b = asyncio.run(run_for(job_b, att_b))
    assert queue.finalize(factory, job_b, att_b, out_b) == "finalized"
    # A is still refused afterwards, and cannot add a second counted result.
    assert queue.finalize(factory, job_a, att_a, out_a) == "stale"

    (r,) = runs(factory, exp)
    assert r.final_attempt_id == att_b
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(m.Score).where(
            m.Score.attempt_id == att_a)) == 0
        assert s.scalar(select(func.count()).select_from(m.Score).where(
            m.Score.attempt_id == att_b)) > 0
        assert s.get(m.Attempt, att_a).execution_status == "interrupted"


def test_a_worker_that_loses_its_lease_stops_and_writes_nothing(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE)
    seen = {"calls": 0}

    async def loop(task_input, tools, events, limits):
        for _ in range(100):
            seen["calls"] += 1
            await tools.execute("search_policy", {"query": "refund"}, f"c{seen['calls']}")
            await asyncio.sleep(0.05)
        return AgentResult("x", "model_final")

    w = worker(factory, heartbeat_seconds=0.1, adapter_factory=factory_of(loop))
    job = queue.claim_job(factory, w.worker_id, 5.0)
    result = {}
    t = threading.Thread(target=lambda: result.update(r=w.process(job)))
    t.start()
    time.sleep(0.4)
    with factory() as s, s.begin():                              # someone else takes the job
        s.execute(text("UPDATE jobs SET lease_token = 'someone_else' WHERE id = :i"), {"i": job.id})
    t.join(10)
    assert result["r"] == "stale" and seen["calls"] < 30         # stopped well before finishing
    (r,) = runs(factory, exp)
    assert r.final_attempt_id is None
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(m.Score)) == 0


# ------------------------------------------------------------------ cancellation
def test_cancel_stops_queued_work(factory, ids):
    exp = new_experiment(factory, ids, split="test")
    with factory() as s:
        out = queue.cancel_experiment(s, exp)
        s.commit()
    assert out["cancelled_queued_jobs"] == 12
    assert not worker(factory).run_once()                        # nothing left to claim
    with factory() as s:
        summary = experiment_summary(s, s.get(m.Experiment, exp))
    assert summary["status"] == "cancelled" and summary["execution"] == {"cancelled": 12}


def test_cancel_stops_a_running_attempt_before_its_next_tool_call(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE)
    state = {"calls": 0, "at_cancel": None}

    async def loop(task_input, tools, events, limits):
        for _ in range(200):
            state["calls"] += 1
            await tools.execute("search_policy", {"query": "refund"}, f"c{state['calls']}")
            await asyncio.sleep(0.03)
        return AgentResult("x", "model_final")

    w = worker(factory, heartbeat_seconds=0.1, adapter_factory=factory_of(loop))
    job = queue.claim_job(factory, w.worker_id, 5.0)
    result = {}
    t = threading.Thread(target=lambda: result.update(r=w.process(job)))
    t.start()
    while state["calls"] < 3:
        time.sleep(0.01)
    with factory() as s:
        queue.cancel_experiment(s, exp)
        s.commit()
    state["at_cancel"] = state["calls"]
    t.join(10)
    assert result["r"] == "cancelled"
    assert state["calls"] - state["at_cancel"] < 15              # bounded by heartbeat interval
    (r,) = runs(factory, exp)
    assert r.execution_status == "cancelled" and r.evaluation_status == "pending"
    assert r.final_attempt_id is None                            # cancelled work is never scored
    with factory() as s:
        assert s.get(m.Job, job.id).status == "cancelled"


def test_cancel_that_arrives_after_claim_prevents_the_attempt_from_starting(factory, ids):
    exp = new_experiment(factory, ids, task_ids=ONE)
    job = queue.claim_job(factory, "w", 30)
    with factory() as s:
        queue.cancel_experiment(s, exp)
        s.commit()
    assert worker(factory).process(job) == "cancelled"
    (r,) = runs(factory, exp)
    assert r.execution_status == "cancelled" and attempts(factory, r.id) == []
