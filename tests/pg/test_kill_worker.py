"""Demo 3 / Phase 4 acceptance: kill a worker process mid-attempt; recovery is explicit and clean."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from sqlalchemy import func, select

from app.db import models as m
from app.execution import queue
from app.execution.worker import Worker
from tests.pg.conftest import new_experiment

ROOT = Path(__file__).resolve().parents[2]


def test_killing_a_worker_yields_an_interrupted_attempt_and_one_clean_counted_result(factory, ids):
    exp = new_experiment(factory, ids, task_ids=["refund_eligible_004"])
    proc = subprocess.Popen([sys.executable, "-m", "tests.pg.hang_worker", "1.5"], cwd=ROOT,
                            stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    try:
        assert proc.stdout.readline().strip() == "HUNG"           # attempt is genuinely in flight
        with factory() as s:
            (a,) = s.scalars(select(m.Attempt)).all()
            assert a.execution_status == "running"
        proc.send_signal(signal.SIGKILL)                          # no cleanup, no goodbye
        proc.wait(10)
    finally:
        if proc.poll() is None:
            proc.kill()

    # While the lease is still valid nothing is reclaimed (we cannot yet know the worker is dead).
    assert queue.recover_expired(factory) == []
    deadline, actions = time.time() + 15, []
    while not actions and time.time() < deadline:                 # lease expires -> sweeper acts
        time.sleep(0.3)
        actions = queue.recover_expired(factory)
    assert [a for _, a in actions] == ["requeued"]

    w = Worker(factory, concurrency=1, lease_seconds=10)
    while w.run_once():
        pass
    with factory() as s:
        run = s.scalars(select(m.Run)).one()
        a = s.scalars(select(m.Attempt).where(m.Attempt.run_id == run.id)
                      .order_by(m.Attempt.attempt_number)).all()
        assert [x.execution_status for x in a] == ["interrupted", "completed"]
        assert a[0].environment_id != a[1].environment_id         # replacement got clean state
        assert run.final_attempt_id == a[1].id and run.evaluation_status == "pass"
        # No duplicate counted result: scores exist only for the replacement attempt.
        assert s.scalar(select(func.count()).select_from(m.Score).where(
            m.Score.attempt_id == a[0].id)) == 0
        assert s.scalar(select(func.count(func.distinct(m.Score.attempt_id)))) == 1
