"""Evaluation worker: claims jobs, runs attempts, records results, recovers lost leases.

    python -m app.execution.worker        # env: DATABASE_URL, WORKER_CONCURRENCY, LEASE_SECONDS

Concurrency is bounded by the number of worker threads. SIGTERM/SIGINT stop claiming new jobs and
let in-flight attempts finish (graceful shutdown). If a worker is killed instead, its lease
expires and any worker's sweeper requeues the job as a fresh attempt.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
import uuid
from typing import Callable

from sqlalchemy.orm import sessionmaker

from app.agents.base import AgentAdapter
from app.db import models as m
from app.execution import queue
from app.execution.adapters import build_adapter
from app.execution.runner import run_attempt
from app.schemas.task import Task

log = logging.getLogger("worker")


def load_task(tv: m.TaskVersion) -> Task:
    return Task.model_validate({
        "id": tv.task_id, "version": tv.version, "category": tv.category, "split": tv.split,
        "agent_input": tv.agent_input, "environment_fixture": tv.environment_fixture,
        "reviewer_behavior": tv.reviewer_behavior, "evaluation": tv.evaluation_spec})


class Heartbeat(threading.Thread):
    """Keeps the lease alive and watches for cancellation while an attempt runs."""

    def __init__(self, factory: sessionmaker, job: queue.ClaimedJob, lease_seconds: float,
                 interval: float):
        super().__init__(daemon=True)
        self._factory, self._job, self._lease, self._interval = factory, job, lease_seconds, interval
        self._halt = threading.Event()
        self.lost = threading.Event()
        self.cancelled = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(self._interval):
            try:
                with self._factory() as s, s.begin():
                    ok = queue.heartbeat(s, self._job, self._lease)
                    cancel = queue.run_cancel_requested(s, self._job.run_id) if ok else False
            except Exception:  # transient DB error: try again next tick; expiry is the backstop
                log.exception("heartbeat failed")
                continue
            if not ok:
                self.lost.set()
                return
            if cancel:
                self.cancelled.set()

    def stop(self) -> None:
        self._halt.set()

    def should_stop(self) -> str | None:
        if self.lost.is_set():
            return "lease_lost"
        return "cancelled" if self.cancelled.is_set() else None


class Worker:
    def __init__(self, factory: sessionmaker, *, worker_id: str | None = None,
                 lease_seconds: float = 60.0, heartbeat_seconds: float | None = None,
                 poll_seconds: float = 1.0, sweep_seconds: float = 5.0, concurrency: int = 2,
                 retry_delay_seconds: float = 0.0,
                 adapter_factory: Callable[[m.AgentConfig], AgentAdapter] = build_adapter):
        self.factory = factory
        self.worker_id = worker_id or f"w_{uuid.uuid4().hex[:8]}"
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds or max(0.1, lease_seconds / 3)
        self.poll_seconds, self.sweep_seconds = poll_seconds, sweep_seconds
        self.concurrency = concurrency
        self.retry_delay_seconds = retry_delay_seconds
        self.adapter_factory = adapter_factory

    # ------------------------------------------------------------------ one job
    def process(self, job: queue.ClaimedJob) -> str:
        with self.factory() as s:
            run = s.get(m.Run, job.run_id)
            if queue.run_cancel_requested(s, job.run_id):
                return "cancelled" if queue.cancel_claimed_job(self.factory, job) else "stale"
            task = load_task(s.get(m.TaskVersion, run.task_version_id))
            config = s.get(m.AgentConfig, run.agent_config_id)
            s.expunge(config)

        attempt_id = queue.start_attempt(self.factory, job)
        if attempt_id is None:
            return "stale"
        hb = Heartbeat(self.factory, job, self.lease_seconds, self.heartbeat_seconds)
        hb.start()
        try:
            try:
                adapter = self.adapter_factory(config)
                outcome = asyncio.run(run_attempt(task, adapter, attempt_id=attempt_id,
                                                  should_stop=hb.should_stop))
            except Exception as exc:  # cannot even build the adapter: infrastructure, not agent
                from app.execution.runner import AttemptOutcome
                outcome = AttemptOutcome(
                    attempt_id=attempt_id, environment_id=uuid.uuid4().hex,
                    execution_status="infrastructure_error", stop_reason="adapter_setup",
                    final_response="", final_state={}, events=[], tool_calls=[],
                    error=f"{type(exc).__name__}: {exc}")
        finally:
            hb.stop()
        if outcome.execution_status == "lease_lost":
            log.warning("job %s: lease lost; discarding result", job.id)
            return "stale"
        return queue.finalize(self.factory, job, attempt_id, outcome, self.retry_delay_seconds)

    def run_once(self) -> bool:
        """Claim and process at most one job. Returns False if nothing was available."""
        job = queue.claim_job(self.factory, self.worker_id, self.lease_seconds)
        if job is None:
            return False
        try:
            self.process(job)
        except Exception:  # never kill the loop; the lease will expire and recovery takes over
            log.exception("job %s crashed in worker", job.id)
        return True

    # ------------------------------------------------------------------ loops
    def _loop(self, stop: threading.Event) -> None:
        last_sweep = 0.0
        while not stop.is_set():
            if time.monotonic() - last_sweep >= self.sweep_seconds:
                try:
                    for job_id, action in queue.recover_expired(self.factory):
                        log.warning("recovered expired lease: job %s -> %s", job_id, action)
                except Exception:
                    log.exception("sweep failed")
                last_sweep = time.monotonic()
            if not self.run_once():
                stop.wait(self.poll_seconds)

    def run_forever(self, stop: threading.Event) -> None:
        threads = [threading.Thread(target=self._loop, args=(stop,), name=f"{self.worker_id}-{i}")
                   for i in range(self.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from app.db.session import SessionLocal
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    worker = Worker(SessionLocal,
                    lease_seconds=float(os.environ.get("LEASE_SECONDS", "60")),
                    concurrency=int(os.environ.get("WORKER_CONCURRENCY", "2")),
                    retry_delay_seconds=float(os.environ.get("RETRY_DELAY_SECONDS", "5")))
    log.info("worker %s started (concurrency=%d)", worker.worker_id, worker.concurrency)
    worker.run_forever(stop)
    log.info("worker stopped cleanly")


if __name__ == "__main__":
    main()
