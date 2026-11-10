"""Guide Demo 2: compare two agent configurations on the same tasks and explain the difference.

Runs the careless and careful scripted stand-ins ("Version A" / "Version B") through a real
experiment on the validation split using an in-process worker, then prints the paired comparison.
Needs a migrated, seeded database (DATABASE_URL):

    make migrate && make seed && python -m scripts.demo_comparison

The stand-ins are scripted, not models: the numbers show the platform's mechanics, not model quality.
"""
from __future__ import annotations

import sys

from sqlalchemy import select

from app.analysis.comparison import Side, compare
from app.db import models as m
from app.db.session import SessionLocal
from app.execution.adapters import build_adapter
from app.execution.planning import ExperimentRequest, create_experiment
from app.execution.worker import Worker

A, B = "scripted:careless_agent", "scripted:careful_verbose_agent"


def main(split: str = "validation") -> int:
    with SessionLocal() as s:
        dv = s.scalars(select(m.DatasetVersion)).first()
        cfg = {c.name: c.id for c in s.scalars(select(m.AgentConfig))}
        if dv is None or A not in cfg or B not in cfg:
            print("Run `make migrate && make seed` first.")
            return 1
        exp, _ = create_experiment(s, ExperimentRequest(
            dataset_version_id=dv.id, agent_config_ids=[cfg[A], cfg[B]], split=split,
            repetitions=1, max_retries=1), None)
        s.commit()
        exp_id = exp.id
    worker = Worker(SessionLocal, concurrency=1, lease_seconds=30, heartbeat_seconds=5,
                    adapter_factory=build_adapter)
    while worker.run_once():
        pass
    with SessionLocal() as s:
        r = compare(s, Side(exp_id, cfg[A]), Side(exp_id, cfg[B]))
    print(f"experiment {exp_id}  split={split}\n")
    for key, v in r["metrics"].items():
        print(f"{key:34} {v}")
    print(f"\nsame-task outcomes: {r['task_outcomes']}")
    print(f"regressed: {[t['task_id'] for t in r['regressed_cases']]}")
    print(f"improved:  {len(r['improved_cases'])} tasks   (dashboard: /ui/compare?experiment_id={exp_id}"
          f"&baseline={A}&candidate={B})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
