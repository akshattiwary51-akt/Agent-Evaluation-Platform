"""Evaluate every scripted behavior against the sample task. No model key or spend needed.

    python -m scripts.smoke            # print a scorecard
    python -m scripts.smoke --db       # also persist traces and scores to PostgreSQL
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.agents.scripted import BEHAVIORS, ScriptedAdapter
from app.execution.runner import run_attempt
from app.schemas.task import load_task

TASK_PATH = Path(__file__).resolve().parent.parent / "datasets/support/refund_eligible_001.yaml"


async def main(persist: bool) -> None:
    task = load_task(TASK_PATH)
    print(f"Task {task.id} v{task.version}\n")
    print(f"{'behavior':28} {'exec':16} {'eval':6} {'viol':5} {'unsafe':6} failing checks")
    print("-" * 100)
    session = None
    if persist:
        from app.db.persist import persist_outcome
        from app.db.session import SessionLocal
        session = SessionLocal()
    for name in BEHAVIORS:
        outcome = await run_attempt(task, ScriptedAdapter(name))
        v = outcome.verdict
        failing = [f"{s.dimension}:{s.reason_code}" for s in outcome.scores
                   if s.status in {"fail", "partial", "evaluation_error"}]
        print(f"{name:28} {outcome.execution_status:16} {outcome.evaluation_status:6} "
              f"{str(v.executed_violation) if v else '-':5} "
              f"{str(v.unsafe_attempt) if v else '-':6} {', '.join(failing) or '-'}")
        if session:
            persist_outcome(session, task, f"scripted:{name}", outcome)
    if session:
        session.commit()
        print("\nPersisted to PostgreSQL.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", action="store_true")
    asyncio.run(main(parser.parse_args().db))
