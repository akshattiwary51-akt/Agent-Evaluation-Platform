"""Check that the dataset's grading is satisfiable AND discriminating.

    python -m scripts.validate_dataset

- reference_solver must pass every task (the grading can be satisfied honestly).
- do_nothing and false_success must fail every task (the grading demands real work).
- eager_refund must fail every task where a refund is not warranted.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.agents.scripted import ScriptedAdapter
from app.datasets import load_dataset, validate_dataset
from app.execution.runner import run_attempt

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets/support"
CONTROLS = ["reference_solver", "do_nothing", "false_success", "eager_refund"]


async def validation_matrix(tasks) -> dict[str, dict[str, object]]:
    """task_id -> behavior -> AttemptOutcome."""
    return {t.id: {b: await run_attempt(t, ScriptedAdapter(b)) for b in CONTROLS} for t in tasks}


def _issues(outcome) -> str:
    return ", ".join(f"{s.dimension}:{s.reason_code}" for s in outcome.scores
                     if s.status in {"fail", "evaluation_error"}) or "-"


async def main() -> int:
    tasks = load_dataset(DATASET_DIR)
    problems = validate_dataset(tasks)
    matrix = await validation_matrix(tasks)
    bad = list(problems)
    print(f"{'task':24} {'split':11} " + " ".join(f"{b[:9]:>9}" for b in CONTROLS))
    for t in tasks:
        row = matrix[t.id]
        cells = " ".join(f"{row[b].evaluation_status:>9}" for b in CONTROLS)
        print(f"{t.id:24} {t.split:11} {cells}")
        ref = row["reference_solver"]
        if ref.evaluation_status != "pass":
            bad.append(f"{t.id}: reference solver did not pass ({ref.execution_status}; "
                       f"{_issues(ref)}; reply={ref.final_response!r})")
        for b in ("do_nothing", "false_success"):
            if row[b].evaluation_status == "pass":
                bad.append(f"{t.id}: control {b} passes (grading is not discriminating)")
        if t.evaluation.outcome.refund_total_minor == 0 and row["eager_refund"].evaluation_status == "pass":
            bad.append(f"{t.id}: eager_refund passes a task where no refund is warranted")
    print()
    if bad:
        print("PROBLEMS:\n  " + "\n  ".join(bad))
        return 1
    print(f"OK: {len(tasks)} tasks; reference passes all, controls fail as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
