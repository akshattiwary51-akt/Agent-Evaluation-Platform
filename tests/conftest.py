from pathlib import Path

import pytest

from app.schemas.task import Task, load_task

TASK_PATH = Path(__file__).resolve().parent.parent / "datasets/support/refund_eligible_001.yaml"


@pytest.fixture
def task() -> Task:
    return load_task(TASK_PATH)
