"""PostgreSQL-backed tests. Skipped unless TEST_DATABASE_URL points at a scratch database, e.g.

    TEST_DATABASE_URL=postgresql+psycopg://postgres@localhost:5433/evaltest pytest tests/pg

The schema is dropped and recreated for every test. NEVER point this at a database you care about.
"""
import os

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import models as m
from scripts.seed import seed

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


def pytest_collection_modifyitems(config, items):
    if URL:
        return
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL not set")
    for item in items:
        if "tests/pg" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture
def engine():
    eng = create_engine(URL, pool_size=20, max_overflow=30)
    m.Base.metadata.drop_all(eng)
    m.Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def factory(engine):
    f = sessionmaker(engine, expire_on_commit=False)
    seed(f)
    return f


@pytest.fixture
def ids(factory):
    with factory() as s:
        dv = s.scalars(select(m.DatasetVersion)).one()
        cfgs = {c.name: c.id for c in s.scalars(select(m.AgentConfig))}
    return {"dv": dv.id, "cfg": cfgs}


def new_experiment(factory, ids, configs=("scripted:reference_solver",), split="test", reps=1,
                   max_retries=2, task_ids=None, key=None):
    from app.execution.planning import ExperimentRequest, create_experiment
    req = ExperimentRequest(dataset_version_id=ids["dv"],
                            agent_config_ids=[ids["cfg"][c] for c in configs], split=split,
                            repetitions=reps, max_retries=max_retries, task_ids=task_ids)
    with factory() as s:
        exp, _ = create_experiment(s, req, key)
        s.commit()
        return exp.id


def run_to_completion(factory, ids, configs, *, split="all", reps=1, task_ids=None,
                      max_retries=2, adapter_factory=None):
    """Create an experiment and drain it with an in-process worker. Returns the experiment id."""
    from app.execution.worker import Worker
    from app.execution.adapters import build_adapter
    exp = new_experiment(factory, ids, configs=configs, split=split, reps=reps,
                         task_ids=task_ids, max_retries=max_retries)
    w = Worker(factory, concurrency=1, lease_seconds=30, heartbeat_seconds=5,
               adapter_factory=adapter_factory or build_adapter)
    while w.run_once():
        pass
    return exp
