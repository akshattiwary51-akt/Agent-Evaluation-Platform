"""Import the versioned support dataset and register scripted agent configurations. Idempotent.

    python -m scripts.seed
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.agents.scripted import BEHAVIORS
from app.datasets import build_manifest, load_dataset, validate_dataset
from app.db import models as m
from app.db.session import SessionLocal

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets/support"


def seed(factory) -> dict:
    tasks = load_dataset(DATASET_DIR)
    problems = validate_dataset(tasks)
    if problems:
        raise SystemExit("Refusing to seed an invalid dataset:\n  " + "\n  ".join(problems))
    manifest = build_manifest(tasks)
    with factory() as s:
        ds = s.scalars(select(m.Dataset).filter_by(name=manifest["dataset"])).first()
        if ds is None:
            ds = m.Dataset(name=manifest["dataset"])
            s.add(ds)
            s.flush()
        dv = s.scalars(select(m.DatasetVersion).filter_by(
            dataset_id=ds.id, version=manifest["version"])).first()
        if dv is None:
            dv = m.DatasetVersion(dataset_id=ds.id, version=manifest["version"],
                                  content_hash=manifest["content_hash"], manifest=manifest)
            s.add(dv)
            s.flush()
            for t in tasks:
                s.add(m.TaskVersion(
                    dataset_version_id=dv.id, task_id=t.id, version=t.version,
                    category=t.category, split=t.split,
                    agent_input=t.agent_input.model_dump(),
                    environment_fixture=t.environment_fixture.model_dump(by_alias=True),
                    evaluation_spec=t.evaluation.model_dump(),
                    reviewer_behavior=t.reviewer_behavior.model_dump(),
                    limits=t.evaluation.limits.model_dump()))
        elif dv.content_hash != manifest["content_hash"]:
            raise SystemExit(f"Dataset {manifest['dataset']} {manifest['version']} already exists "
                             "with different content. Dataset versions are immutable: bump the "
                             "version instead of editing it.")
        for name in BEHAVIORS:
            cfg = f"scripted:{name}"
            if not s.scalars(select(m.AgentConfig).filter_by(name=cfg)).first():
                s.add(m.AgentConfig(name=cfg, adapter="scripted", settings={"behavior": name},
                                    config_hash=hashlib.sha256(cfg.encode()).hexdigest()))
        s.commit()
    return manifest


def main() -> None:
    manifest = seed(SessionLocal)
    print(f"Seeded {manifest['dataset']} {manifest['version']}: {manifest['task_count']} tasks, "
          f"{len(BEHAVIORS)} scripted configs.")


if __name__ == "__main__":
    main()
