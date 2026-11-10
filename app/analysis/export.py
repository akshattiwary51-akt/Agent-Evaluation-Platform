"""Reproducible experiment report (JSON) and per-run CSV (guide section 9 of the user story)."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import platform
from importlib import metadata
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.prompts import PROMPTS, prompt_hash
from app.analysis import costing, scorecard
from app.db import models as m

ENVIRONMENT_VERSION = "support-env-v1"

CSV_COLUMNS = ["experiment_id", "task_id", "category", "split", "config", "repetition", "run_id",
                "execution_status", "evaluation_status", "attempts", "failing_checks",
                "unsafe_attempt", "executed_violation", "tool_calls", "input_tokens",
                "output_tokens", "cost", "queue_seconds", "active_seconds", "final_attempt_id"]


def dependency_fingerprint() -> dict[str, Any]:
    """Hash of every installed distribution name==version: a fingerprint of the environment that
    produced the results (the repo has no lockfile yet, so this is the honest substitute)."""
    pins = sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in metadata.distributions()
                  if d.metadata["Name"])
    return {"installed_distributions_sha256": hashlib.sha256("\n".join(pins).encode()).hexdigest(),
            "count": len(pins), "python": platform.python_version()}


def build_report(s: Session, exp: m.Experiment) -> dict[str, Any]:
    prices = costing.load_prices(s)
    configs = {c["id"]: c for c in exp.manifest["configs"]}
    cfg_rows = {c.id: c for c in s.scalars(select(m.AgentConfig).where(m.AgentConfig.id.in_(configs)))}
    scorecards, runs, versions = {}, [], set()
    for cid, c in configs.items():
        facts = scorecard.load_run_facts(s, exp.id, cid)
        scorecards[c["name"]] = scorecard.build_scorecard(facts, prices, seed=f"{exp.id}:{cid}")
        for f in facts:
            cost = costing.cost_calls(f.model_calls, prices)
            runs.append({
                "experiment_id": exp.id, "task_id": f.task_id, "category": f.category,
                "split": f.split, "config": c["name"], "repetition": f.repetition,
                "run_id": f.run_id, "execution_status": f.execution_status,
                "evaluation_status": f.evaluation_status, "attempts": f.attempts,
                "failing_checks": ";".join(sorted(set(f.failing))),
                "unsafe_attempt": f.unsafe_attempt, "executed_violation": f.executed_violation,
                "tool_calls": f.calls_total,
                "input_tokens": sum(x.input_tokens or 0 for x in f.model_calls),
                "output_tokens": sum(x.output_tokens or 0 for x in f.model_calls),
                "cost": cost.total if cost.status != "unknown" else "unknown",
                "queue_seconds": f.queue_seconds, "active_seconds": f.active_seconds,
                "final_attempt_id": f.final_attempt_id})
    for ev in s.execute(select(m.Score.evaluator_name, m.Score.evaluator_version).distinct()
                        .join(m.Run, m.Run.final_attempt_id == m.Score.attempt_id)
                        .where(m.Run.experiment_id == exp.id)):
        versions.add(f"{ev[0]}@{ev[1]}")
    limits = {tuple(sorted(tv.limits.items())) for tv in s.scalars(
        select(m.TaskVersion).join(m.Run, m.Run.task_version_id == m.TaskVersion.id)
        .where(m.Run.experiment_id == exp.id))}
    return {
        "report_version": 1, "experiment_id": exp.id, "created_at": exp.created_at.isoformat(),
        "manifest": exp.manifest,
        "reproducibility": {
            "dataset_version_id": exp.manifest["dataset_version_id"],
            "dataset_content_hash": exp.manifest["dataset_content_hash"],
            "environment_version": ENVIRONMENT_VERSION,
            "evaluator_versions": sorted(versions),
            "task_limits_distinct": [dict(x) for x in sorted(limits)],
            "configs": [{
                "id": cid, "name": c["name"], "config_hash": c["config_hash"],
                "adapter": cfg_rows[cid].adapter, "model": cfg_rows[cid].model,
                "settings": cfg_rows[cid].settings,
                "prompt_hash": (prompt_hash(cfg_rows[cid].settings["prompt"])
                                if (cfg_rows[cid].settings or {}).get("prompt") in PROMPTS else None)}
                for cid, c in configs.items()],
            "agent_code_commit": os.environ.get("GIT_COMMIT", "unknown"),
            "dependencies": dependency_fingerprint()},
        "scorecards": scorecards, "runs": runs}


def runs_csv(report: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(report["runs"])
    return buf.getvalue()
