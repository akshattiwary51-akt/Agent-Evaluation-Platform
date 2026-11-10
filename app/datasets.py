"""Load and validate a task dataset directory (YAML tasks + manifest.json)."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from app.schemas.task import Task, load_task

KNOWN_INVARIANTS = {"no_cross_customer_access", "refund_requires_valid_approval",
                    "no_duplicate_refund_effect"}
KNOWN_REQUIREMENTS = {"inspect_order_before_refund", "inspect_return_before_refund"}


def load_dataset(directory: str | Path) -> list[Task]:
    paths = sorted(Path(directory).glob("*.yaml"))
    return sorted((load_task(p) for p in paths), key=lambda t: t.id)


def dataset_hash(tasks: list[Task]) -> str:
    ordered = sorted(tasks, key=lambda t: t.id)  # hash must not depend on load order
    blob = json.dumps([t.model_dump(by_alias=True) for t in ordered], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def validate_dataset(tasks: list[Task]) -> list[str]:
    """Return a list of problems; empty means structurally sound."""
    problems: list[str] = []
    ids = [t.id for t in tasks]
    problems += [f"duplicate id {i}" for i in {i for i in ids if ids.count(i) > 1}]

    family_splits: dict[str, set[str]] = defaultdict(set)
    for t in tasks:
        fx, me = t.environment_fixture, t.agent_input.authenticated_customer_id
        if t.family is None:
            problems.append(f"{t.id}: missing family")
        else:
            family_splits[t.family].add(t.split)
        if fx.order.customer_id != me:
            problems.append(f"{t.id}: main order is not owned by the authenticated customer")
        if any(o.customer_id == me for o in fx.extra_orders):
            problems.append(f"{t.id}: extra_orders must belong to other customers")
        if t.agent_input.ticket_id != fx.ticket.id:
            problems.append(f"{t.id}: ticket id mismatch between agent view and fixture")
        for inv in t.evaluation.invariants:
            if inv not in KNOWN_INVARIANTS:
                problems.append(f"{t.id}: unknown invariant {inv}")
        for req in t.evaluation.required_behavior:
            if req not in KNOWN_REQUIREMENTS:
                problems.append(f"{t.id}: unknown requirement {req}")
        # The agent-visible message must not leak hidden fixture values.
        msg = t.agent_input.user_message
        hidden = [str(fx.order.paid_amount_minor)] + [o.customer_id for o in fx.extra_orders]
        problems += [f"{t.id}: user message leaks hidden value {h!r}" for h in hidden if h in msg]
        e = t.evaluation
        if (e.outcome.refund_total_minor is None and e.outcome.ticket_status is None
                and not e.outcome.ticket_status_not):
            problems.append(f"{t.id}: no outcome check")
    for fam, splits in family_splits.items():
        if len(splits) > 1:
            problems.append(f"family {fam} straddles splits {sorted(splits)}")
    return problems


def build_manifest(tasks: list[Task], name: str = "support", version: str = "v1") -> dict:
    by_split: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, int] = defaultdict(int)
    for t in tasks:
        by_split[t.split].append(t.id)
        by_category[t.category] += 1
    return {"dataset": name, "version": version, "content_hash": dataset_hash(tasks),
            "task_count": len(tasks), "categories": dict(sorted(by_category.items())),
            "splits": {k: sorted(v) for k, v in sorted(by_split.items())}}
