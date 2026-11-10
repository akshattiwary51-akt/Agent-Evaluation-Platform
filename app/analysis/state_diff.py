"""Human-readable differences between a task's initial fixture and an attempt's final state."""
from __future__ import annotations

from typing import Any


def state_diff(fixture: dict[str, Any], final: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    orders = {o["id"]: o for o in [fixture["order"], *fixture.get("extra_orders", [])]}
    for oid, before in orders.items():
        issued = sum(r["amount_minor"] for r in final.get("refunds", []) if r["order_id"] == oid)
        if issued:
            changes.append({"entity": f"order {oid}", "field": "refunded_amount_minor",
                            "before": before["refunded_amount_minor"],
                            "after": before["refunded_amount_minor"] + issued})
    for r in final.get("refunds", []):
        changes.append({"entity": "refund", "field": r["refund_id"], "before": None,
                        "after": f"{r['amount_minor']} on {r['order_id']} (approval {r['approval_id']})"})
    for a in final.get("approvals", {}).values():
        changes.append({"entity": "approval", "field": a["id"], "before": None,
                        "after": f"{a['status']} for {a['amount_minor']} on {a['order_id']}"
                                 f"{' (used)' if a['used'] else ''}"})
    tid = fixture["ticket"]["id"]
    ticket = final.get("tickets", {}).get(tid, {})
    if ticket.get("status") != fixture["ticket"]["status"]:
        changes.append({"entity": f"ticket {tid}", "field": "status",
                        "before": fixture["ticket"]["status"], "after": ticket.get("status")})
    if final.get("uncertain_operations"):
        changes.append({"entity": "environment", "field": "uncertain_operations", "before": [],
                        "after": final["uncertain_operations"]})
    return changes
