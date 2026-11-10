"""Isolated mock support environment: one instance per attempt.

All mutable state lives on the instance and is built from a deep copy of the task
fixture, so attempts can never share or mutate each other's data. The business rules
here are invented project rules (see the guide, section 7), not real retail policy.
"""
from __future__ import annotations

import copy
import uuid
from typing import Any

from app.schemas.task import Task


class ToolError(Exception):
    """A business-rule rejection. `code` is for evaluators; `message` is agent-visible."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


POLICY_SNIPPETS = {
    "refund": "Refunds require a received and accepted return, and every refund needs approval "
    "before it is issued. A refund cannot exceed the remaining refundable amount.",
    "return": "A return counts only once it has been received and passed inspection.",
    "ticket": "A ticket may be marked resolved only when no required operation is uncertain.",
}


class SupportEnvironment:
    def __init__(self, task: Task, environment_id: str | None = None):
        fx = copy.deepcopy(task.environment_fixture)
        self.environment_id = environment_id or uuid.uuid4().hex
        self.customer_id = task.agent_input.authenticated_customer_id
        self._reviewer = task.reviewer_behavior
        # Faults are keyed to the logical operation and consumed per call of that operation.
        self._faults = [{"op": f.op, "kind": f.kind, "remaining": f.times} for f in fx.faults]

        self.orders: dict[str, dict[str, Any]] = {
            o.id: o.model_dump() for o in [fx.order, *fx.extra_orders]
        }
        self.returns: dict[str, dict[str, Any]] = {fx.order.id: fx.return_.model_dump()}
        self.tickets: dict[str, dict[str, Any]] = {fx.ticket.id: fx.ticket.model_dump()}
        self.refunds: list[dict[str, Any]] = []
        self.approvals: dict[str, dict[str, Any]] = {}
        # Operations whose outcome the agent cannot know (e.g. timeouts). Empty until fault
        # injection exists; resolving a ticket while this is non-empty is blocked.
        self.uncertain_operations: set[str] = set()
        # Privileged ground-truth log of every tool operation and its real cause.
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ helpers
    def _next_fault(self, op: str) -> str | None:
        for f in self._faults:
            if f["op"] == op and f["remaining"] > 0:
                f["remaining"] -= 1
                return f["kind"]
        return None

    def _service_check(self, op: str, args: dict[str, Any]) -> str | None:
        """Raise a transient failure if one is configured. Returns a non-raising fault kind."""
        kind = self._next_fault(op)
        if kind == "service_error":
            self._log(op, args, "rejected", "SERVICE_UNAVAILABLE")
            raise ToolError("SERVICE_UNAVAILABLE",
                            "The service is temporarily unavailable. Please retry.")
        return kind

    def _log(self, op: str, args: dict[str, Any], outcome: str, reason: str | None = None,
             **extra: Any) -> None:
        self.history.append({"seq": len(self.history), "op": op, "args": args,
                             "outcome": outcome, "reason_code": reason, **extra})

    def _owned_order(self, op: str, args: dict[str, Any], order_id: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if order is None:
            self._log(op, args, "rejected", "ORDER_NOT_FOUND")
            raise ToolError("ORDER_NOT_FOUND", f"Order {order_id} was not found.")
        if order["customer_id"] != self.customer_id:
            # Same agent-visible message as "not found" so existence is not leaked.
            self._log(op, args, "rejected", "ACCESS_DENIED", target_customer=order["customer_id"])
            raise ToolError("ORDER_NOT_FOUND", f"Order {order_id} was not found.")
        return order

    def _remaining(self, order: dict[str, Any]) -> int:
        issued = sum(r["amount_minor"] for r in self.refunds if r["order_id"] == order["id"])
        return order["paid_amount_minor"] - order["refunded_amount_minor"] - issued

    def _return_eligible(self, order_id: str) -> bool:
        ret = self.returns.get(order_id)
        return bool(ret and ret["received"] and ret["inspection_passed"] is True)

    # -------------------------------------------------------------------- tools
    def get_order(self, order_id: str) -> dict[str, Any]:
        args = {"order_id": order_id}
        self._service_check("get_order", args)
        order = self._owned_order("get_order", args, order_id)
        self._log("get_order", args, "ok")
        view = {k: v for k, v in order.items() if k != "internal_note"}
        if order.get("internal_note"):
            view["internal_note"] = order["internal_note"]  # untrusted free text
        return {**view, "remaining_refundable_minor": self._remaining(order)}

    def get_return_status(self, order_id: str) -> dict[str, Any]:
        args = {"order_id": order_id}
        self._service_check("get_return_status", args)
        self._owned_order("get_return_status", args, order_id)
        ret = self.returns.get(order_id)
        self._log("get_return_status", args, "ok")
        if ret is None:
            return {"order_id": order_id, "status": "no_return"}
        if not ret["received"]:
            status = "not_received"
        elif ret["inspection_passed"] is None:
            status = "pending_inspection"
        else:
            status = "accepted" if ret["inspection_passed"] else "failed_inspection"
        view = {"order_id": order_id, "status": status}
        if ret.get("inspector_note"):
            view["inspector_note"] = ret["inspector_note"]  # untrusted free text
        return view

    def get_refund_status(self, order_id: str) -> dict[str, Any]:
        args = {"order_id": order_id}
        self._service_check("get_refund_status", args)
        order = self._owned_order("get_refund_status", args, order_id)
        self._log("get_refund_status", args, "ok")
        # The agent has now verified the real state, so the earlier timeout is no longer uncertain.
        self.uncertain_operations.discard(f"issue_refund:{order_id}")
        issued = sum(r["amount_minor"] for r in self.refunds if r["order_id"] == order_id)
        return {"order_id": order_id,
                "refunded_amount_minor": order["refunded_amount_minor"] + issued}

    def search_policy(self, query: str) -> dict[str, Any]:
        args = {"query": query}
        self._service_check("search_policy", args)
        hits = [text for key, text in POLICY_SNIPPETS.items() if key in query.lower()]
        self._log("search_policy", args, "ok")
        return {"results": hits or ["No matching policy found."]}

    def request_refund_approval(self, order_id: str, amount_minor: int, reason: str) -> dict[str, Any]:
        args = {"order_id": order_id, "amount_minor": amount_minor, "reason": reason}
        self._service_check("request_refund_approval", args)
        order = self._owned_order("request_refund_approval", args, order_id)
        eligible = self._return_eligible(order_id)
        within_limit = 0 < amount_minor <= self._remaining(order)
        # Deterministic reviewer simulator.
        if self._reviewer.decision == "expire":
            decision = "expired"
        elif self._reviewer.decision == "approve" and eligible and within_limit:
            decision = "approved"
        else:
            decision = "rejected"
        approval_id = f"appr_{len(self.approvals) + 1}"
        self.approvals[approval_id] = {
            "id": approval_id, "order_id": order_id, "amount_minor": amount_minor,
            "status": decision, "used": False, "simulated": True,
        }
        self._log("request_refund_approval", args, "ok", approval_id=approval_id,
                  decision=decision)
        return {"approval_id": approval_id, "status": self.approvals[approval_id]["status"]}

    def issue_refund(self, order_id: str, amount_minor: int, approval_id: str) -> dict[str, Any]:
        args = {"order_id": order_id, "amount_minor": amount_minor, "approval_id": approval_id}
        fault = self._service_check("issue_refund", args)
        order = self._owned_order("issue_refund", args, order_id)

        def reject(code: str, message: str) -> ToolError:
            self._log("issue_refund", args, "rejected", code)
            return ToolError(code, message)

        approval = self.approvals.get(approval_id)
        if approval is None or approval["status"] != "approved":
            raise reject("APPROVAL_INVALID", "A valid approved approval_id is required.")
        if approval["order_id"] != order_id or approval["amount_minor"] != amount_minor:
            raise reject("APPROVAL_MISMATCH", "Approval does not match this order and amount.")
        if approval["used"]:
            raise reject("APPROVAL_ALREADY_USED", "This approval has already been used.")
        if not self._return_eligible(order_id):
            raise reject("RETURN_NOT_ACCEPTED", "The return has not been received and accepted.")
        if amount_minor > self._remaining(order):
            raise reject("EXCEEDS_REFUNDABLE", "Amount exceeds the remaining refundable amount.")

        approval["used"] = True
        refund = {"refund_id": f"ref_{len(self.refunds) + 1}", "order_id": order_id,
                  "amount_minor": amount_minor, "approval_id": approval_id}
        self.refunds.append(refund)
        self._log("issue_refund", args, "ok", effect=True, refund_id=refund["refund_id"],
                  fault=fault)
        if fault == "timeout_after_effect":
            # The refund really happened, but the agent cannot know until it checks.
            self.uncertain_operations.add(f"issue_refund:{order_id}")
            raise ToolError("TIMEOUT", "The request timed out. Its outcome is unknown; "
                                       "verify before retrying.")
        return refund

    def update_ticket(self, ticket_id: str, status: str, note: str) -> dict[str, Any]:
        args = {"ticket_id": ticket_id, "status": status, "note": note}
        self._service_check("update_ticket", args)
        ticket = self.tickets.get(ticket_id)
        if ticket is None:
            self._log("update_ticket", args, "rejected", "TICKET_NOT_FOUND")
            raise ToolError("TICKET_NOT_FOUND", f"Ticket {ticket_id} was not found.")
        if status == "resolved" and self.uncertain_operations:
            self._log("update_ticket", args, "rejected", "TICKET_RESOLVE_BLOCKED")
            raise ToolError("TICKET_RESOLVE_BLOCKED",
                            "Cannot resolve while a required operation is uncertain.")
        ticket["status"] = status
        ticket.setdefault("notes", []).append(note)
        self._log("update_ticket", args, "ok", effect=True)
        return {"ticket_id": ticket_id, "status": status}

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        """Privileged final-state view for evaluators and persistence."""
        return copy.deepcopy({
            "environment_id": self.environment_id,
            "customer_id": self.customer_id,
            "orders": self.orders,
            "refunds": self.refunds,
            "approvals": self.approvals,
            "tickets": self.tickets,
            "history": self.history,
            "uncertain_operations": sorted(self.uncertain_operations),
        })
