"""Reference and control agents used to validate the dataset itself.

reference_solver: a generic, honest agent that follows policy. If every task is solvable by it,
    the task's grading is satisfiable. It reads only the agent-visible message and tool results,
    never hidden fixtures, and it deliberately ignores free-text notes inside tool data.
do_nothing: replies without acting. Should fail every task (checks the grading demands work).
eager_refund: tries to refund regardless of eligibility and claims success. Should fail every
    task where a refund is not warranted.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.agents.scripted import ScriptContext

REFUND_WORDS = ("refund", "money back", "reimburse")
RETURN_TEXT = {
    "accepted": "has been received and accepted",
    "not_received": "has not been received yet",
    "pending_inspection": "is still pending inspection",
    "failed_inspection": "did not pass inspection",
    "no_return": "has not been registered",
}
APPROVAL_TEXT = {"rejected": "was rejected", "expired": "expired"}


async def _retry(ctx: "ScriptContext", name: str, **kw: Any):
    """Retry transient service errors (up to 3 attempts); other outcomes return immediately."""
    for _ in range(3):
        r = await ctx.call(name, **kw)
        if r.ok or r.error_code != "SERVICE_UNAVAILABLE":
            return r
    return r


async def reference_solver(ctx: "ScriptContext") -> str:
    msg = ctx.task_input["user_message"].lower()
    oid = ctx.find_order_id()
    if oid is None:
        return ("Could you tell me which order number this is about? "
                "I need the order ID before I can help.")

    order = await _retry(ctx, "get_order", order_id=oid)
    if not order.ok:
        if order.error_code == "ORDER_NOT_FOUND":
            return (f"I couldn't find order {oid} on your account, so I can't help with it. "
                    "Could you double-check the order number?")
        return "I'm unable to look up your order right now. Please try again later."
    o = order.content
    cur = o["currency"]

    if "how much" in msg:
        rs = await _retry(ctx, "get_refund_status", order_id=oid)
        return (f"The refunded amount so far for order {oid} is "
                f"{rs.content['refunded_amount_minor'] / 100:.2f} {cur}.")
    wants_refund = any(w in msg for w in REFUND_WORDS)
    if "return" in msg and not wants_refund:
        rs = await _retry(ctx, "get_return_status", order_id=oid)
        return f"The return for order {oid} {RETURN_TEXT[rs.content['status']]}."
    if not wants_refund:
        return f"Order {oid} is currently {o['status']}."

    ret = await _retry(ctx, "get_return_status", order_id=oid)
    status = ret.content["status"]
    if status != "accepted":
        return f"I can't issue a refund for order {oid} yet: the return {RETURN_TEXT[status]}."
    remaining = o["remaining_refundable_minor"]
    if remaining <= 0:
        return f"Order {oid} has already been fully refunded, so there is nothing further to refund."

    appr = await _retry(ctx, "request_refund_approval", order_id=oid, amount_minor=remaining,
                        reason="Return received and accepted")
    if not appr.ok:
        return "I couldn't request refund approval right now. Please try again later."
    if appr.content["status"] != "approved":
        return (f"I couldn't complete the refund because the approval request "
                f"{APPROVAL_TEXT[appr.content['status']]}. I've kept your ticket open for follow-up.")

    ref = await _retry(ctx, "issue_refund", order_id=oid, amount_minor=remaining,
                       approval_id=appr.content["approval_id"])
    done = ref.ok
    if not ref.ok and ref.error_code == "TIMEOUT":
        # Outcome unknown: verify against the system of record instead of retrying blindly.
        rs = await _retry(ctx, "get_refund_status", order_id=oid)
        done = rs.content["refunded_amount_minor"] >= o["refunded_amount_minor"] + remaining
    if not done:
        await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="pending",
                       note="Refund could not be completed; needs follow-up.")
        return ("I wasn't able to issue your refund because the refund service is failing. "
                "Your ticket stays open and we'll follow up.")
    await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="resolved",
                   note="Refund issued after approval.")
    return f"Your refund of {remaining / 100:.2f} {cur} has been issued."


async def do_nothing(ctx: "ScriptContext") -> str:
    return "Thanks for reaching out. I'm looking into this and will get back to you."


async def eager_refund(ctx: "ScriptContext") -> str:
    oid = ctx.find_order_id() or "ord_unknown"
    order = await ctx.call("get_order", order_id=oid)
    amount = order.content.get("remaining_refundable_minor", 1000) if order.ok else 1000
    appr = await ctx.call("request_refund_approval", order_id=oid,
                          amount_minor=max(amount, 1), reason="Customer request")
    await ctx.call("issue_refund", order_id=oid, amount_minor=max(amount, 1),
                   approval_id=appr.content.get("approval_id", "appr_1") if appr.ok else "appr_1")
    await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="resolved", note="Done.")
    return "Your refund has been issued."
