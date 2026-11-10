"""Scripted STAND-INS for two prompt versions, so comparisons can be demonstrated offline.

These are deterministic scripts, NOT language models: they imitate the *shape* of failure the
guide describes (a basic prompt that trusts tool calls and claims success; a revised prompt that
investigates and verifies, at the cost of extra tool calls). Real prompt comparisons need
real model configurations (`tool_loop`), run through the same experiment machinery.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.reference import REFUND_WORDS, reference_solver

if TYPE_CHECKING:
    from app.agents.scripted import ScriptContext

GIVE_UP = "Sorry, something went wrong on our side."


async def careless_agent(ctx: "ScriptContext") -> str:
    """'Version A': single attempt per call, never checks eligibility or results, and tells the
    customer the refund was issued whatever the tools said."""
    msg = ctx.task_input["user_message"].lower()
    oid = ctx.find_order_id()
    if oid is None or not any(w in msg for w in REFUND_WORDS):
        return await reference_solver(ctx)           # lookups and clarifications are fine
    order = await ctx.call("get_order", order_id=oid)
    if not order.ok:
        return GIVE_UP
    amount = order.content["remaining_refundable_minor"]
    appr = await ctx.call("request_refund_approval", order_id=oid, amount_minor=amount,
                          reason="Customer request")
    if not appr.ok:
        return GIVE_UP
    await ctx.call("issue_refund", order_id=oid, amount_minor=amount,
                   approval_id=appr.content["approval_id"])        # result never inspected
    await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="resolved", note="Refund issued.")
    return f"Your refund of {amount / 100:.2f} {order.content['currency']} has been issued."


async def careful_verbose_agent(ctx: "ScriptContext") -> str:
    """'Version B': follows policy and verifies, with redundant lookups before and after."""
    oid = ctx.find_order_id()
    if oid:
        await ctx.call("get_order", order_id=oid)                  # pre-check (redundant later)
    reply = await reference_solver(ctx)
    if oid:
        await ctx.call("get_refund_status", order_id=oid)          # post-verification
    return reply
