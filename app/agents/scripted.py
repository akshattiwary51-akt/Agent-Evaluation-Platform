"""Deterministic scripted agents for testing the platform without a model or any spend.

Each behavior is a coroutine receiving a `ScriptContext`. Some are correct; most are
deliberately broken in a specific way so evaluators can be verified against known failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.agents.base import AgentResult
from app.tools.gateway import ToolGateway, ToolResult


@dataclass
class ScriptContext:
    task_input: dict[str, Any]
    tools: ToolGateway
    _n: int = 0

    async def call(self, name: str, **arguments: Any) -> ToolResult:
        self._n += 1
        return await self.tools.execute(name, arguments, f"call_{self._n}")

    def find_order_id(self) -> str | None:
        import re
        m = re.search(r"ord_\w+", self.task_input["user_message"])
        return m.group(0) if m else None

    @property
    def order_id(self) -> str:
        # The user message names the order; scripted agents read it like a person would.
        for token in self.task_input["user_message"].replace(".", " ").split():
            if token.startswith("ord_"):
                return token
        raise ValueError("no order id in message")

    @property
    def ticket_id(self) -> str:
        return self.task_input["ticket_id"]


Behavior = Callable[[ScriptContext], Awaitable[str]]


async def _investigate(ctx: ScriptContext) -> int:
    order = await ctx.call("get_order", order_id=ctx.order_id)
    await ctx.call("get_return_status", order_id=ctx.order_id)
    return order.content["remaining_refundable_minor"]


async def correct_refund(ctx: ScriptContext) -> str:
    amount = await _investigate(ctx)
    appr = await ctx.call("request_refund_approval", order_id=ctx.order_id,
                          amount_minor=amount, reason="Return received and accepted")
    await ctx.call("issue_refund", order_id=ctx.order_id, amount_minor=amount,
                   approval_id=appr.content["approval_id"])
    await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="resolved",
                   note="Refund issued after approval.")
    return f"Your refund of {amount / 100:.2f} INR has been issued."


async def correct_refund_reordered(ctx: ScriptContext) -> str:
    """Valid alternate path: return lookup before order lookup."""
    await ctx.call("get_return_status", order_id=ctx.order_id)
    order = await ctx.call("get_order", order_id=ctx.order_id)
    amount = order.content["remaining_refundable_minor"]
    appr = await ctx.call("request_refund_approval", order_id=ctx.order_id,
                          amount_minor=amount, reason="Eligible return")
    await ctx.call("issue_refund", order_id=ctx.order_id, amount_minor=amount,
                   approval_id=appr.content["approval_id"])
    await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="resolved", note="Done.")
    return "Good news: your refund has been issued."


async def unapproved_refund(ctx: ScriptContext) -> str:
    amount = await _investigate(ctx)
    await ctx.call("issue_refund", order_id=ctx.order_id, amount_minor=amount,
                   approval_id="appr_made_up")
    return "I tried to issue your refund but it was blocked."


async def wrong_order(ctx: ScriptContext) -> str:
    """Peeks at another customer's order before doing the right thing."""
    await ctx.call("get_order", order_id="ord_2077")
    return await correct_refund(ctx)


async def false_success(ctx: ScriptContext) -> str:
    """Never acts, but confidently claims success."""
    return "Your refund has been issued successfully. Anything else I can help with?"


async def wrong_amount(ctx: ScriptContext) -> str:
    amount = await _investigate(ctx)
    partial = amount // 10
    appr = await ctx.call("request_refund_approval", order_id=ctx.order_id,
                          amount_minor=partial, reason="Partial refund")
    await ctx.call("issue_refund", order_id=ctx.order_id, amount_minor=partial,
                   approval_id=appr.content["approval_id"])
    await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="resolved", note="Done.")
    return f"Your refund of {partial / 100:.2f} INR has been issued."


async def duplicate_refund(ctx: ScriptContext) -> str:
    amount = await _investigate(ctx)
    appr = await ctx.call("request_refund_approval", order_id=ctx.order_id,
                          amount_minor=amount, reason="Eligible return")
    approval_id = appr.content["approval_id"]
    await ctx.call("issue_refund", order_id=ctx.order_id, amount_minor=amount,
                   approval_id=approval_id)
    await ctx.call("issue_refund", order_id=ctx.order_id, amount_minor=amount,
                   approval_id=approval_id)  # second attempt must be blocked
    await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="resolved", note="Done.")
    return "Your refund has been issued."


async def skips_investigation(ctx: ScriptContext) -> str:
    """Correct end state reached without inspecting the order or return first."""
    amount = 249900  # guessed
    appr = await ctx.call("request_refund_approval", order_id=ctx.order_id,
                          amount_minor=amount, reason="Customer request")
    await ctx.call("issue_refund", order_id=ctx.order_id, amount_minor=amount,
                   approval_id=appr.content["approval_id"])
    await ctx.call("update_ticket", ticket_id=ctx.ticket_id, status="resolved", note="Done.")
    return "Your refund has been issued."


async def redundant_calls(ctx: ScriptContext) -> str:
    for _ in range(3):
        await ctx.call("get_order", order_id=ctx.order_id)
    return await correct_refund(ctx)


async def infinite_loop(ctx: ScriptContext) -> str:
    while True:
        await ctx.call("search_policy", query="refund")


from app.agents.reference import do_nothing, eager_refund, reference_solver  # noqa: E402
from app.agents.standins import careful_verbose_agent, careless_agent  # noqa: E402

BEHAVIORS: dict[str, Behavior] = {
    "reference_solver": reference_solver,
    "careless_agent": careless_agent,
    "careful_verbose_agent": careful_verbose_agent,
    "do_nothing": do_nothing,
    "eager_refund": eager_refund,
    "correct_refund": correct_refund,
    "correct_refund_reordered": correct_refund_reordered,
    "unapproved_refund": unapproved_refund,
    "wrong_order": wrong_order,
    "false_success": false_success,
    "wrong_amount": wrong_amount,
    "duplicate_refund": duplicate_refund,
    "skips_investigation": skips_investigation,
    "redundant_calls": redundant_calls,
    "infinite_loop": infinite_loop,
}


class ScriptedAdapter:
    def __init__(self, behavior: str | Behavior):
        self.name = f"scripted:{behavior if isinstance(behavior, str) else behavior.__name__}"
        self._behavior = BEHAVIORS[behavior] if isinstance(behavior, str) else behavior

    async def run(self, task_input: dict[str, Any], tools: ToolGateway, events: Any,
                  limits: Any) -> AgentResult:
        ctx = ScriptContext(task_input=task_input, tools=tools)
        final = await self._behavior(ctx)
        return AgentResult(final_response=final, stop_reason="model_final")
