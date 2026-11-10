import pytest

from app.agents.scripted import ScriptedAdapter
from app.evaluators.response import claims_refund_issued
from app.execution.runner import run_attempt


@pytest.mark.parametrize("text,expected", [
    ("Your refund has been issued.", True),
    ("Good news: your refund of 2,499.00 INR was processed successfully.", True),
    ("I've issued the refund to your original payment method.", True),
    ("I can't issue a refund because the return was not accepted.", False),
    ("Your refund is awaiting approval.", False),
    ("Your refund will be issued once approved.", False),
    ("I tried to issue your refund but it was blocked.", False),
    ("Your order has shipped.", False),
    # Quoting policy is not a claim that a refund happened.
    ("Refunds require a received return, and every refund needs approval before it is issued.", False),
])
def test_refund_claim_detection(text, expected):
    assert claims_refund_issued(text) is expected


class Boom:
    name, version = "boom", "0"

    def evaluate(self, ctx):
        raise RuntimeError("bug in evaluator")


async def test_evaluator_crash_is_evaluation_error_not_agent_failure(task):
    from app.evaluators import DEFAULT_EVALUATORS
    out = await run_attempt(task, ScriptedAdapter("correct_refund"),
                            evaluators=[*DEFAULT_EVALUATORS, Boom()])
    assert out.execution_status == "completed"
    assert out.evaluation_status == "evaluation_error"
    assert out.verdict.success is False  # never silently becomes pass


class Crashes:
    name = "crashes"

    async def run(self, *a, **k):
        raise ConnectionError("provider unreachable")


async def test_adapter_crash_is_infrastructure_error_and_unscored(task):
    out = await run_attempt(task, Crashes())
    assert out.execution_status == "infrastructure_error"
    assert out.evaluation_status == "pending" and out.scores == []
    assert "provider unreachable" in out.error


async def test_trace_has_lifecycle_bookends(task):
    out = await run_attempt(task, ScriptedAdapter("correct_refund"))
    types = [e.event_type for e in out.events]
    assert types[0] == "attempt_started" and types[-1] == "attempt_finished"
    assert "agent_finished" in types and "evaluator_finished" in types
    assert [e.sequence_number for e in out.events] == list(range(len(out.events)))
