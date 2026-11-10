"""Phase 2 acceptance: a model-driven agent resolves the case and every model/tool operation is traced."""
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.agents.prompts import PROMPTS
from app.agents.tool_loop import ToolLoopAdapter
from app.db import models as m
from app.db.persist import persist_outcome
from app.execution.runner import run_attempt
from app.providers.base import ProviderError, Usage
from app.providers.fake import FakeProvider, reply

ORDER = {"order_id": "ord_1042"}


def happy_path() -> FakeProvider:
    # approval id is deterministic in a fresh environment: appr_1
    return FakeProvider([
        reply("Checking.", ("get_order", ORDER), ("get_return_status", ORDER)),
        reply("", ("request_refund_approval",
                   {**ORDER, "amount_minor": 249900, "reason": "Eligible return"})),
        reply("", ("issue_refund", {**ORDER, "amount_minor": 249900, "approval_id": "appr_1"})),
        reply("", ("update_ticket", {"ticket_id": "tkt_5001", "status": "resolved",
                                     "note": "Refund issued."})),
        reply("Your refund of 2,499.00 INR has been issued."),
    ])


def adapter(provider) -> ToolLoopAdapter:
    return ToolLoopAdapter(provider, "fake-model-1", PROMPTS["support-v2"], "support-v2")


async def test_model_agent_resolves_task_and_everything_is_traced(task):
    out = await run_attempt(task, adapter(happy_path()))
    assert out.execution_status == "completed" and out.evaluation_status == "pass"

    types = [e.event_type for e in out.events]
    assert types.count("model_request_started") == types.count("model_request_finished") == 5
    assert types.count("tool_requested") == types.count("tool_finished") == 5
    # Ordering: every request precedes its response, and model turns interleave with tools.
    assert types.index("model_request_finished") < types.index("tool_requested")
    assert len(out.model_calls) == 5 and out.model_calls[0]["input_tokens"] == 100


async def test_hidden_data_never_reaches_the_model(task):
    provider = happy_path()
    await run_attempt(task, adapter(provider))
    first = provider.requests[0]
    assert first["tools"] and "ord_2077" not in str(first) and "249900" not in str(first)


async def test_overconfident_model_is_caught_by_evaluators(task):
    provider = FakeProvider([reply("All done, your refund has been issued!")])
    out = await run_attempt(task, adapter(provider))
    assert out.execution_status == "completed" and out.evaluation_status == "fail"
    assert any(s.reason_code == "FALSE_REFUND_CLAIM" for s in out.scores)


async def test_turn_limit_is_scored_as_unsuccessful_with_reason(task):
    task.evaluation.limits.max_model_turns = 3
    provider = FakeProvider([reply("", ("search_policy", {"query": "refund"}))])  # repeats forever
    out = await run_attempt(task, adapter(provider))
    assert out.execution_status == "budget_exceeded" and out.stop_reason == "budget_exceeded"
    assert out.evaluation_status == "fail" and len(out.model_calls) == 3


async def test_schema_invalid_tool_call_is_recorded_and_agent_can_recover(task):
    provider = FakeProvider([
        reply("", ("get_order", {"order_id": 1042})),          # wrong type
        reply("Sorry, I could not look that up."),
    ])
    out = await run_attempt(task, adapter(provider))
    assert out.tool_calls[0].reason_code == "SCHEMA_INVALID"
    assert out.execution_status == "completed"


async def test_provider_outage_is_infrastructure_error_not_a_failed_agent(task):
    class Down:
        name = "down"
        async def generate(self, **kw):
            raise ProviderError("HTTP 529")
    out = await run_attempt(task, ToolLoopAdapter(Down(), "m", "sys"))
    assert out.execution_status == "infrastructure_error" and out.scores == []


async def test_unknown_usage_is_stored_as_null_not_zero(task):
    provider = FakeProvider([reply("Hello.", usage=Usage())])
    out = await run_attempt(task, adapter(provider))
    assert out.model_calls[0]["input_tokens"] is None
    engine = create_engine("sqlite://")
    m.Base.metadata.create_all(engine)
    with Session(engine) as s:
        persist_outcome(s, task, "tool_loop:fake", out)
        s.commit()
        row = s.scalars(select(m.ModelCall)).one()
        assert row.input_tokens is None and row.output_tokens is None
