import json

import pytest

from app.environments.support.env import SupportEnvironment
from app.execution.limits import LimitExceeded, Limits
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway


@pytest.fixture
def gw(task):
    return ToolGateway(SupportEnvironment(task), TraceRecorder("a1"), Limits(task.evaluation.limits))


async def test_unknown_tool_rejected(gw):
    r = await gw.execute("drop_database", {}, "c1")
    assert not r.ok and r.error_code == "UNKNOWN_TOOL"


async def test_extra_fields_and_wrong_types_rejected(gw):
    extra = await gw.execute("get_order", {"order_id": "ord_1042", "customer_id": "cust_99"}, "c1")
    wrong = await gw.execute("issue_refund",
                             {"order_id": "ord_1042", "amount_minor": "100", "approval_id": "x"}, "c2")
    assert extra.error_code == wrong.error_code == "SCHEMA_INVALID"
    assert gw.env.refunds == []


async def test_malformed_json_string_rejected(gw):
    r = await gw.execute("get_order", "{not json", "c1")
    assert r.error_code == "SCHEMA_INVALID"


async def test_json_string_arguments_are_parsed(gw):
    r = await gw.execute("get_order", json.dumps({"order_id": "ord_1042"}), "c1")
    assert r.ok and r.content["id"] == "ord_1042"


async def test_identity_is_not_model_supplied(gw):
    # No tool accepts a customer id; ownership comes from the environment.
    params = {p for s in gw.visible_schemas() for p in s["parameters"]["properties"]}
    assert not {p for p in params if "customer" in p or "user" in p}
    r = await gw.execute("get_order", {"order_id": "ord_2077"}, "c1")
    assert not r.ok and "not found" in r.content["error"]


async def test_every_call_is_traced_with_ordered_sequence(gw):
    await gw.execute("get_order", {"order_id": "ord_1042"}, "c1")
    await gw.execute("get_order", {"order_id": "ord_2077"}, "c2")
    types = [e.event_type for e in gw.recorder.events]
    assert types == ["tool_requested", "tool_finished", "tool_requested", "tool_rejected"]
    assert [e.sequence_number for e in gw.recorder.events] == [0, 1, 2, 3]


async def test_tool_budget_enforced(task):
    task.evaluation.limits.max_tool_calls = 2
    gw = ToolGateway(SupportEnvironment(task), TraceRecorder("a1"), Limits(task.evaluation.limits))
    await gw.execute("search_policy", {"query": "refund"}, "c1")
    await gw.execute("search_policy", {"query": "refund"}, "c2")
    with pytest.raises(LimitExceeded) as exc:
        await gw.execute("search_policy", {"query": "refund"}, "c3")
    assert exc.value.kind == "budget_exceeded"


def test_secrets_redacted_in_trace():
    rec = TraceRecorder("a1")
    e = rec.emit("model_request_started", {"headers": {"Authorization": "Bearer x"}, "n": 1})
    assert e.payload["headers"]["Authorization"] == "[REDACTED]" and e.payload["n"] == 1
