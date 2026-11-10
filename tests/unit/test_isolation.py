import copy

from app.agents.scripted import ScriptedAdapter
from app.environments.support.env import SupportEnvironment
from app.execution.runner import run_attempt


def test_agent_view_excludes_fixture_and_grading_data(task):
    view = task.agent_view()
    assert set(view) == {"user_message", "authenticated_customer_id", "ticket_id"}
    blob = str(view)
    for hidden in ("249900", "inspection_passed", "refund_total_minor", "cust_99", "ord_2077",
                   "invariants", "approve"):
        assert hidden not in blob


def test_environments_do_not_share_state(task):
    a, b = SupportEnvironment(task), SupportEnvironment(task)
    assert a.environment_id != b.environment_id
    a.orders["ord_1042"]["refunded_amount_minor"] = 999
    a.tickets["tkt_5001"]["status"] = "closed"
    assert b.orders["ord_1042"]["refunded_amount_minor"] == 0
    assert b.tickets["tkt_5001"]["status"] == "open"


async def test_running_an_attempt_never_mutates_the_task_fixture(task):
    before = copy.deepcopy(task.model_dump())
    await run_attempt(task, ScriptedAdapter("correct_refund"))
    assert task.model_dump() == before


async def test_repeated_runs_start_clean_and_get_distinct_environments(task):
    first = await run_attempt(task, ScriptedAdapter("correct_refund"))
    second = await run_attempt(task, ScriptedAdapter("correct_refund"))
    assert first.environment_id != second.environment_id
    assert len(first.final_state["refunds"]) == len(second.final_state["refunds"]) == 1
    assert second.final_state["history"][0]["seq"] == 0  # history is per-attempt
