"""Task schema with a hard split between the agent-visible and evaluator-visible views."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ToolName = Literal["get_order", "get_return_status", "get_refund_status", "search_policy",
                   "request_refund_approval", "issue_refund", "update_ticket"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentInput(_Strict):
    """Everything the agent is legitimately told. Nothing from the fixture or grading spec."""

    user_message: str
    authenticated_customer_id: str
    ticket_id: str


class OrderFixture(_Strict):
    id: str
    customer_id: str
    paid_amount_minor: int = Field(gt=0)
    refunded_amount_minor: int = Field(default=0, ge=0)
    currency: str = "INR"
    status: str = "delivered"
    # Free text returned inside tool data. Used to plant untrusted instructions.
    internal_note: str | None = None


class ReturnFixture(_Strict):
    received: bool
    inspection_passed: bool | None = None  # None = inspection pending / not applicable
    inspector_note: str | None = None


class TicketFixture(_Strict):
    id: str
    status: str = "open"


class FaultSpec(_Strict):
    """Fault keyed to a *logical operation* (the tool), not a global call number, so two agents
    taking different paths still face the same faults.

    service_error: the call fails before any effect; `times` calls fail, then it works.
    timeout_after_effect (issue_refund only): the effect IS applied but the agent is told it
    timed out, leaving the outcome uncertain until it checks get_refund_status.
    """

    op: ToolName
    kind: Literal["service_error", "timeout_after_effect"]
    times: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "FaultSpec":
        if self.kind == "timeout_after_effect" and self.op != "issue_refund":
            raise ValueError("timeout_after_effect is only supported for issue_refund")
        return self


class EnvironmentFixture(_Strict):
    order: OrderFixture
    # Return status for `order`.
    return_: ReturnFixture = Field(alias="return")
    ticket: TicketFixture
    # Orders owned by other customers; present so access-boundary behavior is testable.
    extra_orders: list[OrderFixture] = Field(default_factory=list)
    faults: list[FaultSpec] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ReviewerBehavior(_Strict):
    """Deterministic approval simulator. Reported as *simulated* approval outcomes."""

    decision: Literal["approve", "reject", "expire"] = "approve"
    applies_to: str = "matching_eligible_refund_proposal"


class LimitsSpec(_Strict):
    max_model_turns: int = 12
    max_tool_calls: int = 20
    timeout_seconds: float = 90


class OutcomeSpec(_Strict):
    refund_total_minor: int | None = None
    ticket_status: str | list[str] | None = None      # observed must equal / be one of
    ticket_status_not: list[str] = Field(default_factory=list)  # observed must not be any of


class ResponseSpec(_Strict):
    no_false_refund_claim: bool = True
    must_communicate_refund: bool = False
    must_mention: list[str] = Field(default_factory=list)      # every one (case-insensitive)
    must_mention_any: list[str] = Field(default_factory=list)  # at least one
    must_not_mention: list[str] = Field(default_factory=list)  # none
    must_ask_question: bool = False                            # response contains "?"


class EvaluationSpec(_Strict):
    outcome: OutcomeSpec = Field(default_factory=OutcomeSpec)
    invariants: list[str] = Field(default_factory=list)
    required_behavior: list[str] = Field(default_factory=list)
    response: ResponseSpec = Field(default_factory=ResponseSpec)
    limits: LimitsSpec = Field(default_factory=LimitsSpec)


class Task(_Strict):
    id: str
    version: int
    category: str
    # Scenario family: variants of one situation. Splits are assigned per family so
    # near-duplicates never straddle development and held-out sets.
    family: str | None = None
    difficulty: str = "medium"
    split: Literal["development", "validation", "test"] = "development"
    agent_input: AgentInput
    environment_fixture: EnvironmentFixture
    reviewer_behavior: ReviewerBehavior = Field(default_factory=ReviewerBehavior)
    evaluation: EvaluationSpec

    def agent_view(self) -> dict[str, Any]:
        """The ONLY task data an agent adapter may receive."""
        return self.agent_input.model_dump()


def load_task(path: str | Path) -> Task:
    with open(path, encoding="utf-8") as fh:
        return Task.model_validate(yaml.safe_load(fh))
