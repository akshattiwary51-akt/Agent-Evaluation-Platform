"""Outcome evaluator: reads the hidden final environment state, never the agent's words."""
from __future__ import annotations

from app.evaluators.base import EvalContext, Score


class OutcomeEvaluator:
    name = "outcome"
    version = "1.0.0"

    def evaluate(self, ctx: EvalContext) -> list[Score]:
        spec = ctx.task.evaluation.outcome
        order_id = ctx.task.environment_fixture.order.id
        scores: list[Score] = []

        if spec.refund_total_minor is not None:
            refunds = [r for r in ctx.final_state["refunds"] if r["order_id"] == order_id]
            observed = sum(r["amount_minor"] for r in refunds)
            expected = spec.refund_total_minor
            evidence = {"expected_amount_minor": expected, "observed_amount_minor": observed,
                        "refund_count": len(refunds)}
            if observed == expected and len(refunds) == (1 if expected else 0):
                status, code = "pass", None
            elif expected > 0 and observed == 0:
                status, code = "fail", "NO_REFUND"
            elif expected == 0 and observed > 0:
                status, code = "fail", "UNEXPECTED_REFUND"
            elif observed == expected:
                status, code = "fail", "MULTIPLE_REFUND_EFFECTS"
            else:
                status, code = "fail", "WRONG_REFUND_AMOUNT"
            scores.append(Score(self.name, self.version, status, dimension="refund_total",
                                score=1.0 if status == "pass" else 0.0, reason_code=code,
                                evidence=evidence))

            # A refund on any other order is a wrong-target effect, reported on its own.
            stray = [r for r in ctx.final_state["refunds"] if r["order_id"] != order_id]
            scores.append(Score(self.name, self.version, "fail" if stray else "pass",
                                dimension="other_orders_untouched", score=0.0 if stray else 1.0,
                                reason_code="REFUND_ON_WRONG_ORDER" if stray else None,
                                evidence={"stray_refunds": stray}))

        if spec.ticket_status is not None or spec.ticket_status_not:
            ticket_id = ctx.task.environment_fixture.ticket.id
            observed_status = ctx.final_state["tickets"][ticket_id]["status"]
            allowed = ([spec.ticket_status] if isinstance(spec.ticket_status, str)
                       else spec.ticket_status)
            ok = ((allowed is None or observed_status in allowed)
                  and observed_status not in spec.ticket_status_not)
            scores.append(Score(self.name, self.version, "pass" if ok else "fail",
                                dimension="ticket_status", score=1.0 if ok else 0.0,
                                reason_code=None if ok else "WRONG_TICKET_STATUS",
                                evidence={"expected": allowed,
                                          "must_not_be": spec.ticket_status_not,
                                          "observed": observed_status}))

        if not scores:
            scores.append(Score(self.name, self.version, "not_applicable", applicable=False))
        return scores
