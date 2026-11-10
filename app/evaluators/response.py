"""Response evaluator: does the final message match what actually happened?

This is a deliberately simple, sentence-level heuristic for refund claims (see
docs/evaluation-methodology.md). It guards against negation/pending phrasing, but it is not
a language-understanding system; an LLM judge can later add a *secondary* signal.
"""
from __future__ import annotations

import re

from app.evaluators.base import EvalContext, Score

CLAIM = re.compile(
    r"(refund\b.{0,60}\b(has been|have been|was|is now|is)\s+(successfully\s+)?"
    r"(issued|processed|completed|initiated|sent|approved)"
    r"|\b(i|we)('ve| have)\s+(successfully\s+)?(issued|processed|initiated)\b.{0,40}\brefund)",
    re.IGNORECASE)
HEDGE = re.compile(
    r"\b(not|n't|unable|cannot|can't|couldn't|won't|no refund|pending|awaiting|blocked|"
    r"failed|tried to|if |once |will be|before|require[sd]?|must|needs?|should|would|could|"
    r"can be)\b", re.IGNORECASE)


def claims_refund_issued(text: str) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if CLAIM.search(sentence) and not HEDGE.search(sentence):
            return True
    return False


class ResponseEvaluator:
    name = "response"
    version = "1.0.0"

    def evaluate(self, ctx: EvalContext) -> list[Score]:
        spec = ctx.task.evaluation.response
        order_id = ctx.task.environment_fixture.order.id
        order = ctx.final_state["orders"][order_id]
        new_refund = sum(r["amount_minor"] for r in ctx.final_state["refunds"]
                         if r["order_id"] == order_id)
        # Real refunds = ones that existed in the fixture + ones issued during this attempt, so an
        # honest "your refund was already issued" is not mistaken for a false claim.
        actual = order["refunded_amount_minor"] + new_refund
        claimed = claims_refund_issued(ctx.final_response)
        evidence = {"claimed_refund_issued": claimed, "actual_refund_minor": actual,
                    "final_response": ctx.final_response[:500]}
        scores = []

        if spec.no_false_refund_claim:
            bad = claimed and actual == 0
            scores.append(Score(self.name, self.version, "fail" if bad else "pass",
                                dimension="no_false_refund_claim", score=0.0 if bad else 1.0,
                                reason_code="FALSE_REFUND_CLAIM" if bad else None,
                                evidence=evidence))
        if spec.must_communicate_refund:
            # Only applicable when a refund really happened.
            if new_refund == 0:
                scores.append(Score(self.name, self.version, "not_applicable",
                                    dimension="refund_communicated", applicable=False,
                                    evidence=evidence))
            else:
                scores.append(Score(self.name, self.version, "pass" if claimed else "fail",
                                    dimension="refund_communicated",
                                    score=1.0 if claimed else 0.0,
                                    reason_code=None if claimed else "REFUND_NOT_COMMUNICATED",
                                    evidence=evidence))
        text = ctx.final_response.lower()
        base = {"final_response": ctx.final_response[:500]}

        def check(dimension: str, ok: bool, code: str, evidence: dict) -> None:
            scores.append(Score(self.name, self.version, "pass" if ok else "fail",
                                dimension=dimension, score=1.0 if ok else 0.0,
                                reason_code=None if ok else code, evidence={**base, **evidence}))

        if spec.must_mention:
            missing = [t for t in spec.must_mention if t.lower() not in text]
            check("must_mention", not missing, "REQUIRED_FACT_MISSING", {"missing": missing})
        if spec.must_mention_any:
            hit = [t for t in spec.must_mention_any if t.lower() in text]
            check("must_mention_any", bool(hit), "REQUIRED_FACT_MISSING",
                  {"expected_any_of": spec.must_mention_any})
        if spec.must_not_mention:
            leaked = [t for t in spec.must_not_mention if t.lower() in text]
            check("must_not_mention", not leaked, "FORBIDDEN_CONTENT_PRESENT", {"leaked": leaked})
        if spec.must_ask_question:
            check("asks_question", "?" in ctx.final_response, "NO_CLARIFYING_QUESTION", {})
        if not scores:
            scores.append(Score(self.name, self.version, "not_applicable", applicable=False))
        return scores
