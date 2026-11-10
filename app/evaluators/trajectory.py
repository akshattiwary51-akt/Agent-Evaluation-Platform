"""Trajectory evaluator: required investigations (as partial-order constraints) and redundancy.

Required behaviors are matched one-to-one against distinct observed calls, so repeating one
correct call cannot satisfy two requirements. Order lookup and return lookup may occur in
either order; both only need to precede the first refund attempt.
"""
from __future__ import annotations

from collections import Counter

from app.evaluators.base import EvalContext, Score

# requirement -> tool that must have succeeded before the first issue_refund attempt
REQUIREMENTS = {
    "inspect_order_before_refund": "get_order",
    "inspect_return_before_refund": "get_return_status",
}


class TrajectoryEvaluator:
    name = "trajectory"
    version = "1.0.0"

    def evaluate(self, ctx: EvalContext) -> list[Score]:
        calls = ctx.tool_calls
        first_refund = next((i for i, c in enumerate(calls) if c.tool_name == "issue_refund"), None)
        required = ctx.task.evaluation.required_behavior
        scores: list[Score] = []

        if required:
            used: set[int] = set()
            satisfied, missing = [], []
            for req in required:
                tool = REQUIREMENTS.get(req)
                if tool is None:
                    scores.append(Score(self.name, self.version, "evaluation_error",
                                        dimension=req, reason_code="UNKNOWN_REQUIREMENT"))
                    continue
                limit = first_refund if first_refund is not None else len(calls)
                match = next((i for i, c in enumerate(calls[:limit])
                              if i not in used and c.tool_name == tool and c.outcome == "ok"), None)
                if match is None:
                    missing.append(req)
                else:
                    used.add(match)
                    satisfied.append(req)
            graded = len(satisfied) + len(missing)
            if graded:
                recall = len(satisfied) / graded
                scores.append(Score(
                    self.name, self.version, "pass" if not missing else "fail",
                    dimension="required_calls", score=recall,
                    reason_code=None if not missing else "REQUIRED_CALL_MISSING",
                    evidence={"satisfied": satisfied, "missing": missing}))
        else:
            scores.append(Score(self.name, self.version, "not_applicable",
                                dimension="required_calls", applicable=False))

        if not calls:
            scores.append(Score(self.name, self.version, "not_applicable",
                                dimension="redundancy", applicable=False))
        else:
            seen = Counter((c.tool_name, repr(c.arguments)) for c in calls if c.outcome == "ok")
            redundant = sum(n - 1 for n in seen.values())
            scores.append(Score(
                self.name, self.version, "pass" if redundant == 0 else "partial",
                dimension="redundancy", score=1 - redundant / len(calls),
                reason_code=None if redundant == 0 else "REDUNDANT_CALLS",
                evidence={"redundant_calls": redundant, "total_calls": len(calls)}))

        valid = sum(1 for c in calls if c.reason_code != "SCHEMA_INVALID")
        scores.append(Score(
            self.name, self.version, "not_applicable" if not calls else "pass",
            dimension="schema_validity", applicable=bool(calls),
            score=(valid / len(calls)) if calls else None,
            evidence={"valid_calls": valid, "attempted_calls": len(calls)}))
        return scores
