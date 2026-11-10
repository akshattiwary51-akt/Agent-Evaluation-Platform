"""Run one attempt: isolated environment -> agent -> snapshot -> independent evaluators.

Execution status and evaluation status are kept separate (guide section 18). Agent behavior
failures (wrong result, turn/tool/time limits) are preserved and scored; only infrastructure
errors leave the attempt unscored.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agents.base import AgentAdapter
from app.environments.support.env import SupportEnvironment
from app.evaluators import DEFAULT_EVALUATORS, EvalContext, Evaluator, Score, Verdict, compute_verdict
from app.execution.limits import LimitExceeded, Limits
from app.schemas.task import Task
from app.telemetry.recorder import TraceEventData, TraceRecorder
from app.tools.gateway import ToolCallRecord, ToolGateway

SCORED_STATUSES = {"completed", "timed_out", "budget_exceeded"}


@dataclass
class AttemptOutcome:
    attempt_id: str
    environment_id: str
    execution_status: str
    stop_reason: str | None
    final_response: str
    final_state: dict[str, Any]
    events: list[TraceEventData]
    tool_calls: list[ToolCallRecord]
    scores: list[Score] = field(default_factory=list)
    verdict: Verdict | None = None
    evaluation_status: str = "pending"
    error: str | None = None
    model_calls: list[dict[str, Any]] = field(default_factory=list)


def _run_evaluators(evaluators: list[Evaluator], ctx: EvalContext) -> list[Score]:
    scores: list[Score] = []
    for ev in evaluators:
        try:
            scores.extend(ev.evaluate(ctx))
        except Exception as exc:  # an evaluator bug must never read as agent pass/fail
            scores.append(Score(ev.name, ev.version, "evaluation_error",
                                reason_code="EVALUATOR_EXCEPTION",
                                evidence={"error": f"{type(exc).__name__}: {exc}"}))
    return scores


async def run_attempt(task: Task, adapter: AgentAdapter, *,
                      evaluators: list[Evaluator] | None = None,
                      attempt_id: str | None = None,
                      should_stop: Callable[[], str | None] | None = None) -> AttemptOutcome:
    attempt_id = attempt_id or uuid.uuid4().hex
    env = SupportEnvironment(task)  # fresh, isolated state per attempt
    recorder = TraceRecorder(attempt_id)
    limits = Limits(task.evaluation.limits, should_stop)
    gateway = ToolGateway(env=env, recorder=recorder, limits=limits)

    recorder.emit("attempt_started", {"task_id": task.id, "task_version": task.version,
                                      "adapter": adapter.name,
                                      "environment_id": env.environment_id})
    started = time.monotonic()
    execution_status, stop_reason, final_response, error = "completed", None, "", None
    try:
        result = await asyncio.wait_for(
            adapter.run(task.agent_view(), gateway, recorder, limits),
            timeout=limits.timeout_seconds)
        final_response, stop_reason = result.final_response, result.stop_reason
    except LimitExceeded as exc:
        execution_status, stop_reason, error = exc.kind, exc.kind, exc.detail
    except asyncio.TimeoutError:
        execution_status, stop_reason = "timed_out", "timed_out"
        error = f"exceeded {limits.timeout_seconds}s"
    except Exception as exc:  # adapter/infrastructure failure: do not score as agent behavior
        execution_status, stop_reason = "infrastructure_error", "adapter_exception"
        error = f"{type(exc).__name__}: {exc}"

    snapshot = env.snapshot()
    recorder.emit("agent_finished", {"stop_reason": stop_reason, "error": error,
                                     "final_response": final_response},
                  status=execution_status, duration_ms=(time.monotonic() - started) * 1000)

    outcome = AttemptOutcome(
        attempt_id=attempt_id, environment_id=env.environment_id,
        execution_status=execution_status, stop_reason=stop_reason,
        final_response=final_response, final_state=snapshot, events=recorder.events,
        tool_calls=gateway.log, error=error, model_calls=recorder.model_calls)

    if execution_status in SCORED_STATUSES:
        ctx = EvalContext(task=task, final_state=snapshot, tool_calls=gateway.log,
                          final_response=final_response, stop_reason=stop_reason,
                          execution_status=execution_status)
        outcome.scores = _run_evaluators(evaluators or DEFAULT_EVALUATORS, ctx)
        outcome.verdict = compute_verdict(outcome.scores)
        outcome.evaluation_status = outcome.verdict.evaluation_status
        for s in outcome.scores:
            recorder.emit("evaluator_finished", {"evaluator": s.evaluator,
                                                 "dimension": s.dimension,
                                                 "reason_code": s.reason_code}, status=s.status)

    recorder.emit("attempt_finished", {"execution_status": execution_status,
                                       "evaluation_status": outcome.evaluation_status},
                  status=execution_status)
    return outcome
