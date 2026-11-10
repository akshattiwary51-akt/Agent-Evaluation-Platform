"""Python tool-loop adapter: a small, fully instrumented agent loop over any ModelProvider.

Tool calls run sequentially. Every model request and response is traced, with usage kept as
reported (None = unknown). Provider failures propagate and are classified by the runner as
infrastructure errors; they are not scored as agent behavior.
"""
from __future__ import annotations

import time
from typing import Any

from app.agents.base import AgentResult
from app.providers.base import ModelProvider


class ToolLoopAdapter:
    def __init__(self, provider: ModelProvider, model: str, system_prompt: str,
                 prompt_name: str = "custom", max_output_tokens: int = 1024):
        self.name = f"tool_loop:{provider.name}:{model}:{prompt_name}"
        self._provider, self._model = provider, model
        self._system, self._max_out = system_prompt, max_output_tokens

    @staticmethod
    def _opening_message(task_input: dict[str, Any]) -> str:
        return (f"[Authenticated customer: {task_input['authenticated_customer_id']}; "
                f"ticket: {task_input['ticket_id']}]\n{task_input['user_message']}")

    async def run(self, task_input, tools, events, limits) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._opening_message(task_input)}]
        model_calls = events.model_calls

        while True:
            limits.register_model_turn()  # raises LimitExceeded past max_model_turns / deadline
            events.emit("model_request_started", {"model": self._model,
                                                  "n_messages": len(messages)})
            started = time.monotonic()
            response = await self._provider.generate(
                model=self._model, system=self._system, messages=messages,
                tools=tools.visible_schemas(), timeout=limits.remaining_seconds(),
                max_output_tokens=self._max_out)
            elapsed_ms = (time.monotonic() - started) * 1000

            usage = response.usage
            model_calls.append({
                "provider": self._provider.name, "model": self._model,
                "resolved_model": response.resolved_model,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens, "duration_ms": elapsed_ms})
            events.emit("model_request_finished", {
                "stop_reason": response.stop_reason, "text": response.text,
                "tool_calls": [c.name for c in response.tool_calls],
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens},
                status="ok", duration_ms=elapsed_ms)

            messages.append(response.as_message())
            if not response.tool_calls:
                return AgentResult(response.text, "model_final",
                                   {"turns": limits.model_turns})

            for call in response.tool_calls:
                result = await tools.execute(call.name, call.arguments, call.id)
                messages.append(result.as_tool_message(call.id))
