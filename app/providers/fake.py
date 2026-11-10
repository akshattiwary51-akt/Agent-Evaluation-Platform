"""Scripted provider for tests: replays canned responses, no network, no cost."""
from __future__ import annotations

from typing import Any, Callable

from app.providers.base import ModelResponse, ProviderToolCall, Usage


def reply(text: str = "", *calls: tuple[str, dict[str, Any]], usage: Usage | None = None) -> Callable:
    """Build a response step. Tool-call ids are assigned from the turn number."""
    def build(turn: int, messages: list[dict[str, Any]]) -> ModelResponse:
        return ModelResponse(
            text=text,
            tool_calls=[ProviderToolCall(f"t{turn}_{i}", n, a) for i, (n, a) in enumerate(calls)],
            usage=usage or Usage(input_tokens=100, output_tokens=20, cached_input_tokens=0),
            resolved_model="fake-model-1")
    return build


class FakeProvider:
    name = "fake"

    def __init__(self, steps: list[Callable]):
        self._steps = steps
        self.requests: list[dict[str, Any]] = []

    async def generate(self, *, model, system, messages, tools, timeout, max_output_tokens):
        self.requests.append({"model": model, "system": system, "n_messages": len(messages),
                              "tools": [t["name"] for t in tools], "timeout": timeout})
        turn = len(self.requests) - 1
        step = self._steps[min(turn, len(self._steps) - 1)]  # last step repeats
        return step(turn, messages)
