"""Provider-neutral model interface.

Internal message format (what the tool loop keeps):
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "arguments"}]}
  {"role": "tool", "tool_call_id": str, "content": str}
Each provider converts this to its own wire format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(Exception):
    """Provider unreachable, rejected the request, or returned something unusable."""


@dataclass
class Usage:
    # None means unknown (not zero). Input is normalized to *uncached* input tokens;
    # cached reads are reported separately so cost never double counts them.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None


@dataclass
class ProviderToolCall:
    id: str
    name: str
    arguments: Any  # dict, or a raw string if the provider returned unparsed JSON


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ProviderToolCall]
    usage: Usage
    stop_reason: str | None = None
    resolved_model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def as_message(self) -> dict[str, Any]:
        return {"role": "assistant", "text": self.text,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                               for c in self.tool_calls]}


class ModelProvider(Protocol):
    name: str

    async def generate(self, *, model: str, system: str, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]], timeout: float,
                       max_output_tokens: int) -> ModelResponse: ...
