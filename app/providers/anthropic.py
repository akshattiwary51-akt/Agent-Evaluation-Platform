"""Anthropic Messages API provider, via plain HTTPX (no SDK dependency).

NOTE: the request/response mapping is covered by offline tests with canned payloads, but it has
not been exercised against the live API in this repo's CI. Run `scripts/evaluate.py` once with a
real key and check the trace before trusting results.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from app.providers.base import ModelResponse, ProviderError, ProviderToolCall, Usage

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


def to_wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "user":
            wire.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks: list[dict[str, Any]] = []
            if m["text"]:
                blocks.append({"type": "text", "text": m["text"]})
            for c in m["tool_calls"]:
                args = c["arguments"]
                if isinstance(args, str):  # unparsed; keep the turn well-formed
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": args})
            wire.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                     "content": m["content"]}
            if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                wire[-1]["content"].append(block)  # results of one turn share a user message
            else:
                wire.append({"role": "user", "content": [block]})
    return wire


def parse_response(data: dict[str, Any]) -> ModelResponse:
    text, calls = [], []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text.append(block["text"])
        elif block.get("type") == "tool_use":
            calls.append(ProviderToolCall(block["id"], block["name"], block.get("input", {})))
    u = data.get("usage") or {}
    usage = Usage(input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"),
                  cached_input_tokens=u.get("cache_read_input_tokens"))
    return ModelResponse(text="".join(text), tool_calls=calls, usage=usage,
                         stop_reason=data.get("stop_reason"), resolved_model=data.get("model"),
                         raw={"id": data.get("id")})


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        self._api_key = api_key
        self._client = client

    async def generate(self, *, model, system, messages, tools, timeout, max_output_tokens):
        body = {
            "model": model, "max_tokens": max_output_tokens, "system": system,
            "messages": to_wire_messages(messages),
            "tools": [{"name": t["name"], "description": t["description"],
                       "input_schema": t["parameters"]} for t in tools],
        }
        headers = {"x-api-key": self._api_key, "anthropic-version": API_VERSION,
                   "content-type": "application/json"}
        client = self._client or httpx.AsyncClient()
        try:
            resp = await client.post(API_URL, json=body, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ProviderError(f"request failed: {type(exc).__name__}") from exc
        finally:
            if self._client is None:
                await client.aclose()
        if resp.status_code != 200:
            raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return parse_response(resp.json())
