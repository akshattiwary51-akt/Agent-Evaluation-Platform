"""Build an agent adapter from a stored AgentConfig row."""
from __future__ import annotations

import os

from app.agents.base import AgentAdapter
from app.agents.prompts import PROMPTS
from app.agents.scripted import BEHAVIORS, ScriptedAdapter
from app.agents.tool_loop import ToolLoopAdapter
from app.db import models as m


def build_adapter(config: m.AgentConfig) -> AgentAdapter:
    cfg = config.settings or {}
    if config.adapter == "scripted":
        return ScriptedAdapter(cfg["behavior"])
    if config.adapter == "tool_loop":
        if cfg.get("provider") != "anthropic":
            raise ValueError(f"unsupported provider {cfg.get('provider')!r}")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set for this worker")
        from app.providers.anthropic import AnthropicProvider
        prompt = cfg["prompt"]
        return ToolLoopAdapter(AnthropicProvider(key), config.model or cfg["model"],
                               PROMPTS[prompt], prompt, cfg.get("max_output_tokens", 1024))
    raise ValueError(f"unknown adapter {config.adapter!r}")
