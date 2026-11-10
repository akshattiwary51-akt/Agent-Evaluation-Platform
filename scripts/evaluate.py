"""Run the sample task against a real model with each prompt version. Spends real money.

    ANTHROPIC_API_KEY=... python -m scripts.evaluate --model <model-id> [--db]

Model ids and prices change; pass the id you intend to test. Cost is shown only if you also
pass --in-price/--out-price (USD per million tokens); otherwise it is reported as unknown.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from app.accounting.cost import Price, total_cost
from app.agents.prompts import PROMPTS
from app.agents.tool_loop import ToolLoopAdapter
from app.execution.runner import run_attempt
from app.providers.anthropic import AnthropicProvider
from app.providers.base import Usage
from app.schemas.task import load_task
from scripts.smoke import TASK_PATH


async def main(args: argparse.Namespace) -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY (use `make smoke` for the no-key scripted run).")
    task = load_task(TASK_PATH)
    provider = AnthropicProvider(key)
    price = (Price("anthropic", args.model, args.in_price, args.out_price)
             if args.in_price is not None and args.out_price is not None else None)
    session = None
    if args.db:
        from app.db.persist import persist_outcome
        from app.db.session import SessionLocal
        session = SessionLocal()
    for prompt_name in PROMPTS:
        adapter = ToolLoopAdapter(provider, args.model, PROMPTS[prompt_name], prompt_name)
        out = await run_attempt(task, adapter)
        usages = [Usage(c["input_tokens"], c["output_tokens"], c["cached_input_tokens"])
                  for c in out.model_calls]
        cost = total_cost(usages, price) if price else None
        failing = [f"{s.dimension}:{s.reason_code}" for s in out.scores
                   if s.status in {"fail", "partial", "evaluation_error"}]
        print(f"{prompt_name}: exec={out.execution_status} eval={out.evaluation_status} "
              f"calls={len(out.tool_calls)} model_turns={len(out.model_calls)} "
              f"cost={'unknown' if cost is None else f'{cost:.4f} USD'} "
              f"issues={failing or '-'}")
        if out.error:
            print(f"  error: {out.error}")
        if session:
            persist_outcome(session, task, adapter.name, out)
    if session:
        session.commit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--in-price", type=float)
    p.add_argument("--out-price", type=float)
    p.add_argument("--db", action="store_true")
    asyncio.run(main(p.parse_args()))
