"""Subprocess worker whose agent never finishes. Used to be killed with SIGKILL mid-attempt."""
import asyncio
import os
import sys
import threading

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.base import AgentResult
from app.execution.worker import Worker


class Hang:
    name = "hang"

    async def run(self, task_input, tools, events, limits):
        await tools.execute("search_policy", {"query": "refund"}, "c1")
        print("HUNG", flush=True)
        await asyncio.sleep(3600)
        return AgentResult("never", "model_final")


factory = sessionmaker(create_engine(os.environ["TEST_DATABASE_URL"]))
Worker(factory, lease_seconds=float(sys.argv[1]), heartbeat_seconds=0.2, concurrency=1,
       adapter_factory=lambda cfg: Hang()).run_forever(threading.Event())
