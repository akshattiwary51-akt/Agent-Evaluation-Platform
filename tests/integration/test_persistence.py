"""Round-trip an attempt through the real schema (SQLite here; the Alembic migration is Postgres)."""
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.scripted import ScriptedAdapter
from app.db import models as m
from app.db.persist import persist_outcome
from app.execution.runner import run_attempt


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    m.Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


async def test_attempt_trace_and_scores_round_trip(session, task):
    out = await run_attempt(task, ScriptedAdapter("false_success"))
    attempt = persist_outcome(session, task, "scripted:false_success", out)
    session.commit()

    events = session.scalars(select(m.TraceEvent).where(m.TraceEvent.attempt_id == attempt.id)
                             .order_by(m.TraceEvent.sequence_number)).all()
    assert [e.event_type for e in events][0] == "attempt_started"
    fail = session.scalars(select(m.Score).where(m.Score.attempt_id == attempt.id,
                                                 m.Score.reason_code == "FALSE_REFUND_CLAIM")).one()
    assert fail.status == "fail" and fail.evidence["actual_refund_minor"] == 0
    assert session.get(m.Attempt, attempt.id).final_state["refunds"] == []


async def test_event_sequence_is_unique_per_attempt(session, task):
    out = await run_attempt(task, ScriptedAdapter("correct_refund"))
    attempt = persist_outcome(session, task, "scripted:correct_refund", out)
    session.flush()
    dup = out.events[0]
    session.add(m.TraceEvent(attempt_id=attempt.id, sequence_number=dup.sequence_number,
                             trace_id="t", span_id="s", event_type="x", timestamp=dup.timestamp))
    with pytest.raises(IntegrityError):
        session.flush()
