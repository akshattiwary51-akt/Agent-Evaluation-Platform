"""Server-rendered dashboard (Jinja2, no JS framework). Five screens per guide section 25."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis import comparison, costing, scorecard
from app.analysis.state_diff import state_diff
from app.api.deps import get_session
from app.db import models as m
from app.metrics import experiment_summary

router = APIRouter(prefix="/ui", include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
SessionDep = Annotated[Session, Depends(get_session)]
PAGE = 50


# ------------------------------------------------------------------ template helpers
def pct(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:,}" if isinstance(v, int) else f"{v:,.{digits}f}"


def money(v: float | None, currency: str | None = None) -> str:
    return "—" if v is None else f"{currency or ''} {v:,.4f}".strip()


def secs(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"


def signed(v: float | None, digits: int = 3) -> str:
    return "—" if v is None else f"{v:+,.{digits}f}"


def when(v: datetime | None) -> str:
    return "—" if v is None else v.strftime("%Y-%m-%d %H:%M:%S UTC")


for name, fn in dict(pct=pct, num=num, money=money, secs=secs, signed=signed, when=when).items():
    templates.env.filters[name] = fn


def _404(obj: Any, what: str) -> Any:
    if obj is None:
        raise HTTPException(404, f"{what} not found")
    return obj


def _render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, name, ctx)


# ------------------------------------------------------------------ 1. experiment list
@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def experiment_list(request: Request, s: SessionDep) -> HTMLResponse:
    prices = costing.load_prices(s)
    rows = []
    for exp in s.scalars(select(m.Experiment).order_by(m.Experiment.created_at.desc()).limit(50)):
        summ = experiment_summary(s, exp)
        finished = sum(n for k, n in summ["execution"].items()
                       if k in {"completed", "timed_out", "budget_exceeded", "infrastructure_error",
                                "cancelled"})
        calls = [costing.CallRow(mc.provider, mc.model, mc.input_tokens, mc.output_tokens,
                                 mc.cached_input_tokens, a.started_at)
                 for mc, a in s.execute(select(m.ModelCall, m.Attempt)
                                        .join(m.Attempt, m.Attempt.id == m.ModelCall.attempt_id)
                                        .join(m.Run, m.Run.id == m.Attempt.run_id)
                                        .where(m.Run.experiment_id == exp.id))]
        spend = costing.cost_calls(calls, prices)
        dv = s.get(m.DatasetVersion, exp.dataset_version_id)
        ds = s.get(m.Dataset, dv.dataset_id)
        rows.append({"exp": exp, "summary": summ, "finished": finished, "spend": spend,
                     "dataset": f"{ds.name} {dv.version}",
                     "configs": [c["name"] for c in exp.manifest["configs"]]})
    return _render(request, "experiments.html", rows=rows)


# ------------------------------------------------------------------ 2. experiment detail
@router.get("/experiments/{experiment_id}", response_class=HTMLResponse)
def experiment_detail(request: Request, experiment_id: str, s: SessionDep,
                      config: str | None = None, category: str | None = None,
                      task: str | None = None, execution_status: str | None = None,
                      evaluation_status: str | None = None,
                      offset: int = Query(0, ge=0)) -> HTMLResponse:
    exp = _404(s.get(m.Experiment, experiment_id), "experiment")
    cards = scorecard.experiment_scorecards(s, exp)
    q = (select(m.Run, m.TaskVersion, m.AgentConfig)
         .join(m.TaskVersion, m.TaskVersion.id == m.Run.task_version_id)
         .join(m.AgentConfig, m.AgentConfig.id == m.Run.agent_config_id)
         .where(m.Run.experiment_id == exp.id))
    for col, val in ((m.AgentConfig.name, config), (m.TaskVersion.category, category),
                     (m.TaskVersion.task_id, task), (m.Run.execution_status, execution_status),
                     (m.Run.evaluation_status, evaluation_status)):
        if val:
            q = q.where(col == val)
    total = s.scalar(select(func.count()).select_from(q.subquery()))
    runs = s.execute(q.order_by(m.TaskVersion.task_id, m.AgentConfig.name, m.Run.repetition)
                     .limit(PAGE).offset(offset)).all()
    categories = sorted({c for card in cards.values() for c in card["by_category"]})
    cfg_ids = {c["name"]: c["id"] for c in exp.manifest["configs"]}
    names = list(cfg_ids)
    return _render(request, "experiment.html", exp=exp, summary=experiment_summary(s, exp),
                   cards=cards, categories=categories, runs=runs, total=total, offset=offset,
                   page=PAGE, cfg_ids=cfg_ids,
                   pairs=[(names[0], n) for n in names[1:]],
                   filters={"config": config, "category": category, "task": task,
                            "execution_status": execution_status,
                            "evaluation_status": evaluation_status})


# ------------------------------------------------------------------ 3. run detail
@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str, s: SessionDep,
               attempt: str | None = None) -> HTMLResponse:
    run = _404(s.get(m.Run, run_id), "run")
    tv, cfg = s.get(m.TaskVersion, run.task_version_id), s.get(m.AgentConfig, run.agent_config_id)
    attempts = s.scalars(select(m.Attempt).where(m.Attempt.run_id == run_id)
                         .order_by(m.Attempt.attempt_number)).all()
    shown = next((a for a in attempts if a.id == (attempt or run.final_attempt_id)), None) \
        or (attempts[-1] if attempts else None)
    events, scores, tools, diff = [], [], [], []
    if shown:
        events = s.scalars(select(m.TraceEvent).where(m.TraceEvent.attempt_id == shown.id)
                           .order_by(m.TraceEvent.sequence_number)).all()
        scores = s.scalars(select(m.Score).where(m.Score.attempt_id == shown.id)
                           .order_by(m.Score.evaluator_name, m.Score.dimension)).all()
        tools = s.scalars(select(m.ToolCall).where(m.ToolCall.attempt_id == shown.id)).all()
        if shown.final_state:
            diff = state_diff(tv.environment_fixture, shown.final_state)
    t0 = events[0].timestamp if events else None
    timeline = [{"offset_ms": (e.timestamp - t0).total_seconds() * 1000, "e": e} for e in events]
    return _render(request, "run.html", run=run, tv=tv, cfg=cfg, attempts=attempts, shown=shown,
                   timeline=timeline, scores=scores, diff=diff,
                   is_final=bool(shown and shown.id == run.final_attempt_id),
                   n_tools=len(tools))


# ------------------------------------------------------------------ 4. comparison
@router.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request, s: SessionDep, experiment_id: str, baseline: str,
                 candidate: str, candidate_experiment_id: str | None = None) -> HTMLResponse:
    exp_a = _404(s.get(m.Experiment, experiment_id), "experiment")
    exp_b = _404(s.get(m.Experiment, candidate_experiment_id or experiment_id), "experiment")
    ids_a = {c["name"]: c["id"] for c in exp_a.manifest["configs"]}
    ids_b = {c["name"]: c["id"] for c in exp_b.manifest["configs"]}
    if baseline not in ids_a or candidate not in ids_b:
        raise HTTPException(404, "config not in experiment")
    try:
        result = comparison.compare(s, comparison.Side(exp_a.id, ids_a[baseline]),
                                    comparison.Side(exp_b.id, ids_b[candidate]))
        problems = None
    except comparison.ComparisonError as exc:
        if exc.status_code == 404:
            raise HTTPException(404, str(exc)) from exc
        result, problems = None, exc.problems
    return _render(request, "compare.html", r=result, problems=problems, exp_a=exp_a, exp_b=exp_b,
                   baseline=baseline, candidate=candidate, metric_dirs={
                       label: hb for label, _, hb in comparison.METRICS})


# ------------------------------------------------------------------ 5. benchmark results
@router.get("/benchmarks", response_class=HTMLResponse)
def benchmarks(request: Request) -> HTMLResponse:
    return _render(request, "benchmarks.html")
