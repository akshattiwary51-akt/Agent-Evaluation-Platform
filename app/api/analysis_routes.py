from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis import comparison, costing, export, scorecard
from app.api.deps import get_session
from app.db import models as m

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _exp(s: Session, experiment_id: str) -> m.Experiment:
    exp = s.get(m.Experiment, experiment_id)
    if exp is None:
        raise HTTPException(404, "experiment not found")
    return exp


@router.get("/experiments/{experiment_id}/scorecards")
def scorecards(experiment_id: str, s: SessionDep) -> dict[str, Any]:
    return scorecard.experiment_scorecards(s, _exp(s, experiment_id))


class SideModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_id: str
    config_id: str


class ComparisonRequest(BaseModel):
    """Either two explicit sides, or the shorthand: one experiment + two of its configs."""
    model_config = ConfigDict(extra="forbid")
    baseline: SideModel | None = None
    candidate: SideModel | None = None
    experiment_id: str | None = None
    baseline_config_id: str | None = None
    candidate_config_id: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> "ComparisonRequest":
        shorthand = (self.experiment_id, self.baseline_config_id, self.candidate_config_id)
        if self.baseline and self.candidate and not any(shorthand):
            return self
        if all(shorthand) and not (self.baseline or self.candidate):
            self.baseline = SideModel(experiment_id=self.experiment_id, config_id=self.baseline_config_id)
            self.candidate = SideModel(experiment_id=self.experiment_id, config_id=self.candidate_config_id)
            return self
        raise ValueError("give baseline+candidate, or experiment_id+baseline_config_id+"
                         "candidate_config_id")


@router.post("/comparisons")
def compare(req: ComparisonRequest, s: SessionDep) -> dict[str, Any]:
    try:
        return comparison.compare(
            s, comparison.Side(req.baseline.experiment_id, req.baseline.config_id),
            comparison.Side(req.candidate.experiment_id, req.candidate.config_id))
    except comparison.ComparisonError as exc:
        raise HTTPException(exc.status_code, {"compatible": False, "problems": exc.problems}) from exc


@router.get("/experiments/{experiment_id}/export")
def export_experiment(experiment_id: str, s: SessionDep,
                      format: Literal["json", "csv"] = "json") -> Response:
    report = export.build_report(s, _exp(s, experiment_id))
    if format == "csv":
        return Response(export.runs_csv(report), media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="experiment_{experiment_id}.csv"'})
    import json
    return Response(json.dumps(report, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition":
                             f'attachment; filename="experiment_{experiment_id}.json"'})


@router.get("/experiments/{experiment_id}/projection")
def projection(experiment_id: str, s: SessionDep, planned_runs: int = Query(..., ge=1),
               headroom: float = Query(0.2, ge=0, le=5)) -> dict[str, Any]:
    """Planning estimate: pilot mean cost per run x planned runs, plus headroom.
    Not a billing guarantee."""
    exp = _exp(s, experiment_id)
    prices = costing.load_prices(s)
    out = {}
    for c in exp.manifest["configs"]:
        facts = scorecard.load_run_facts(s, exp.id, c["id"])
        cost = costing.cost_calls([x for f in facts for x in f.model_calls], prices)
        if cost.status == "unknown" or not facts:
            out[c["name"]] = {"estimate": None, "reason": "pilot cost is unknown (missing usage or "
                              "price); register prices via POST /prices"}
            continue
        per_run = cost.total / len(facts)
        out[c["name"]] = {"pilot_runs": len(facts), "mean_cost_per_run": per_run,
                          "currency": cost.currency, "planned_runs": planned_runs,
                          "headroom": headroom,
                          "estimate": per_run * planned_runs * (1 + headroom),
                          "note": "planning estimate, not a billing guarantee"}
    return out


class PriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    model: str
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    cached_input_per_million: float | None = Field(default=None, ge=0)
    currency: str = "USD"
    effective_date: datetime
    source: str = Field(min_length=1, description="where the rate came from (URL or note)")


@router.post("/prices", status_code=201)
def add_price(req: PriceRequest, s: SessionDep) -> dict[str, Any]:
    eff = req.effective_date if req.effective_date.tzinfo else req.effective_date.replace(tzinfo=timezone.utc)
    pv = m.PriceVersion(provider=req.provider, model=req.model, currency=req.currency,
                        input_per_million=req.input_per_million,
                        cached_input_per_million=req.cached_input_per_million,
                        output_per_million=req.output_per_million, effective_date=eff,
                        source=req.source)
    s.add(pv)
    s.commit()
    return {"id": pv.id}


@router.get("/prices")
def list_prices(s: SessionDep) -> list[dict[str, Any]]:
    return [{"id": p.id, "provider": p.provider, "model": p.model, "currency": p.currency,
             "input_per_million": float(p.input_per_million),
             "output_per_million": float(p.output_per_million),
             "cached_input_per_million": (float(p.cached_input_per_million)
                                          if p.cached_input_per_million is not None else None),
             "effective_date": p.effective_date.isoformat(), "source": p.source}
            for p in s.scalars(select(m.PriceVersion).order_by(
                m.PriceVersion.provider, m.PriceVersion.model, m.PriceVersion.effective_date))]
