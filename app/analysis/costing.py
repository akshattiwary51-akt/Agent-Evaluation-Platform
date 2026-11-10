"""Resolve dated prices and cost model usage. Missing usage or price means UNKNOWN, not free."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.accounting.cost import Price, estimate_cost
from app.db import models as m
from app.providers.base import Usage


@dataclass
class CallRow:
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    at: datetime | None  # attempt start, used to pick the price in effect


@dataclass
class CostResult:
    status: str                      # none_incurred | estimated | unknown
    total: float | None
    currency: str | None = None
    unknown_calls: int = 0
    known_lower_bound: float | None = None
    pricing_basis: list[dict[str, Any]] = field(default_factory=list)


def load_prices(s: Session) -> dict[tuple[str, str], list[m.PriceVersion]]:
    out: dict[tuple[str, str], list[m.PriceVersion]] = {}
    for p in s.scalars(select(m.PriceVersion).order_by(m.PriceVersion.effective_date)):
        out.setdefault((p.provider, p.model), []).append(p)
    return out


def resolve_price(prices: dict, provider: str, model: str, at: datetime | None) -> m.PriceVersion | None:
    candidates = prices.get((provider, model), [])
    if not candidates:
        return None
    if at is None:
        return candidates[-1]
    eligible = [p for p in candidates if p.effective_date <= at]
    return eligible[-1] if eligible else None


def _to_price(p: m.PriceVersion) -> Price:
    return Price(p.provider, p.model, float(p.input_per_million), float(p.output_per_million),
                 float(p.cached_input_per_million) if p.cached_input_per_million is not None else None,
                 p.currency, p.effective_date.date().isoformat(), p.source or "")


def cost_calls(calls: list[CallRow], prices: dict) -> CostResult:
    if not calls:
        return CostResult("none_incurred", 0.0)
    total, unknown, known_sum, currencies, basis = 0.0, 0, 0.0, set(), {}
    for c in calls:
        pv = resolve_price(prices, c.provider, c.model, c.at)
        est = (estimate_cost(Usage(c.input_tokens, c.output_tokens, c.cached_input_tokens),
                             _to_price(pv)) if pv else None)
        if est is None:
            unknown += 1
            continue
        known_sum += est
        currencies.add(pv.currency)
        basis[pv.id] = {"price_version_id": pv.id, "provider": pv.provider, "model": pv.model,
                        "currency": pv.currency, "effective_date": pv.effective_date.date().isoformat(),
                        "source": pv.source, "input_per_million": float(pv.input_per_million),
                        "output_per_million": float(pv.output_per_million)}
    if len(currencies) > 1:
        return CostResult("unknown", None, None, unknown, known_sum, list(basis.values()))
    currency = next(iter(currencies), None)
    if unknown:
        return CostResult("unknown", None, currency, unknown, known_sum, list(basis.values()))
    return CostResult("estimated", known_sum, currency, 0, known_sum, list(basis.values()))
