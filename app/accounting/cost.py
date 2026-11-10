"""Cost estimation. Prices are supplied by the caller (they change; never hard-code them)."""
from __future__ import annotations

from dataclasses import dataclass

from app.providers.base import Usage


@dataclass(frozen=True)
class Price:
    """Rates per million tokens, in one currency, with provenance."""
    provider: str
    model: str
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    currency: str = "USD"
    effective_date: str = ""
    source: str = ""


def estimate_cost(usage: Usage, price: Price) -> float | None:
    """Return None (unknown) rather than 0 when usage is missing."""
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    cached = usage.cached_input_tokens or 0
    if cached and price.cached_input_per_million is None:
        return None  # cached tokens were used but we have no rate for them
    cost = usage.input_tokens * price.input_per_million            # uncached input only
    cost += cached * (price.cached_input_per_million or 0.0)
    cost += usage.output_tokens * price.output_per_million
    return cost / 1_000_000


def total_cost(usages: list[Usage], price: Price) -> float | None:
    parts = [estimate_cost(u, price) for u in usages]
    return None if any(p is None for p in parts) else sum(parts)  # any unknown => unknown total
