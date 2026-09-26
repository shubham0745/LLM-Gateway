"""Turns token counts into money."""

from __future__ import annotations

from gateway.config import GatewayConfig, Price, Target

MICRO = 1_000_000  # budgets are tracked in integer micro-dollars in Redis


def cost_usd(pricing: dict[str, Price], model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    price = pricing.get(model or "")
    if price is None:
        return 0.0
    return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000


def worst_case_price(config: GatewayConfig, chain: list[Target]) -> Price:
    """The most expensive target in a chain; used to size budget reservations."""
    prices = [config.pricing[t.model] for t in chain if t.model in config.pricing]
    if not prices:
        return Price(input=0.0, output=0.0)
    return Price(input=max(p.input for p in prices), output=max(p.output for p in prices))


def to_micro(usd: float) -> int:
    return int(round(usd * MICRO))
