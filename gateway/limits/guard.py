"""Admission control: rate limits and budget, reserved before and settled after."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gateway.accounting.pricing import to_micro, worst_case_price
from gateway.accounting.tokens import estimate_prompt_tokens, max_completion_tokens
from gateway.api.auth import Principal
from gateway.config import GatewayConfig, Target
from gateway.errors import GatewayError
from gateway.limits.budget import BudgetTracker
from gateway.limits.ratelimit import RateLimiter
from gateway.telemetry.metrics import Metrics


@dataclass
class Reservation:
    tenant: str
    tpm: int
    tokens: int
    cost_micro: int
    budget_usd: float | None
    headers: dict[str, str]
    settled: bool = False


class LimitGuard:
    def __init__(self, limiter: RateLimiter, budgets: BudgetTracker, metrics: Metrics):
        self.limiter = limiter
        self.budgets = budgets
        self.metrics = metrics

    async def admit(self, principal: Principal, body: dict, chain: list[Target], config: GatewayConfig) -> Reservation:
        d = config.limits
        rpm = principal.rpm_limit or d.rpm
        tpm = principal.tpm_limit or d.tpm
        budget = principal.monthly_budget_usd if principal.monthly_budget_usd is not None else d.monthly_budget_usd
        tenant = principal.tenant_id

        prompt_est = estimate_prompt_tokens(body)
        completion_est = max_completion_tokens(body, d.default_completion_estimate)
        tokens = prompt_est + completion_est

        rate = await self.limiter.take(tenant, rpm, tpm, tokens)
        headers = {
            "x-ratelimit-limit-requests": str(rpm),
            "x-ratelimit-remaining-requests": str(max(0, rate.remaining_requests)),
            "x-ratelimit-limit-tokens": str(tpm),
            "x-ratelimit-remaining-tokens": str(max(0, rate.remaining_tokens)),
        }
        if not rate.allowed:
            if rate.retry_after_s is None:
                self.metrics.rejected.labels(tenant, "request_too_large").inc()
                raise GatewayError(
                    429, f"This request needs ~{tokens} tokens, more than your limit of {tpm} tokens per minute.",
                    "rate_limit_error", "request_too_large", headers,
                )
            self.metrics.rejected.labels(tenant, "rate_limit").inc()
            retry = max(1, math.ceil(rate.retry_after_s))
            raise GatewayError(
                429, f"Rate limit reached. Retry in {retry}s.", "rate_limit_error", "rate_limit_exceeded",
                {**headers, "retry-after": str(retry)},
            )

        price = worst_case_price(config, chain)
        reserve = to_micro((prompt_est * price.input + completion_est * price.output) / 1_000_000)
        ok, spent = await self.budgets.reserve(tenant, budget, reserve)
        if budget is not None:
            self.metrics.budget_limit.labels(tenant).set(budget)
        self.metrics.budget_spent.labels(tenant).set(spent)
        if not ok:
            # Give back the rate-limit tokens: this request is not going anywhere.
            await self.limiter.settle_tokens(tenant, tpm, tokens, 0)
            self.metrics.rejected.labels(tenant, "budget").inc()
            raise GatewayError(
                429, f"Monthly budget of ${budget:.2f} reached (spent ${spent:.4f}).",
                "insufficient_quota", "budget_exceeded", headers,
            )
        return Reservation(tenant, tpm, tokens, reserve, budget, headers)

    async def settle(self, res: Reservation, actual_tokens: int, actual_cost_usd: float) -> None:
        if res.settled:
            return
        res.settled = True
        await self.limiter.settle_tokens(res.tenant, res.tpm, res.tokens, actual_tokens)
        await self.budgets.settle(res.tenant, res.cost_micro, to_micro(actual_cost_usd))
