"""Monthly per-tenant budgets with a hard stop.

Spend for the current month is an integer micro-dollar counter in Redis, so
checking it costs one round trip. A request reserves its worst-case cost
before it is sent (price of the most expensive model in its chain times
prompt estimate plus max completion), and settles to the real cost when it
finishes. Reservations are what make the stop "hard": fifty concurrent
requests cannot each see $0.01 of headroom and together overspend by $0.50.

Postgres (request_logs) is the durable record. If Redis loses the counter,
the first request of the month for that tenant rebuilds it from Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg
from redis.asyncio import Redis

from gateway.accounting.pricing import MICRO

# KEYS[1]=spend counter  ARGV: reserve_micro, budget_micro (-1 = unlimited), ttl_s
_RESERVE = """
local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
local reserve = tonumber(ARGV[1])
local budget = tonumber(ARGV[2])
if budget >= 0 and spent + reserve > budget then
  return {0, spent}
end
local now = redis.call('INCRBY', KEYS[1], reserve)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return {1, now}
"""

MONTH_TTL_S = 40 * 86_400


def month_key(tenant: str, when: datetime | None = None) -> str:
    when = when or datetime.now(UTC)
    return f"gw:budget:{tenant}:{when:%Y%m}"


class BudgetTracker:
    def __init__(self, redis: Redis, pool: asyncpg.Pool):
        self.redis = redis
        self.pool = pool
        self._reserve = redis.register_script(_RESERVE)
        self._loaded: set[str] = set()

    async def _ensure_loaded(self, tenant: str, key: str) -> None:
        if key in self._loaded:
            return
        if not await self.redis.exists(key):
            now = datetime.now(UTC)
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            spent = await self.pool.fetchval(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM request_logs WHERE tenant_id = $1 AND ts >= $2", tenant, start
            )
            # NX: if another instance rebuilt it first, keep theirs.
            await self.redis.set(key, int(round(float(spent) * MICRO)), ex=MONTH_TTL_S, nx=True)
        self._loaded.add(key)

    async def reserve(self, tenant: str, budget_usd: float | None, reserve_micro: int) -> tuple[bool, float]:
        key = month_key(tenant)
        await self._ensure_loaded(tenant, key)
        budget_micro = -1 if budget_usd is None else int(round(budget_usd * MICRO))
        ok, spent = await self._reserve(keys=[key], args=[reserve_micro, budget_micro, MONTH_TTL_S])
        return bool(ok), int(spent) / MICRO

    async def settle(self, tenant: str, reserved_micro: int, actual_micro: int) -> None:
        delta = actual_micro - reserved_micro
        if delta:
            await self.redis.incrby(month_key(tenant), delta)

    async def spent_usd(self, tenant: str) -> float:
        key = month_key(tenant)
        await self._ensure_loaded(tenant, key)
        return int(await self.redis.get(key) or 0) / MICRO
