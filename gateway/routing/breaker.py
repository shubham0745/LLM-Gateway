"""Per-provider circuit breakers stored in Redis.

State lives in Redis (not in process memory) so every gateway instance agrees
on when a provider is down: one instance's failures protect all of them, and
one successful probe closes the circuit everywhere.

    closed --(failure_threshold consecutive failures)--> open
    open --(cooldown elapsed)--> half_open  (lets max_probes requests through)
    half_open --probe succeeds--> closed
    half_open --probe fails--> open (cooldown restarts)

Each transition is a single Lua script, so concurrent requests on many
instances cannot race each other into an inconsistent state. A half-open probe
that never reports back (its instance crashed) expires after one cooldown, so
the circuit cannot get stuck half-open.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis

from gateway.config import BreakerConfig

KEY = "gw:cb:{}"

# KEYS[1]=hash  ARGV: now_ms, cooldown_ms, max_probes
# Returns {allowed(0/1), state}
_ACQUIRE = """
local st = redis.call('HGET', KEYS[1], 'state') or 'closed'
local now = tonumber(ARGV[1])
local cooldown = tonumber(ARGV[2])
local max_probes = tonumber(ARGV[3])
if st == 'closed' then
  return {1, 'closed'}
end
if st == 'open' then
  local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
  if now - opened >= cooldown then
    redis.call('HSET', KEYS[1], 'state', 'half_open', 'probes', 1, 'half_open_at', now)
    return {1, 'half_open'}
  end
  return {0, 'open'}
end
-- half_open
local probes = tonumber(redis.call('HGET', KEYS[1], 'probes') or '0')
local since = tonumber(redis.call('HGET', KEYS[1], 'half_open_at') or '0')
if probes < max_probes or now - since >= cooldown then
  if now - since >= cooldown then probes = 0; redis.call('HSET', KEYS[1], 'half_open_at', now) end
  redis.call('HSET', KEYS[1], 'probes', probes + 1)
  return {1, 'half_open'}
end
return {0, 'half_open'}
"""

# KEYS[1]=hash  ARGV: outcome ('success'|'failure'|'release'), now_ms, threshold
# Returns {state_after, transitioned(0/1)}
_RECORD = """
local st = redis.call('HGET', KEYS[1], 'state') or 'closed'
local outcome = ARGV[1]
local now = tonumber(ARGV[2])
local threshold = tonumber(ARGV[3])
if outcome == 'release' then
  if st == 'half_open' then
    local p = tonumber(redis.call('HGET', KEYS[1], 'probes') or '0')
    if p > 0 then redis.call('HSET', KEYS[1], 'probes', p - 1) end
  end
  return {st, 0}
end
if outcome == 'success' then
  if st ~= 'closed' then
    redis.call('HSET', KEYS[1], 'state', 'closed', 'failures', 0, 'probes', 0)
    redis.call('HINCRBY', KEYS[1], 'transitions', 1)
    return {'closed', 1}
  end
  if redis.call('HGET', KEYS[1], 'failures') ~= '0' then
    redis.call('HSET', KEYS[1], 'failures', 0)
  end
  return {'closed', 0}
end
-- failure
local failures = redis.call('HINCRBY', KEYS[1], 'failures', 1)
if st == 'half_open' or (st == 'closed' and failures >= threshold) then
  redis.call('HSET', KEYS[1], 'state', 'open', 'opened_at', now, 'probes', 0)
  redis.call('HINCRBY', KEYS[1], 'transitions', 1)
  return {'open', 1}
end
return {st, 0}
"""

STATE_VALUE = {"closed": 0, "half_open": 1, "open": 2}


@dataclass
class BreakerDecision:
    allowed: bool
    state: str


class CircuitBreakers:
    def __init__(self, redis: Redis, cfg: BreakerConfig):
        self.redis = redis
        self.cfg = cfg
        self._acquire = redis.register_script(_ACQUIRE)
        self._record = redis.register_script(_RECORD)
        # Last state seen per provider, for metrics and the admin API.
        self.observed: dict[str, str] = {}

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    async def acquire(self, provider: str) -> BreakerDecision:
        allowed, state = await self._acquire(
            keys=[KEY.format(provider)],
            args=[self._now_ms(), int(self.cfg.cooldown_s * 1000), self.cfg.half_open_max_probes],
        )
        state = state.decode() if isinstance(state, bytes) else state
        self.observed[provider] = state
        return BreakerDecision(bool(allowed), state)

    async def _record_outcome(self, provider: str, outcome: str) -> tuple[str, bool]:
        state, changed = await self._record(
            keys=[KEY.format(provider)], args=[outcome, self._now_ms(), self.cfg.failure_threshold]
        )
        state = state.decode() if isinstance(state, bytes) else state
        self.observed[provider] = state
        return state, bool(changed)

    async def success(self, provider: str) -> tuple[str, bool]:
        return await self._record_outcome(provider, "success")

    async def failure(self, provider: str) -> tuple[str, bool]:
        return await self._record_outcome(provider, "failure")

    async def release(self, provider: str) -> None:
        """The attempt ended without telling us anything about provider health."""
        await self._record_outcome(provider, "release")

    async def snapshot(self, providers: list[str]) -> dict[str, dict]:
        out = {}
        for p in providers:
            h = await self.redis.hgetall(KEY.format(p))
            h = {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in h.items()}
            out[p] = {
                "state": h.get("state", "closed"),
                "consecutive_failures": int(h.get("failures", 0)),
                "opened_at_ms": int(h["opened_at"]) if "opened_at" in h else None,
                "transitions": int(h.get("transitions", 0)),
            }
            self.observed[p] = out[p]["state"]
        return out

    async def reset(self, provider: str) -> None:
        await self.redis.delete(KEY.format(provider))
        self.observed[provider] = "closed"
