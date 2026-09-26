"""Request and token rate limits as Redis token buckets (Lua, so atomic).

Each tenant has two buckets that refill continuously:
  * requests per minute (rpm): every request costs 1
  * tokens per minute (tpm): a request costs prompt + expected completion tokens

Token limits have a catch: the completion length is unknown until the
response ends. So we reserve an estimate up front (prompt estimate plus the
caller's max_tokens, or a default) and settle the difference afterwards. When
the real usage is higher than the reservation the bucket goes into debt, which
naturally delays the tenant's next requests; when lower, the unused tokens are
refunded.

Both buckets are checked and charged in one script: a request rejected for
tokens does not also burn a request slot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis

_TAKE = """
local now = tonumber(ARGV[1])
local result = {}
local ok = 1
local retry_ms = 0
local levels = {}
for i = 1, 2 do
  local cap = tonumber(ARGV[2 + (i - 1) * 3])
  local rate = tonumber(ARGV[3 + (i - 1) * 3])   -- tokens per ms
  local cost = tonumber(ARGV[4 + (i - 1) * 3])
  local d = redis.call('HMGET', KEYS[i], 'tokens', 'ts')
  local tokens = tonumber(d[1])
  local ts = tonumber(d[2])
  if tokens == nil then tokens = cap; ts = now end
  tokens = math.min(cap, tokens + math.max(0, now - ts) * rate)
  levels[i] = tokens
  if cost > cap then
    ok = 0
    retry_ms = -1
  elseif tokens < cost then
    ok = 0
    local wait = math.ceil((cost - tokens) / rate)
    if retry_ms >= 0 and wait > retry_ms then retry_ms = wait end
  end
end
for i = 1, 2 do
  local cap = tonumber(ARGV[2 + (i - 1) * 3])
  local rate = tonumber(ARGV[3 + (i - 1) * 3])
  local cost = tonumber(ARGV[4 + (i - 1) * 3])
  local tokens = levels[i]
  if ok == 1 then tokens = tokens - cost end
  redis.call('HSET', KEYS[i], 'tokens', tostring(tokens), 'ts', now)
  redis.call('PEXPIRE', KEYS[i], math.ceil(cap / rate) + 60000)
  levels[i] = tokens
end
return {ok, retry_ms, tostring(levels[1]), tostring(levels[2])}
"""

# Return (actual - reserved) tokens: negative delta refunds, positive adds debt.
_SETTLE = """
local d = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
if d[1] == false then return 0 end
local cap = tonumber(ARGV[1])
local tokens = math.min(cap, tonumber(d[1]) - tonumber(ARGV[2]))
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens))
return 1
"""


@dataclass
class RateDecision:
    allowed: bool
    retry_after_s: float | None  # None when the request can never fit
    remaining_requests: int
    remaining_tokens: int
    rpm: int
    tpm: int


class RateLimiter:
    def __init__(self, redis: Redis):
        self.redis = redis
        self._take = redis.register_script(_TAKE)
        self._settle = redis.register_script(_SETTLE)

    @staticmethod
    def _keys(tenant: str) -> list[str]:
        # Hash tag keeps both buckets in one slot if this ever runs on Redis Cluster.
        return [f"gw:rl:{{{tenant}}}:req", f"gw:rl:{{{tenant}}}:tok"]

    async def take(self, tenant: str, rpm: int, tpm: int, token_cost: int) -> RateDecision:
        now = int(time.time() * 1000)
        ok, retry_ms, req_left, tok_left = await self._take(
            keys=self._keys(tenant),
            args=[now, rpm, rpm / 60_000, 1, tpm, tpm / 60_000, token_cost],
        )
        retry = None if int(retry_ms) < 0 else int(retry_ms) / 1000
        return RateDecision(bool(ok), retry, int(float(req_left)), int(float(tok_left)), rpm, tpm)

    async def settle_tokens(self, tenant: str, tpm: int, reserved: int, actual: int) -> None:
        if actual == reserved:
            return
        await self._settle(keys=[self._keys(tenant)[1]], args=[tpm, actual - reserved])
