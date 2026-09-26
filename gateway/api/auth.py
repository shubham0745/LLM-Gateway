"""API key issuing and verification.

Keys look like ``gw-<43 random url-safe chars>``. Only an HMAC-SHA256 of the key
(keyed with a server-side pepper) is stored. A slow password hash such as bcrypt
is designed for low-entropy human passwords; these keys carry 256 bits of
randomness, so brute force is already infeasible and a fast keyed hash keeps
auth off the latency budget (bcrypt would add ~100 ms to every request). A
leaked database alone is useless without the pepper.

Verified keys are cached in-process for a short TTL. Revocation publishes an
invalidation so every instance drops the key immediately; the TTL is the
backstop if that message is missed.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

import asyncpg

from gateway.errors import GatewayError

KEY_PREFIX = "gw-"


@dataclass(frozen=True)
class Principal:
    key_id: int
    tenant_id: str
    tenant_name: str
    monthly_budget_usd: float | None
    rpm_limit: int | None
    tpm_limit: int | None


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str, pepper: str) -> str:
    return hmac.new(pepper.encode(), key.encode(), hashlib.sha256).hexdigest()


class KeyStore:
    def __init__(self, pool: asyncpg.Pool, pepper: str, ttl_s: float = 30.0):
        self.pool = pool
        self.pepper = pepper
        self.ttl_s = ttl_s
        self._cache: dict[str, tuple[float, Principal | None]] = {}

    async def authenticate(self, authorization: str | None) -> Principal:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise GatewayError(401, "Missing API key. Send 'Authorization: Bearer <key>'.", "authentication_error", "invalid_api_key")
        key = authorization[7:].strip()
        if not key.startswith(KEY_PREFIX):
            raise GatewayError(401, "Invalid API key.", "authentication_error", "invalid_api_key")
        digest = hash_key(key, self.pepper)
        now = time.monotonic()
        cached = self._cache.get(digest)
        if cached and cached[0] > now:
            principal = cached[1]
        else:
            principal = await self._lookup(digest)
            # Negative results are cached too, so a flood of bad keys can't hammer Postgres.
            self._cache[digest] = (now + self.ttl_s, principal)
            if len(self._cache) > 100_000:
                self._cache.clear()
        if principal is None:
            raise GatewayError(401, "Invalid API key.", "authentication_error", "invalid_api_key")
        return principal

    async def _lookup(self, digest: str) -> Principal | None:
        row = await self.pool.fetchrow(
            """
            SELECT k.id, k.tenant_id, t.name, t.monthly_budget_usd, t.rpm_limit, t.tpm_limit
            FROM api_keys k JOIN tenants t ON t.id = k.tenant_id
            WHERE k.key_hash = $1 AND k.revoked_at IS NULL
            """,
            digest,
        )
        if row is None:
            return None
        return Principal(
            key_id=row["id"],
            tenant_id=row["tenant_id"],
            tenant_name=row["name"],
            monthly_budget_usd=float(row["monthly_budget_usd"]) if row["monthly_budget_usd"] is not None else None,
            rpm_limit=row["rpm_limit"],
            tpm_limit=row["tpm_limit"],
        )

    def invalidate_all(self) -> None:
        self._cache.clear()

    # -- management (used by the admin API and CLI) ------------------------

    async def create_tenant(
        self,
        tenant_id: str,
        name: str,
        monthly_budget_usd: float | None = None,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
    ) -> dict:
        row = await self.pool.fetchrow(
            """
            INSERT INTO tenants (id, name, monthly_budget_usd, rpm_limit, tpm_limit)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name,
                monthly_budget_usd = EXCLUDED.monthly_budget_usd,
                rpm_limit = EXCLUDED.rpm_limit, tpm_limit = EXCLUDED.tpm_limit
            RETURNING *
            """,
            tenant_id,
            name,
            monthly_budget_usd,
            rpm_limit,
            tpm_limit,
        )
        return _tenant_dict(row)

    async def create_key(self, tenant_id: str, name: str = "") -> tuple[str, dict]:
        key = generate_key()
        row = await self.pool.fetchrow(
            """
            INSERT INTO api_keys (tenant_id, name, key_prefix, key_hash)
            VALUES ($1, $2, $3, $4)
            RETURNING id, tenant_id, name, key_prefix, created_at, revoked_at
            """,
            tenant_id,
            name,
            key[:10],
            hash_key(key, self.pepper),
        )
        return key, _key_dict(row)


def _tenant_dict(row: asyncpg.Record) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "monthly_budget_usd": float(row["monthly_budget_usd"]) if row["monthly_budget_usd"] is not None else None,
        "rpm_limit": row["rpm_limit"],
        "tpm_limit": row["tpm_limit"],
        "created_at": row["created_at"].isoformat(),
    }


def _key_dict(row: asyncpg.Record) -> dict:
    return {
        "id": row["id"],
        "tenant_id": row["tenant_id"],
        "name": row["name"],
        "key_prefix": row["key_prefix"],
        "created_at": row["created_at"].isoformat(),
        "revoked_at": row["revoked_at"].isoformat() if row["revoked_at"] else None,
    }
