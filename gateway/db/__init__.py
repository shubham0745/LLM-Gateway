"""Postgres access: a shared asyncpg pool and a tiny forward-only migration runner."""

from __future__ import annotations

import json
from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def _init_conn(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    try:
        # pgvector >= 0.8: keep scanning the HNSW graph until the tenant/scope
        # filter has found a match, instead of returning nothing when the
        # nearest neighbours overall belong to other tenants.
        await conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except asyncpg.PostgresError:
        pass


async def create_pool(dsn: str, min_size: int = 2, max_size: int = 20) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=_init_conn)


async def migrate(pool: asyncpg.Pool) -> list[str]:
    """Apply any migrations not yet recorded. Safe to run from several processes."""
    applied: list[str] = []
    async with pool.acquire() as conn:
        # Serialize concurrent starters (gateway replicas + worker) on one lock.
        await conn.execute("SELECT pg_advisory_lock(727274)")
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
            )
            done = {r["name"] for r in await conn.fetch("SELECT name FROM schema_migrations")}
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if path.name in done:
                    continue
                async with conn.transaction():
                    await conn.execute(path.read_text())
                    await conn.execute("INSERT INTO schema_migrations (name) VALUES ($1)", path.name)
                applied.append(path.name)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(727274)")
    return applied
