"""Command-line helpers.

    python -m gateway.cli create-tenant acme "Acme Inc" --budget 25
    python -m gateway.cli create-key acme --name laptop
    python -m gateway.cli load-config deploy/config/gateway.yaml   # new config version, hot-reloaded
"""

from __future__ import annotations

import argparse
import asyncio
import json

from gateway import db
from gateway.api.auth import KeyStore
from gateway.config import Settings


async def _run(args: argparse.Namespace) -> None:
    settings = Settings()
    pool = await db.create_pool(settings.database_url, min_size=1, max_size=2)
    try:
        await db.migrate(pool)
        store = KeyStore(pool, settings.key_pepper)
        if args.cmd == "migrate":
            print("migrations applied")
        elif args.cmd == "create-tenant":
            t = await store.create_tenant(args.tenant_id, args.name, args.budget, args.rpm, args.tpm)
            print(json.dumps(t, indent=2))
        elif args.cmd == "load-config":
            from redis.asyncio import Redis

            from gateway.config import load_config_file
            from gateway.routing.config_store import ConfigStore

            redis = Redis.from_url(settings.redis_url)
            try:
                version = await ConfigStore(pool, redis).save(load_config_file(args.path), f"loaded from {args.path}")
            finally:
                await redis.aclose()
            print(f"config version {version} saved; running gateways reload it now")
        elif args.cmd == "create-key":
            key, meta = await store.create_key(args.tenant_id, args.name)
            print(json.dumps({**meta, "key": key}, indent=2))
            print("\nStore this key now; it cannot be shown again.")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="gateway")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    t = sub.add_parser("create-tenant")
    t.add_argument("tenant_id")
    t.add_argument("name")
    t.add_argument("--budget", type=float, default=None, help="monthly budget in USD")
    t.add_argument("--rpm", type=int, default=None)
    t.add_argument("--tpm", type=int, default=None)
    lc = sub.add_parser("load-config")
    lc.add_argument("path")
    k = sub.add_parser("create-key")
    k.add_argument("tenant_id")
    k.add_argument("--name", default="")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
