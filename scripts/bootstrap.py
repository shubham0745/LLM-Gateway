"""Create the tenants and keys the demo and benchmarks use.

    python scripts/bootstrap.py [--gateway http://localhost:8080] [--admin-token ...]

Writes the keys to .gateway-keys.json (git-ignored). Safe to re-run: tenants
are upserted and a fresh key is issued each time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

TENANTS = [
    # id, name, monthly budget (USD), rpm, tpm
    ("demo", "Public demo key", 2.00, 30, 60_000),
    ("bench", "Benchmarks", 1_000_000.0, 100_000_000, 2_000_000_000),
    ("acme", "Example tenant: Acme", 25.00, 600, 1_000_000),
    ("globex", "Example tenant: Globex", 10.00, 600, 1_000_000),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.environ.get("GATEWAY_URL", "http://localhost:8080"))
    ap.add_argument("--admin-token", default=os.environ.get("GATEWAY_ADMIN_TOKEN", "change-me-admin-token"))
    args = ap.parse_args()
    h = {"authorization": f"Bearer {args.admin_token}"}
    keys = {}
    with httpx.Client(base_url=args.gateway, headers=h, timeout=10) as c:
        for tid, name, budget, rpm, tpm in TENANTS:
            c.post("/admin/tenants", json={"id": tid, "name": name, "monthly_budget_usd": budget, "rpm_limit": rpm, "tpm_limit": tpm}).raise_for_status()
            r = c.post(f"/admin/tenants/{tid}/keys", json={"name": "bootstrap"})
            r.raise_for_status()
            keys[tid] = r.json()["key"]
    out = Path(".gateway-keys.json")
    out.write_text(json.dumps(keys, indent=2) + "\n")
    print(f"created {len(keys)} keys -> {out}")
    print(f"try: curl {args.gateway}/v1/chat/completions -H 'authorization: Bearer {keys['demo']}' "
          "-H 'content-type: application/json' -d '{\"model\": \"mock\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}'")


if __name__ == "__main__":
    main()
