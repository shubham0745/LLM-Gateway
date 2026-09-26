"""Production entry point: ``python -m gateway.serve``.

One Python process tops out at roughly one CPU core of stream relaying, so
the gateway runs ``GATEWAY_WORKERS`` processes behind one port (default: one
per CPU, at most 4, which keeps 4 x 20 Postgres connections under the
default limit of 100). All shared state already lives in Redis and Postgres; the only
per-process state is Prometheus metrics, which switch to multiprocess mode.
"""

from __future__ import annotations

import os
import shutil

import uvicorn


def main() -> None:
    workers = int(os.environ.get("GATEWAY_WORKERS") or min(os.cpu_count() or 1, 4))
    if workers > 1:
        # Must be set before prometheus_client is imported by the workers.
        prom_dir = os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", "/tmp/gateway-prometheus")
        shutil.rmtree(prom_dir, ignore_errors=True)
        os.makedirs(prom_dir, exist_ok=True)
    uvicorn.run(
        "gateway.main:app",
        host=os.environ.get("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.environ.get("GATEWAY_PORT", "8080")),
        workers=workers,
        loop="uvloop",
        http="httptools",
        access_log=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
