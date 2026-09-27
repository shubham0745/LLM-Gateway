"""App factory.

Run with::

    GATEWAY_CONFIG=config/gateway.yaml uvicorn gateway.main:create_app --factory
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from gateway.api import router
from gateway.config import GatewayConfig, load_config
from gateway.errors import GatewayError, gateway_error_handler, validation_error_handler
from gateway.providers.registry import build_adapters
from gateway.request_id import RequestIDMiddleware

DEFAULT_CONFIG_PATH = "config/gateway.yaml"

# One pooled client for all providers; httpx keeps a pool per host.
POOL_LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=50, keepalive_expiry=30)


def create_app(
    config: GatewayConfig | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the gateway.

    ``transport`` replaces the network for provider calls; tests pass an
    ASGI transport wrapping the mock provider.
    """
    if config is None:
        config = load_config(os.environ.get("GATEWAY_CONFIG", DEFAULT_CONFIG_PATH))
    adapters = build_adapters(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(limits=POOL_LIMITS, transport=transport) as client:
            app.state.http_client = client
            yield

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.adapters = adapters
    app.include_router(router)
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_middleware(RequestIDMiddleware)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return app
