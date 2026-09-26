"""FastAPI application factory. Run with ``uvicorn gateway.main:app``."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gateway import db
from gateway.api import openai_routes
from gateway.api.auth import KeyStore
from gateway.config import Settings, load_config_file
from gateway.errors import GatewayError
from gateway.providers.registry import ProviderRegistry
from gateway.routing.router import Router
from gateway.services import Services
from gateway.telemetry.emitter import Telemetry
from gateway.telemetry.logging import setup_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        pool = await db.create_pool(settings.database_url)
        await db.migrate(pool)
        config = load_config_file(settings.config_path)
        registry = ProviderRegistry()
        registry.apply(config)
        services = Services(
            settings=settings,
            config=config,
            pool=pool,
            keys=KeyStore(pool, settings.key_pepper, settings.auth_cache_ttl_s),
            registry=registry,
            router=Router(registry, config),
            telemetry=Telemetry(),
        )
        app.state.services = services
        try:
            yield
        finally:
            await registry.aclose()
            await pool.close()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return exc.response()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    app.include_router(openai_routes.router)
    return app


app = create_app()
