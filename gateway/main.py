"""FastAPI application factory. Run with ``uvicorn gateway.main:app``."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import BlockingConnectionPool, Redis

from gateway import db
from gateway.api import admin_routes, openai_routes
from gateway.api.auth import KeyStore
from gateway.config import Settings, load_config_file
from gateway.errors import GatewayError
from gateway.limits.budget import BudgetTracker
from gateway.limits.guard import LimitGuard
from gateway.limits.ratelimit import RateLimiter
from gateway.providers.registry import ProviderRegistry
from gateway.routing.breaker import CircuitBreakers
from gateway.routing.config_store import ConfigStore, ControlListener
from gateway.routing.router import Router
from gateway.services import Services
from gateway.telemetry.emitter import Telemetry
from gateway.telemetry.logging import setup_logging
from gateway.telemetry.metrics import Metrics


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        pool = await db.create_pool(settings.database_url)
        await db.migrate(pool)
        # Blocking pool: under a burst, wait briefly for a connection instead of erroring.
        redis = Redis(connection_pool=BlockingConnectionPool.from_url(settings.redis_url, max_connections=512, timeout=5))

        store = ConfigStore(pool, redis)
        await store.seed_if_empty(load_config_file(settings.config_path))
        version, config = await store.latest()  # type: ignore[misc]

        metrics = Metrics()
        registry = ProviderRegistry()
        breakers = CircuitBreakers(redis, config.breaker)
        telemetry = Telemetry(metrics, redis, settings.events_stream, settings.events_stream_maxlen)
        budgets = BudgetTracker(redis, pool)
        services = Services(
            settings=settings,
            config=config,
            config_version=version,
            pool=pool,
            redis=redis,
            breakers=breakers,
            keys=KeyStore(pool, settings.key_pepper, settings.auth_cache_ttl_s),
            registry=registry,
            router=Router(registry, config, breakers, hooks=metrics),
            telemetry=telemetry,
            limits=LimitGuard(RateLimiter(redis), budgets, metrics),
            budgets=budgets,
            config_store=store,
        )
        services.apply_config(version, config)
        app.state.services = services
        telemetry.start()

        listener = ControlListener(
            store,
            redis,
            current_version=lambda: services.config_version,
            on_config=services.apply_config,
            on_auth_invalidate=services.keys.invalidate_all,
            poll_interval_s=settings.config_poll_interval_s,
        )
        listener.start()
        try:
            yield
        finally:
            await listener.stop()
            await services.drain()
            await telemetry.stop()
            await registry.aclose()
            await redis.aclose()
            await pool.close()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return exc.response()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        svc: Services = request.app.state.services
        try:
            await svc.redis.ping()
            await svc.pool.fetchval("SELECT 1")
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"status": "unavailable", "error": str(exc)}, status_code=503)
        return JSONResponse({"status": "ready", "config_version": svc.config_version})

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:
        svc: Services = request.app.state.services
        return Response(generate_latest(svc.telemetry.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.include_router(openai_routes.router)
    app.include_router(admin_routes.router)
    return app


app = create_app()
