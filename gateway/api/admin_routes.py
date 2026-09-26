"""Admin API: tenants, keys, routing config (hot reload), usage and circuits.

Every endpoint requires ``Authorization: Bearer $GATEWAY_ADMIN_TOKEN``.
"""

from __future__ import annotations

import hmac
from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, ValidationError

from gateway.config import GatewayConfig
from gateway.errors import GatewayError
from gateway.services import Services

router = APIRouter(prefix="/admin", tags=["admin"])


def _svc(request: Request) -> Services:
    return request.app.state.services


async def require_admin(request: Request) -> None:
    token = _svc(request).settings.admin_token
    auth = request.headers.get("authorization") or ""
    supplied = auth[7:] if auth.lower().startswith("bearer ") else ""
    if not supplied or not hmac.compare_digest(supplied.encode(), token.encode()):
        raise GatewayError(401, "Admin token required.", "authentication_error", "invalid_admin_token")


admin = [Depends(require_admin)]


class TenantIn(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,64}$")
    name: str
    monthly_budget_usd: float | None = Field(default=None, ge=0)
    rpm_limit: int | None = Field(default=None, gt=0, le=2_147_483_647)
    tpm_limit: int | None = Field(default=None, gt=0, le=2_147_483_647)


class TenantPatch(BaseModel):
    name: str | None = None
    monthly_budget_usd: float | None = Field(default=None, ge=0)
    rpm_limit: int | None = Field(default=None, gt=0, le=2_147_483_647)
    tpm_limit: int | None = Field(default=None, gt=0, le=2_147_483_647)


class KeyIn(BaseModel):
    name: str = ""


# -- tenants and keys --------------------------------------------------------


@router.post("/tenants", dependencies=admin, status_code=201)
async def create_tenant(body: TenantIn, request: Request) -> dict:
    svc = _svc(request)
    t = await svc.keys.create_tenant(body.id, body.name, body.monthly_budget_usd, body.rpm_limit, body.tpm_limit)
    await svc.config_store.publish({"type": "auth"})
    return t


@router.get("/tenants", dependencies=admin)
async def list_tenants(request: Request) -> dict:
    svc = _svc(request)
    tenants = await svc.keys.list_tenants()
    for t in tenants:
        t["spent_this_month_usd"] = round(await svc.budgets.spent_usd(t["id"]), 6)
    return {"data": tenants}


@router.get("/tenants/{tenant_id}", dependencies=admin)
async def get_tenant(tenant_id: str, request: Request) -> dict:
    svc = _svc(request)
    t = await svc.keys.get_tenant(tenant_id)
    if t is None:
        raise GatewayError(404, "No such tenant.", code="not_found")
    t["spent_this_month_usd"] = round(await svc.budgets.spent_usd(tenant_id), 6)
    t["effective_limits"] = {
        "monthly_budget_usd": t["monthly_budget_usd"] if t["monthly_budget_usd"] is not None else svc.config.limits.monthly_budget_usd,
        "rpm": t["rpm_limit"] or svc.config.limits.rpm,
        "tpm": t["tpm_limit"] or svc.config.limits.tpm,
    }
    return t


@router.patch("/tenants/{tenant_id}", dependencies=admin)
async def update_tenant(tenant_id: str, body: TenantPatch, request: Request) -> dict:
    svc = _svc(request)
    t = await svc.keys.update_tenant(tenant_id, body.model_dump(exclude_unset=True))
    if t is None:
        raise GatewayError(404, "No such tenant.", code="not_found")
    await svc.config_store.publish({"type": "auth"})  # limits are cached with the key
    return t


@router.post("/tenants/{tenant_id}/keys", dependencies=admin, status_code=201)
async def create_key(tenant_id: str, body: KeyIn, request: Request) -> dict:
    svc = _svc(request)
    if await svc.keys.get_tenant(tenant_id) is None:
        raise GatewayError(404, "No such tenant.", code="not_found")
    key, meta = await svc.keys.create_key(tenant_id, body.name)
    return {**meta, "key": key, "note": "Store this key now; it cannot be shown again."}


@router.get("/tenants/{tenant_id}/keys", dependencies=admin)
async def list_keys(tenant_id: str, request: Request) -> dict:
    return {"data": await _svc(request).keys.list_keys(tenant_id)}


@router.delete("/keys/{key_id}", dependencies=admin)
async def revoke_key(key_id: int, request: Request) -> dict:
    svc = _svc(request)
    k = await svc.keys.revoke_key(key_id)
    if k is None:
        raise GatewayError(404, "No such key.", code="not_found")
    await svc.config_store.publish({"type": "auth"})
    return k


# -- routing config ----------------------------------------------------------


@router.get("/config", dependencies=admin)
async def get_config(request: Request) -> dict:
    svc = _svc(request)
    return {"version": svc.config_version, "config": svc.config.model_dump(mode="json")}


@router.put("/config", dependencies=admin)
async def put_config(request: Request, comment: str = Query(default="")) -> dict:
    """Validate and activate a new config version on every instance, no restart."""
    svc = _svc(request)
    try:
        raw: Any = await request.json()
        config = GatewayConfig.model_validate(raw.get("config", raw) if isinstance(raw, dict) else raw)
    except (ValueError, ValidationError) as exc:
        raise GatewayError(400, f"Invalid config: {exc}", code="invalid_config") from None
    version = await svc.config_store.save(config, comment)
    svc.apply_config(version, config)  # this instance immediately; others via pub/sub
    return {"version": version}


@router.get("/config/versions", dependencies=admin)
async def config_versions(request: Request) -> dict:
    return {"data": await _svc(request).config_store.versions()}


@router.post("/config/rollback/{version}", dependencies=admin)
async def rollback_config(version: int, request: Request) -> dict:
    svc = _svc(request)
    old = await svc.config_store.get(version)
    if old is None:
        raise GatewayError(404, "No such config version.", code="not_found")
    new_version = await svc.config_store.save(old, f"rollback to {version}")
    svc.apply_config(new_version, old)
    return {"version": new_version, "restored_from": version}


# -- usage, requests, circuits ----------------------------------------------

_GROUPS = {"tenant": "tenant_id", "model": "model", "provider": "provider", "alias": "alias", "day": "date_trunc('day', ts)::date", "cache": "cache_status"}


@router.get("/usage", dependencies=admin)
async def usage(
    request: Request,
    start: date | None = None,
    end: date | None = None,
    group_by: str = "tenant,model",
    tenant: str | None = None,
) -> dict:
    """Spend report. Defaults to yesterday (UTC): "who spent how much, on which model, yesterday?"."""
    today = datetime.now(UTC).date()
    start = start or today - timedelta(days=1)
    end = end or start + timedelta(days=1)
    keys = [g.strip() for g in group_by.split(",") if g.strip()]
    if not keys or any(k not in _GROUPS for k in keys):
        raise GatewayError(400, f"group_by must be a comma-separated subset of {sorted(_GROUPS)}", code="invalid_group_by")
    cols = ", ".join(f"{_GROUPS[k]} AS {k}" for k in keys)
    where = "ts >= $1 AND ts < $2" + (" AND tenant_id = $3" if tenant else "")
    args: list[Any] = [datetime.combine(start, datetime.min.time(), UTC), datetime.combine(end, datetime.min.time(), UTC)]
    if tenant:
        args.append(tenant)
    rows = await _svc(request).pool.fetch(
        f"""
        SELECT {cols}, count(*) AS requests,
               sum(prompt_tokens) AS prompt_tokens, sum(completion_tokens) AS completion_tokens,
               sum(cost_usd)::float AS cost_usd, sum(saved_usd)::float AS saved_usd,
               count(*) FILTER (WHERE cache_status LIKE 'hit%') AS cache_hits,
               count(*) FILTER (WHERE status = 'error') AS errors
        FROM request_logs WHERE {where}
        GROUP BY {", ".join(str(i + 1) for i in range(len(keys)))}
        ORDER BY cost_usd DESC NULLS LAST
        """,
        *args,
    )
    data = [{k: (v.isoformat() if isinstance(v, date) else v) for k, v in dict(r).items()} for r in rows]
    return {"start": start.isoformat(), "end": end.isoformat(), "group_by": keys, "data": data,
            "total_cost_usd": round(sum(d["cost_usd"] or 0 for d in data), 8)}


@router.get("/requests/{request_id}", dependencies=admin)
async def get_request(request_id: str, request: Request) -> dict:
    row = await _svc(request).pool.fetchrow("SELECT * FROM request_logs WHERE request_id = $1", request_id)
    if row is None:
        raise GatewayError(404, "No such request (logs are written asynchronously; retry in a moment).", code="not_found")
    out = dict(row)
    out["ts"] = out["ts"].isoformat()
    out["cost_usd"] = float(out["cost_usd"])
    out["saved_usd"] = float(out["saved_usd"])
    return out


@router.get("/circuits", dependencies=admin)
async def circuits(request: Request) -> dict:
    svc = _svc(request)
    return await svc.breakers.snapshot(sorted(svc.config.providers))


@router.post("/circuits/{provider}/reset", dependencies=admin)
async def reset_circuit(provider: str, request: Request) -> dict:
    svc = _svc(request)
    await svc.breakers.reset(provider)
    return (await svc.breakers.snapshot([provider]))[provider]
