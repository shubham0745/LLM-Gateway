CREATE TABLE IF NOT EXISTS tenants (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    monthly_budget_usd  NUMERIC(12, 4),          -- NULL = use the config default
    rpm_limit           INTEGER,                 -- NULL = use the config default
    tpm_limit           INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name         TEXT NOT NULL DEFAULT '',
    key_prefix   TEXT NOT NULL,                  -- first characters, for display only
    key_hash     TEXT NOT NULL UNIQUE,           -- HMAC-SHA256(pepper, key); the key itself is never stored
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS api_keys_tenant_idx ON api_keys (tenant_id);

CREATE TABLE IF NOT EXISTS request_logs (
    request_id         TEXT PRIMARY KEY,
    ts                 TIMESTAMPTZ NOT NULL,
    tenant_id          TEXT,
    key_id             BIGINT,
    alias              TEXT,
    provider           TEXT,
    model              TEXT,
    stream             BOOLEAN NOT NULL DEFAULT false,
    status             TEXT NOT NULL,             -- ok | error | client_disconnected | rejected
    http_status        INTEGER,
    error              TEXT,
    cache_status       TEXT NOT NULL DEFAULT 'miss',
    attempts           JSONB NOT NULL DEFAULT '[]',
    ttft_ms            DOUBLE PRECISION,
    latency_ms         DOUBLE PRECISION,
    prompt_tokens      INTEGER NOT NULL DEFAULT 0,
    completion_tokens  INTEGER NOT NULL DEFAULT 0,
    usage_estimated    BOOLEAN NOT NULL DEFAULT false,
    cost_usd           NUMERIC(14, 8) NOT NULL DEFAULT 0,
    saved_usd          NUMERIC(14, 8) NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS request_logs_ts_idx ON request_logs (ts);
CREATE INDEX IF NOT EXISTS request_logs_tenant_ts_idx ON request_logs (tenant_id, ts);
