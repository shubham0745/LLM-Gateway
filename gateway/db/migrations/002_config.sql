CREATE TABLE IF NOT EXISTS gateway_config (
    version     BIGSERIAL PRIMARY KEY,
    document    JSONB NOT NULL,
    comment     TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
