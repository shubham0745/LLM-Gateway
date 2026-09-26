CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_cache (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    scope_hash  TEXT NOT NULL,          -- alias + system prompt + sampling params, must match exactly
    prompt      TEXT NOT NULL,
    embedding   vector(384) NOT NULL,   -- all-MiniLM-L6-v2, L2-normalised
    response    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx ON semantic_cache USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS semantic_cache_scope_idx ON semantic_cache (tenant_id, scope_hash);
CREATE INDEX IF NOT EXISTS semantic_cache_expiry_idx ON semantic_cache (expires_at);
