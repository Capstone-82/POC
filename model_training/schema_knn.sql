-- Phase 1: quality flags on benchmark_results.
ALTER TABLE benchmark_results
    ADD COLUMN IF NOT EXISTS score_stdev FLOAT,
    ADD COLUMN IF NOT EXISTS eval_conflict_flag BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS high_conflict_flag BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS low_confidence BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS confidence_level FLOAT,
    ADD COLUMN IF NOT EXISTS eval_count INTEGER,
    ADD COLUMN IF NOT EXISTS prompt_hash TEXT,
    ADD COLUMN IF NOT EXISTS invalid BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS syntax_pass BOOLEAN,
    ADD COLUMN IF NOT EXISTS syntax_checked BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS consistency_score FLOAT,
    ADD COLUMN IF NOT EXISTS win_rate FLOAT,
    ADD COLUMN IF NOT EXISTS domain TEXT,
    ADD COLUMN IF NOT EXISTS has_ref_answer BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS reference_answer TEXT,
    ADD COLUMN IF NOT EXISTS is_correct BOOLEAN,
    ADD COLUMN IF NOT EXISTS prompt_version INTEGER DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_br_prompt_hash
    ON benchmark_results (prompt_hash);

CREATE INDEX IF NOT EXISTS idx_br_low_confidence
    ON benchmark_results (low_confidence, eval_conflict_flag)
    WHERE avg_accuracy_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_br_use_case
    ON benchmark_results (use_case);

-- Phase 2: pgvector prompt embedding cache.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS prompt_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash TEXT UNIQUE NOT NULL,
    embedding vector(384),
    model_name TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pe_prompt_hash
    ON prompt_embeddings (prompt_hash);

CREATE INDEX IF NOT EXISTS idx_pe_embedding_hnsw
    ON prompt_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE OR REPLACE FUNCTION knn_search(
    query_embedding vector(384),
    target_use_case TEXT,
    result_limit INT DEFAULT 20,
    min_similarity FLOAT DEFAULT 0.72
)
RETURNS TABLE (
    row_id UUID,
    model_id TEXT,
    provider TEXT,
    avg_accuracy_score FLOAT,
    cost FLOAT,
    latency_ms FLOAT,
    similarity FLOAT,
    eval_conflict_flag BOOLEAN,
    low_confidence BOOLEAN
)
LANGUAGE SQL STABLE AS $$
    SELECT
        br.id AS row_id,
        br.model_id,
        br.provider,
        br.avg_accuracy_score,
        br.cost,
        br.latency_ms,
        1 - (pe.embedding <=> query_embedding) AS similarity,
        COALESCE(br.eval_conflict_flag, FALSE) AS eval_conflict_flag,
        COALESCE(br.low_confidence, FALSE) AS low_confidence
    FROM prompt_embeddings pe
    JOIN benchmark_results br ON pe.prompt_hash = br.prompt_hash
    WHERE br.use_case = target_use_case
      AND br.avg_accuracy_score IS NOT NULL
      AND COALESCE(br.low_confidence, FALSE) = FALSE
      AND COALESCE(br.eval_conflict_flag, FALSE) = FALSE
      AND COALESCE(br.invalid, FALSE) = FALSE
      AND 1 - (pe.embedding <=> query_embedding) >= min_similarity
    ORDER BY pe.embedding <=> query_embedding
    LIMIT result_limit;
$$;

-- Phase 3 scaffolding for later promotion.
CREATE TABLE IF NOT EXISTS model_priors (
    model_id TEXT,
    use_case TEXT,
    prompt_complexity TEXT,
    prior_accuracy FLOAT,
    prior_cost FLOAT,
    prior_latency FLOAT,
    support_n INTEGER,
    last_updated TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (model_id, use_case, prompt_complexity)
);

CREATE TABLE IF NOT EXISTS pairwise_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash TEXT NOT NULL,
    use_case TEXT NOT NULL,
    complexity TEXT,
    model_a TEXT NOT NULL,
    model_b TEXT NOT NULL,
    response_a TEXT,
    response_b TEXT,
    winner TEXT NOT NULL,
    winner_model TEXT NOT NULL,
    loser_model TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pairwise_prompt_hash
    ON pairwise_results (prompt_hash);

CREATE INDEX IF NOT EXISTS idx_pairwise_use_case_complexity
    ON pairwise_results (use_case, complexity);

CREATE INDEX IF NOT EXISTS idx_pairwise_models
    ON pairwise_results (model_a, model_b);

CREATE INDEX IF NOT EXISTS idx_pairwise_winner_model
    ON pairwise_results (winner_model);

CREATE TABLE IF NOT EXISTS model_win_rates (
    model_id TEXT NOT NULL,
    use_case TEXT NOT NULL,
    complexity TEXT NOT NULL,
    win_rate FLOAT,
    total_matches INTEGER NOT NULL,
    total_participations INTEGER,
    decisive_matches INTEGER,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    ties INTEGER NOT NULL,
    tie_rate FLOAT,
    judge_count INTEGER,
    confidence FLOAT,
    last_updated TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (model_id, use_case, complexity)
);

CREATE INDEX IF NOT EXISTS idx_model_win_rates_lookup
    ON model_win_rates (use_case, complexity, win_rate DESC);

CREATE TABLE IF NOT EXISTS routing_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id TEXT,
    prompt_hash TEXT,
    use_case TEXT,
    complexity TEXT,
    clarity TEXT,
    recommended_model TEXT,
    data_source TEXT,
    knn_neighbors INTEGER,
    filter_level TEXT,
    expected_accuracy FLOAT,
    confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_routing_log_prompt_hash
    ON routing_log (prompt_hash);

CREATE INDEX IF NOT EXISTS idx_routing_log_created_at
    ON routing_log (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_routing_log_recommended_model
    ON routing_log (recommended_model);
