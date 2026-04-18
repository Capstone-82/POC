-- =============================================================================
-- FIX 1: Update prompt_embeddings column from vector(384) to vector(1536)
-- =============================================================================
-- Your backfill used OpenAI text-embedding-3-small with dimensions=1536
-- but the table was originally created with vector(384).
-- This ALTER changes the column type to match the actual data.

ALTER TABLE prompt_embeddings
    ALTER COLUMN embedding TYPE vector(1536);

-- Rebuild the HNSW index for the new dimension
DROP INDEX IF EXISTS idx_pe_embedding_hnsw;
CREATE INDEX idx_pe_embedding_hnsw
    ON prompt_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- FIX 2: Recreate knn_search function with vector(1536) input
-- =============================================================================
-- The old function accepted vector(384) which silently returned 0 results
-- when 1536-dim vectors were passed.

DROP FUNCTION IF EXISTS knn_search(vector(384), TEXT, INT, FLOAT);

CREATE OR REPLACE FUNCTION knn_search(
    query_embedding vector(1536),
    target_use_case TEXT,
    result_limit INT DEFAULT 20,
    min_similarity FLOAT DEFAULT 0.60
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
      AND COALESCE(br.invalid, FALSE) = FALSE
      AND 1 - (pe.embedding <=> query_embedding) >= min_similarity
    ORDER BY pe.embedding <=> query_embedding
    LIMIT result_limit;
$$;

-- =============================================================================
-- VERIFY: Quick smoke test
-- =============================================================================
-- After running the above, test with:
--
SELECT count(*) FROM prompt_embeddings WHERE embedding IS NOT NULL;
SELECT vector_dims(embedding) FROM prompt_embeddings LIMIT 1;

Then test the function:
SELECT * FROM knn_search(
  (SELECT embedding FROM prompt_embeddings LIMIT 1),
  'code-generation',
  20,
  0.5
);
