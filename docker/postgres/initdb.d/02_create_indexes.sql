-- Vector and Composite Indexes for Performance
-- This script creates essential indexes for fast similarity search and common query patterns

-- ============================================================================
-- VECTOR SIMILARITY INDEX
-- ============================================================================
-- HNSW index for fast approximate nearest neighbor search on embeddings
-- Uses cosine distance for similarity (best for normalized embeddings)
-- Note: This may take several minutes on large tables
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);

-- Alternative: IVFFlat index (faster build, slightly slower query)
-- Uncomment if HNSW build is too slow:
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
-- ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- COMPOSITE INDEXES FOR COMMON QUERY PATTERNS
-- ============================================================================

-- Document filtering by type and date (for retrieval filters)
CREATE INDEX IF NOT EXISTS idx_documents_type_date
ON documents(doc_type, publication_date DESC);

-- Document status for pipeline processing
CREATE INDEX IF NOT EXISTS idx_documents_status
ON documents(status)
WHERE status IN ('NEW', 'TEXT_EXTRACTED', 'CHUNKS_BUILT', 'FAILED');

-- Chunk document lookup (for joining chunk → document metadata)
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks(document_id)
INCLUDE (text, page_start, page_end, section_hint);

-- Chunks missing embeddings (for batch embedding generation)
CREATE INDEX IF NOT EXISTS idx_chunks_null_embedding
ON chunks(document_id)
WHERE embedding IS NULL;

-- ============================================================================
-- FULL-TEXT SEARCH INDEX (for hybrid retrieval)
-- ============================================================================

-- Add tsvector column for full-text search if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'text_tsv'
    ) THEN
        ALTER TABLE chunks ADD COLUMN text_tsv tsvector;
    END IF;
END $$;

-- Populate tsvector column
UPDATE chunks SET text_tsv = to_tsvector('english', text) WHERE text_tsv IS NULL;

-- Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
ON chunks USING gin(text_tsv);

-- Create trigger to keep tsvector updated
CREATE OR REPLACE FUNCTION chunks_text_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.text_tsv := to_tsvector('english', NEW.text);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvector_update ON chunks;
CREATE TRIGGER tsvector_update
    BEFORE INSERT OR UPDATE OF text
    ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_text_tsv_trigger();

-- ============================================================================
-- CHAT & FEEDBACK INDEXES
-- ============================================================================

-- Chat messages by session (for conversation history)
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
ON chat_messages(session_id, created_at DESC);

-- Chat sessions by creation date
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created
ON chat_sessions(created_at DESC);

-- ============================================================================
-- STATISTICS UPDATE
-- ============================================================================

-- Update table statistics for query planner
ANALYZE chunks;
ANALYZE documents;
ANALYZE chat_messages;
ANALYZE chat_sessions;
