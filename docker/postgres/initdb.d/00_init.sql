-- =============================================================================
-- RBA Document Intelligence - Database Initialization
-- =============================================================================
-- This file creates all tables, indexes, and triggers needed for production.
-- Consolidated from multiple migration files for simplicity.
-- =============================================================================

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =============================================================================
-- TABLES
-- =============================================================================

-- Documents: PDF publications from RBA
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_system TEXT NOT NULL,
    source_url TEXT,
    s3_key TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    title TEXT NOT NULL,
    publication_date DATE,
    content_hash TEXT,
    content_length INTEGER,
    status TEXT NOT NULL DEFAULT 'NEW',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_documents_content_hash UNIQUE (content_hash)
);

-- Pages: Raw and cleaned text per page
CREATE TABLE IF NOT EXISTS pages (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    raw_text TEXT,
    clean_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tables: Structured table data extracted from PDFs
CREATE TABLE IF NOT EXISTS tables (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    structured_data JSONB NOT NULL,
    bbox JSONB,
    accuracy INTEGER,
    caption TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chunks: Text segments for RAG retrieval
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_id BIGINT REFERENCES tables(id) ON DELETE SET NULL,
    page_start INTEGER,
    page_end INTEGER,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(768),
    section_hint TEXT,
    text_tsv TSVECTOR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chat Sessions: User conversation sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata_json JSONB
);

-- Chat Messages: Individual messages in conversations
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata_json JSONB
);

-- Feedback: Thumbs up/down on chat responses
CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    chat_message_id BIGINT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    score INTEGER NOT NULL,
    comment TEXT,
    corrected_answer TEXT,
    tags JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Eval Examples: Golden test cases for RAG evaluation
CREATE TABLE IF NOT EXISTS eval_examples (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    gold_answer TEXT,
    expected_keywords JSONB,
    difficulty TEXT,
    category TEXT,
    metadata_json JSONB
);

-- Eval Runs: Evaluation run metadata
CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    summary_metrics JSONB
);

-- Eval Results: Individual evaluation results
CREATE TABLE IF NOT EXISTS eval_results (
    id BIGSERIAL PRIMARY KEY,
    eval_run_id UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    eval_example_id BIGINT NOT NULL REFERENCES eval_examples(id) ON DELETE CASCADE,
    llm_answer TEXT,
    retrieved_chunks JSONB,
    latency_ms INTEGER,
    scores JSONB,
    passed INTEGER,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Vector similarity (HNSW for fast approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Document queries
CREATE INDEX IF NOT EXISTS idx_documents_type_date
    ON documents(doc_type, publication_date DESC);

CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents(status)
    WHERE status IN ('NEW', 'PROCESSING', 'TEXT_EXTRACTED', 'CHUNKS_BUILT', 'FAILED');

-- Chunk queries
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_table_id
    ON chunks(table_id);

CREATE INDEX IF NOT EXISTS idx_chunks_null_embedding
    ON chunks(document_id)
    WHERE embedding IS NULL;

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
    ON chunks USING gin(text_tsv);

-- Chat & Feedback
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_created
    ON chat_sessions(created_at DESC);

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update tsvector for full-text search
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

-- =============================================================================
-- STATISTICS
-- =============================================================================

ANALYZE documents;
ANALYZE chunks;
ANALYZE tables;
ANALYZE chat_messages;
ANALYZE chat_sessions;
