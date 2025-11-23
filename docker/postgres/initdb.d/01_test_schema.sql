-- =============================================================================
-- Test Schema - Isolated tables for end-to-end testing
-- =============================================================================
-- This schema allows running tests without affecting production data.
-- The test_workflow.py script uses this schema.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS test;

-- Documents
CREATE TABLE IF NOT EXISTS test.documents (
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
    CONSTRAINT test_uq_documents_content_hash UNIQUE (content_hash)
);

-- Pages
CREATE TABLE IF NOT EXISTS test.pages (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES test.documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    raw_text TEXT,
    clean_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tables
CREATE TABLE IF NOT EXISTS test.tables (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES test.documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    structured_data JSONB NOT NULL,
    bbox JSONB,
    accuracy INTEGER,
    caption TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chunks
CREATE TABLE IF NOT EXISTS test.chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES test.documents(id) ON DELETE CASCADE,
    table_id BIGINT REFERENCES test.tables(id) ON DELETE SET NULL,
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

-- Indexes for test schema
CREATE INDEX IF NOT EXISTS test_idx_chunks_embedding_hnsw
    ON test.chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS test_idx_documents_status
    ON test.documents(status);

CREATE INDEX IF NOT EXISTS test_idx_chunks_document_id
    ON test.chunks(document_id);

CREATE INDEX IF NOT EXISTS test_idx_chunks_table_id
    ON test.chunks(table_id);

CREATE INDEX IF NOT EXISTS test_idx_chunks_text_fts
    ON test.chunks USING gin(text_tsv);

-- Trigger for test schema
CREATE OR REPLACE FUNCTION test.chunks_text_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.text_tsv := to_tsvector('english', NEW.text);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvector_update ON test.chunks;
CREATE TRIGGER tsvector_update
    BEFORE INSERT OR UPDATE OF text
    ON test.chunks
    FOR EACH ROW
    EXECUTE FUNCTION test.chunks_text_tsv_trigger();
