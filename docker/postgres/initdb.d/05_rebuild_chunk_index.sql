-- Rebuild idx_chunks_document_id without the bulky text column to avoid
-- "index row size exceeds btree maximum" errors when inserting long table chunks.

DROP INDEX IF EXISTS idx_chunks_document_id;

CREATE INDEX idx_chunks_document_id
    ON chunks(document_id)
    INCLUDE (page_start, page_end, section_hint);
