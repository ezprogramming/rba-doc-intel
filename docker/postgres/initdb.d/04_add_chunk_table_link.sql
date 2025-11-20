-- Link chunk rows back to structured tables for downstream lookups

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS table_id BIGINT REFERENCES tables(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_table_id
    ON chunks(table_id);
