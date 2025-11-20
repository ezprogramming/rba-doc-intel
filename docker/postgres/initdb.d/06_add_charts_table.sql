-- Add charts table and link to chunks for multimodal RAG support
-- Charts (graphs, visualizations) complement tables for financial data analysis

CREATE TABLE IF NOT EXISTS charts (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,

    -- Image metadata (width, height, format, image_index)
    image_metadata JSONB NOT NULL,

    -- Bounding box coordinates on page (x0, y0, x1, y1)
    bbox JSONB,

    -- Optional: S3 key if chart image is saved to MinIO
    -- Format: derived/charts/{document_id}/page_{page_num}_chart_{idx}.{ext}
    s3_key TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Link chunks to charts (similar to table_id)
-- Allows retrieval to surface: "This answer is grounded in Chart X on Page Y"
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS chart_id BIGINT REFERENCES charts(id) ON DELETE SET NULL;

-- Index for efficient chart lookups from chunks
CREATE INDEX IF NOT EXISTS idx_chunks_chart_id
    ON chunks(chart_id);

-- Index for finding all charts in a document
CREATE INDEX IF NOT EXISTS idx_charts_document_id
    ON charts(document_id);

-- Index for finding charts by page
CREATE INDEX IF NOT EXISTS idx_charts_page_number
    ON charts(document_id, page_number);
