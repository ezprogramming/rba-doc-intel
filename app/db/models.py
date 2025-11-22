"""Database models for the RBA Document Intelligence Platform."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PyEnum
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def default_uuid() -> UUID:
    return uuid4()


class DocumentStatus(str, PyEnum):
    NEW = "NEW"
    PROCESSING = "PROCESSING"
    TEXT_EXTRACTED = "TEXT_EXTRACTED"
    CHUNKS_BUILT = "CHUNKS_BUILT"
    EMBEDDED = "EMBEDDED"
    FAILED = "FAILED"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_documents_content_hash"),)

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=default_uuid)
    source_system = Column(String, nullable=False)
    source_url = Column(String, nullable=True)
    s3_key = Column(String, nullable=False)
    doc_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    publication_date = Column(Date, nullable=True)
    content_hash = Column(String(128), nullable=True)
    content_length = Column(Integer, nullable=True)
    status = Column(String, default=DocumentStatus.NEW.value, nullable=False)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    raw_text = Column(Text, nullable=True)
    clean_text = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    document = relationship("Document", back_populates="pages")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    table_id = Column(Integer, ForeignKey("tables.id", ondelete="SET NULL"), nullable=True)
    chart_id = Column(Integer, ForeignKey("charts.id", ondelete="SET NULL"), nullable=True)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=True)
    section_hint = Column(String, nullable=True)
    text_tsv = Column(TSVECTOR, nullable=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    document = relationship("Document", back_populates="chunks")
    table = relationship("Table", back_populates="chunks")
    chart = relationship("Chart", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=default_uuid)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    metadata_json = Column(JSON, nullable=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    metadata_json = Column(JSON, nullable=True)


# ============================================================================
# TABLE EXTRACTION MODELS
# ============================================================================
# Store structured tables extracted from RBA PDFs for better numerical queries


class Table(Base):
    """Structured table extracted from PDF page.

    Why separate table storage?
    - Preserves row/column structure lost in plain text
    - Enables structured queries: "GDP forecast for 2025?"
    - Better grounding for numerical questions
    - Can join with chunks to enrich context

    Example data structure:
    structured_data = [
        {"Year": "2024", "GDP": "2.1%", "Inflation": "3.5%"},
        {"Year": "2025", "GDP": "2.5%", "Inflation": "2.8%"}
    ]
    """

    __tablename__ = "tables"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to source document and page
    document_id = Column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number = Column(Integer, nullable=False)

    # Table data (list of row dicts from pandas DataFrame.to_dict('records'))
    structured_data = Column(JSON, nullable=False)

    # Bounding box coordinates [x1, y1, x2, y2] on page
    # Useful for visual highlighting or re-extraction
    bbox = Column(JSON, nullable=True)

    # Camelot accuracy score (0-100)
    # Higher = more confident detection
    # Typical good tables: 80-100
    accuracy = Column(Integer, nullable=True)

    # Optional caption/title extracted from text near table
    caption = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    chunks = relationship("Chunk", back_populates="table")


class Chart(Base):
    """Chart/graph image extracted from PDF page.

    Why extract charts?
    - Visual data complements tables (trends, distributions)
    - Flag chunks with charts for better retrieval
    - Future multimodal RAG: vision LLM can analyze chart content
    - Preserve context: "GDP chart shows declining trend"

    Example metadata structure:
    image_metadata = {
        "width": 600,
        "height": 400,
        "format": "png",
        "image_index": 0  # Index among images on this page
    }
    """

    __tablename__ = "charts"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to source document and page
    document_id = Column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number = Column(Integer, nullable=False)

    # Image metadata (width, height, format, image_index)
    # Stores: dimensions, format (png/jpeg), and index on page
    image_metadata = Column(JSON, nullable=False)

    # Bounding box coordinates [x0, y0, x1, y1] on page
    # Useful for visual highlighting or OCR re-extraction
    bbox = Column(JSON, nullable=True)

    # Optional: S3 key if chart image saved to MinIO
    # Format: derived/charts/{document_id}/page_{num}_chart_{idx}.{ext}
    # Future use: vision LLM can fetch and analyze
    s3_key = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    chunks = relationship("Chunk", back_populates="chart")


# ============================================================================
# EVALUATION & ML ENGINEERING MODELS
# ============================================================================
# Track evaluation experiments, results, and user feedback


class EvalExample(Base):
    """Golden evaluation examples for offline RAG testing.

    Why golden examples?
    - Consistent benchmark across model changes
    - Catch regressions before deployment
    - Measure improvement from fine-tuning
    - Industry standard practice (see: BEIR, MTEB)

    Example:
    query = "What is the RBA's inflation target?"
    expected_keywords = ["2-3", "percent", "medium term"]
    difficulty = "easy"
    """

    __tablename__ = "eval_examples"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # The question to test
    query = Column(Text, nullable=False)

    # Optional reference answer (for ROUGE/semantic similarity metrics)
    gold_answer = Column(Text, nullable=True)

    # Keywords that MUST appear in answer (simpler than full answer matching)
    # Example: ["2-3", "percent", "target"] for inflation target question
    expected_keywords = Column(JSON, nullable=True)

    # For reporting: easy/medium/hard or domain categories
    difficulty = Column(String, nullable=True)
    category = Column(String, nullable=True)  # inflation/employment/forecasts/etc

    # Flexible extension for future fields
    metadata_json = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class EvalRun(Base):
    """Tracks a single evaluation experiment run.

    Why track runs?
    - Compare model versions (qwen1.5b vs qwen7b)
    - Compare prompt versions (v1.0 vs v1.1)
    - Compare retrieval configs (top_k=5 vs top_k=10)
    - A/B testing for production deployment

    Example:
    config = {
        "model_name": "qwen2.5:7b",
        "prompt_version": "v1.2",
        "retrieval_top_k": 5,
        "reranking_enabled": True
    }
    """

    __tablename__ = "eval_runs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=default_uuid)

    # When was this run executed?
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # What configuration was tested?
    # Stores model name, prompt version, retrieval params, etc.
    config = Column(JSON, nullable=False)

    # Status: running/completed/failed
    status = Column(String, default="running", nullable=False)

    # Summary metrics after all examples processed
    # Example: {"total": 50, "passed": 42, "pass_rate": 0.84, "avg_latency_ms": 850}
    summary_metrics = Column(JSON, nullable=True)


class EvalResult(Base):
    """Individual evaluation result for one example in one run.

    Why per-example results?
    - Debug failures: "Why did question X fail?"
    - Error analysis: "What types of questions fail most?"
    - Fine-tuning data: Use failed examples for improvement
    """

    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Which run does this result belong to?
    eval_run_id = Column(
        PG_UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )

    # Which example was tested?
    eval_example_id = Column(Integer, ForeignKey("eval_examples.id"), nullable=False)

    # What did the LLM generate?
    llm_answer = Column(Text, nullable=True)

    # Which chunks were retrieved? (for grounding analysis)
    retrieved_chunks = Column(JSON, nullable=True)

    # How long did it take?
    latency_ms = Column(Integer, nullable=True)

    # Metrics for this specific result
    # Example: {"keyword_match": 0.8, "semantic_similarity": 0.75, "grounding_score": 0.9}
    scores = Column(JSON, nullable=True)

    # Overall pass/fail (based on thresholds)
    passed = Column(Integer, nullable=True)  # 1=pass, 0=fail, NULL=error

    # Error message if generation failed
    error = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class Feedback(Base):
    """User feedback on chat responses (thumbs up/down).

    Why collect feedback?
    - Identify bad responses for improvement
    - Create preference pairs for DPO/RLHF fine-tuning
    - Track quality trends over time
    - Prioritize which failures to fix first

    Workflow:
    1. User asks question → LLM generates answer
    2. User clicks thumbs up/down in Streamlit UI
    3. Feedback stored with chat_message_id link
    4. Periodically: analyze feedback, create fine-tuning dataset
    5. Fine-tune model on positive examples (SFT) or preference pairs (DPO)
    """

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Which message is this feedback for?
    chat_message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False)

    # Thumbs up (+1), thumbs down (-1), neutral (0)
    score = Column(Integer, nullable=False)

    # Optional: user comment explaining what went wrong
    comment = Column(Text, nullable=True)

    # Optional: what should the answer have been?
    # Useful for creating supervised fine-tuning examples
    corrected_answer = Column(Text, nullable=True)

    # Tags for categorizing feedback
    # Example: ["incorrect", "incomplete", "hallucination"]
    tags = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
