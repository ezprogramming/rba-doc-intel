"""Database models for the RBA Document Intelligence Platform."""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from uuid import uuid4, UUID

from sqlalchemy import Column, Date, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, relationship

from pgvector.sqlalchemy import Vector


Base = declarative_base()


def default_uuid() -> UUID:
    return uuid4()


class DocumentStatus(str, PyEnum):
    NEW = "NEW"
    TEXT_EXTRACTED = "TEXT_EXTRACTED"
    CHUNKS_BUILT = "CHUNKS_BUILT"
    EMBEDDED = "EMBEDDED"
    FAILED = "FAILED"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_documents_content_hash"),
    )

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
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
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
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="pages")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=True)
    section_hint = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=default_uuid)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    metadata_json = Column(JSON, nullable=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    metadata_json = Column(JSON, nullable=True)
