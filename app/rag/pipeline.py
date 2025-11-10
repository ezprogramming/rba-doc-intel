"""End-to-end RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from uuid import UUID

from app.db.models import ChatMessage, ChatSession
from app.db.session import session_scope
from app.embeddings.client import EmbeddingClient
from app.rag.llm_client import LLMClient
from app.rag.retriever import RetrievedChunk, retrieve_similar_chunks


@dataclass
class AnswerResponse:
    answer: str
    evidence: List[dict]
    analysis: str | None = None


SYSTEM_PROMPT = (
    "You are an analyst answering questions strictly using Reserve Bank of Australia report excerpts. "
    "Cite document titles and page ranges from provided context. "
    "If the context lacks the answer, state that clearly."
)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    formatted = []
    for chunk in chunks:
        header = f"[{chunk.doc_type}] {chunk.title} (pages {chunk.page_start}-{chunk.page_end})"
        formatted.append(f"{header}\n{chunk.text}")
    return "\n\n".join(formatted)


def answer_query(query: str, session_id: UUID | None = None, top_k: int = 5) -> AnswerResponse:
    embedding_client = EmbeddingClient()
    llm_client = LLMClient()

    question_vector = embedding_client.embed([query]).vectors[0]
    with session_scope() as session:
        chunks = retrieve_similar_chunks(session, question_vector, limit=top_k)
        chat_session = None
        if session_id:
            chat_session = session.get(ChatSession, session_id)
        if chat_session is None:
            chat_session = ChatSession()
            session.add(chat_session)
            session.flush()
        # Persist user question
        session_id_value = chat_session.id
        session.add(ChatMessage(session_id=session_id_value, role="user", content=query))

    context = _format_context(chunks)
    user_content = f"Question: {query}\n\nContext:\n{context}"
    messages = [{"role": "user", "content": user_content}]
    answer_text = llm_client.complete(SYSTEM_PROMPT, messages)

    evidence_payload = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "doc_type": chunk.doc_type,
            "title": chunk.title,
            "publication_date": chunk.publication_date,
            "pages": [chunk.page_start, chunk.page_end],
            "score": chunk.score,
            "snippet": chunk.text[:500],
        }
        for chunk in chunks
    ]

    with session_scope() as session:
        session.add(ChatMessage(session_id=session_id_value, role="assistant", content=answer_text))

    return AnswerResponse(answer=answer_text, evidence=evidence_payload)
