"""End-to-end RAG pipeline with optional cross-encoder reranking.

RAG Pipeline Stages:
====================
1. Query embedding (bi-encoder)
2. Hybrid retrieval (vector similarity + full-text search)
3. Optional reranking (cross-encoder for better precision)
4. Context formatting
5. LLM generation
6. Response formatting with evidence citations

When to enable reranking?
=========================
- Production systems where answer quality matters
- Complex analytical queries requiring deep understanding
- Trade-off: +200-500ms latency for +25-40% accuracy gain

When to disable reranking?
==========================
- Simple keyword queries (hybrid search already works well)
- Latency-critical applications
- Development/debugging (faster iteration)
- Low-resource environments (reranker needs ~90MB model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List
from uuid import UUID

from app.db.models import ChatMessage, ChatSession, Table
from app.db.session import session_scope
from app.embeddings.client import EmbeddingClient
from app.rag.hooks import hooks
from app.rag.llm_client import LLMClient
from app.rag.retriever import RetrievedChunk, retrieve_similar_chunks
from app.rag.safety import check_answer_safety, check_query_safety

logger = logging.getLogger(__name__)


@dataclass
class AnswerResponse:
    answer: str
    evidence: List[dict]
    analysis: str | None = None


SYSTEM_PROMPT = """
You are a financial analyst. Answer questions using ONLY the RBA document excerpts provided.

STRICT RULES:
1. Use ONLY the provided context - do NOT add outside knowledge
2. If context lacks information, say: "Based on the provided RBA documents, I cannot find specific information about [topic]."
3. Always cite document name and page numbers
4. Include numbers and dates from the context
5. Stay focused on answering the specific question asked

EXAMPLE OF CORRECT ANSWER FORMAT:

Question: "What is the inflation forecast for 2024?"
Context: "[SMP Feb 2024] Inflation is expected to decline to 3.2% by December 2024 (page 12)"

GOOD ANSWER:
"According to the Statement on Monetary Policy - February 2024 (page 12), inflation is forecast to decline to 3.2% by December 2024."

BAD ANSWER (uses outside knowledge):
"Inflation is affected by global supply chains and consumer demand..."

Remember: Answer the question using ONLY what's in the provided context.
"""


def _format_context(chunks: List[RetrievedChunk]) -> str:
    formatted = []
    for chunk in chunks:
        header = f"[{chunk.doc_type}] {chunk.title} (pages {chunk.page_start}-{chunk.page_end})"
        formatted.append(f"{header}\n{chunk.text}")
    return "\n\n".join(formatted)


def _compose_analysis(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "No supporting excerpts retrieved; unable to ground an answer."
    summaries = []
    for chunk in chunks:
        page_range = (
            f"pages {chunk.page_start}-{chunk.page_end}"
            if chunk.page_start is not None
            else "unspecified pages"
        )
        summaries.append(f"{chunk.title} ({chunk.doc_type}, {page_range})")
    return "Answer grounded in " + "; ".join(summaries)


TokenHandler = Callable[[str], None]


def answer_query(
    query: str,
    session_id: UUID | None = None,
    top_k: int = 6,
    stream_handler: TokenHandler | None = None,
    use_reranking: bool = False,
    safety_enabled: bool = True,
) -> AnswerResponse:
    """Run end-to-end RAG pipeline on user query with optional safety checks.

    Args:
        query: User question text
        session_id: Optional chat session ID for persistence
        top_k: Number of chunks to retrieve for context (default: 6)
        stream_handler: Optional callback for streaming LLM responses
        use_reranking: Whether to use cross-encoder reranking (default: False)
                      When True:
                      - Retrieves top_k * 10 candidates (e.g., 2 * 10 = 20)
                      - Reranks to top_k (e.g., 20 → 2 most relevant)
                      - +200-500ms latency, +25-40% accuracy
        safety_enabled: Whether to run safety guardrails (default: True)
                       When True:
                       - Checks query for PII, prompt injection, toxicity
                       - Checks answer for PII, toxicity
                       - Blocks unsafe requests with error message
                       - +< 5ms overhead

    Returns:
        AnswerResponse with answer text, evidence chunks, and analysis

    Pipeline flow:
    1. Safety check on query (if enabled)
    2. Embed query using bi-encoder
    3. Retrieve chunks (hybrid search + optional reranking)
    4. Format context from top chunks
    5. Generate answer using LLM
    6. Safety check on answer (if enabled)
    7. Persist chat history
    8. Return structured response

    Performance examples:
    - Without reranking, with safety (top_k=2): ~55-205ms
    - With reranking, with safety (top_k=2, retrieves 20): ~305-705ms
    """
    # Step 0: Safety check on query
    # Why check query first? Block unsafe requests before expensive operations
    # What we check: PII, prompt injection, toxic content
    if safety_enabled:
        logger.debug("Running safety check on query")
        safety_result = check_query_safety(query)

        if not safety_result.is_safe:
            # Query violated safety policies
            # Return error response without running RAG pipeline
            error_message = (
                "I cannot process this request due to safety concerns. "
                "Please rephrase your question without sensitive information "
                "or potentially harmful content."
            )
            logger.warning(f"Query blocked by safety check: {safety_result.violations}")

            # Return error response
            # Why still return AnswerResponse? Consistent interface for UI
            return AnswerResponse(
                answer=error_message,
                evidence=[],
                analysis=f"Query blocked: {safety_result.details}",
            )

    hooks.emit(
        "rag:query_started",
        query=query,
        session_id=str(session_id) if session_id else None,
        top_k=top_k,
        rerank=use_reranking,
    )

    embedding_client = EmbeddingClient()
    llm_client = LLMClient()
    table_lookup: dict[int, dict] = {}

    # Step 1: Embed query
    # Why bi-encoder? Fast encoding (single forward pass, ~5-10ms)
    question_vector = embedding_client.embed([query]).vectors[0]

    # Step 2: Retrieve relevant chunks
    with session_scope() as session:
        chunks = retrieve_similar_chunks(
            session,
            query_text=query,
            query_embedding=question_vector,
            limit=top_k,
            rerank=use_reranking,  # Enable cross-encoder reranking if requested
        )
        table_ids = {chunk.table_id for chunk in chunks if chunk.table_id is not None}
        if table_ids:
            table_rows = session.query(Table).filter(Table.id.in_(table_ids)).all()
            table_lookup = {
                table_row.id: {
                    "table_id": table_row.id,
                    "page_number": table_row.page_number,
                    "accuracy": table_row.accuracy,
                    "caption": table_row.caption,
                    "structured_data": table_row.structured_data,
                }
                for table_row in table_rows
            }
        hooks.emit(
            "rag:retrieval_complete",
            query=query,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            session_id=str(session_id) if session_id else None,
            rerank=use_reranking,
        )
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

    # Step 3: Format context and generate answer
    context = _format_context(chunks)
    user_content = f"""Question: {query}

RBA Document Excerpts:
{context}

Instructions: Answer the question using ONLY the information from the excerpts above. Include document names and page numbers. If the excerpts don't contain the answer, say so.

Answer:"""
    messages = [{"role": "user", "content": user_content}]

    # Step 4: Generate answer using LLM
    if stream_handler:

        def wrapped(delta: str) -> None:
            hooks.emit("rag:stream_chunk", session_id=str(session_id_value), token_size=len(delta))
            stream_handler(delta)

        answer_text = llm_client.stream(SYSTEM_PROMPT, messages, wrapped)
    else:
        answer_text = llm_client.complete(SYSTEM_PROMPT, messages)

    # Step 5: Safety check on generated answer
    # Why check answer? LLM might hallucinate PII or generate toxic content
    # What we check: PII, toxic content (no prompt injection check on output)
    if safety_enabled:
        logger.debug("Running safety check on answer")
        answer_safety = check_answer_safety(answer_text)

        if not answer_safety.is_safe:
            # Answer violated safety policies
            # Redact answer and return generic response
            logger.warning(f"Answer blocked by safety check: {answer_safety.violations}")
            answer_text = (
                "I apologize, but I cannot provide this information due to "
                "safety and privacy concerns. Please rephrase your question."
            )

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
            "section_hint": chunk.section_hint,
            "table": table_lookup.get(chunk.table_id) if chunk.table_id else None,
        }
        for chunk in chunks
    ]

    with session_scope() as session:
        session.add(ChatMessage(session_id=session_id_value, role="assistant", content=answer_text))

    hooks.emit(
        "rag:answer_completed",
        session_id=str(session_id_value),
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        evidence_count=len(chunks),
        answer_length=len(answer_text),
    )

    analysis = _compose_analysis(chunks)

    return AnswerResponse(answer=answer_text, evidence=evidence_payload, analysis=analysis)
