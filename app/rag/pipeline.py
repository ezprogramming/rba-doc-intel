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

import tiktoken

from app.config import get_settings
from app.db.models import ChatMessage, ChatSession, Table
from app.db.session import session_scope
from app.embeddings.client import EmbeddingClient
from app.rag.hooks import hooks
from app.rag.llm_client import LLMClient
from app.rag.retriever import RetrievedChunk, format_table_as_markdown, retrieve_similar_chunks
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
2. If context lacks information, say: "Based on the provided RBA documents,
   I cannot find specific information about [topic]."
3. Always cite document name and page numbers
4. Include numbers and dates from the context
5. Stay focused on answering the specific question asked

EXAMPLE OF CORRECT ANSWER FORMAT:

Question: "What is the inflation forecast for 2024?"
Context: "[SMP Feb 2024] Inflation is expected to decline to 3.2% by December 2024 (page 12)"

GOOD ANSWER:
"According to the Statement on Monetary Policy - February 2024 (page 12),
inflation is forecast to decline to 3.2% by December 2024."

BAD ANSWER (uses outside knowledge):
"Inflation is affected by global supply chains and consumer demand..."

Remember: Answer the question using ONLY what's in the provided context.
"""


def _format_context(
    chunks: List[RetrievedChunk], table_lookup: dict[int, dict] | None = None
) -> str:
    """Format retrieved chunks as context for LLM prompt.

    For table chunks (chunks with table_id):
    - Uses markdown table format for better LLM reasoning
    - Includes caption and page number
    - Falls back to flattened text if markdown generation fails

    For regular text chunks:
    - Uses existing format (document header + text)

    Args:
        chunks: List of retrieved chunks
        table_lookup: Dict mapping table_id -> table metadata/structured_data
                     (populated during retrieval)

    Returns:
        Formatted context string for LLM prompt
    """
    formatted = []
    table_lookup = table_lookup or {}

    for chunk in chunks:
        header = f"[{chunk.doc_type}] {chunk.title} (pages {chunk.page_start}-{chunk.page_end})"

        # Check if this chunk is from a table
        if chunk.table_id and chunk.table_id in table_lookup:
            table_data = table_lookup[chunk.table_id]
            try:
                # Format table as markdown for better LLM understanding
                markdown_table = format_table_as_markdown(
                    structured_data=table_data["structured_data"],
                    caption=table_data.get("caption"),
                )
                # Add document header + markdown table
                formatted.append(f"{header}\n{markdown_table}")
            except Exception as e:
                # Graceful fallback: use flattened text if markdown generation fails
                logger.warning(
                    f"Failed to format table {chunk.table_id} as markdown: {e}. "
                    f"Falling back to flattened text."
                )
                formatted.append(f"{header}\n{chunk.text}")
        else:
            # Regular text chunk: use existing format
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


def _validate_context_budget(
    chunks: List[RetrievedChunk],
    context_text: str,
    prompt_template: str,
    max_tokens: int,
) -> tuple[List[RetrievedChunk], str]:
    """Validate and truncate context to fit within token budget.

    Phase 6: Context window management to prevent LLM errors.

    Args:
        chunks: Retrieved chunks (sorted by relevance)
        context_text: Formatted context string
        prompt_template: The full prompt template with context
        max_tokens: Maximum allowed tokens (from config)

    Returns:
        (truncated_chunks, truncated_context) if needed, otherwise original

    Strategy:
    - Count tokens in full prompt using tiktoken (GPT tokenizer as proxy)
    - If exceeds budget, truncate chunks from lowest-scoring first
    - Always preserve top-3 chunks for minimum quality
    - Log warnings when truncation occurs

    Why this matters:
    - Prevents "context too long" errors from LLM
    - Ensures consistent behavior regardless of retrieval count
    - Prioritizes most relevant chunks when budget exceeded
    """
    # Initialize tokenizer (use GPT-3.5 as proxy for token counting)
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")

    # Count tokens in full prompt
    total_tokens = len(encoding.encode(prompt_template))

    # If within budget, return as-is
    if total_tokens <= max_tokens:
        logger.debug(f"Context within budget: {total_tokens}/{max_tokens} tokens")
        return chunks, context_text

    # Exceeded budget - need to truncate
    logger.warning(
        f"Context exceeds budget: {total_tokens}/{max_tokens} tokens. "
        f"Truncating from {len(chunks)} chunks."
    )

    # Strategy: Remove chunks from the end (lowest relevance) until we fit
    # Always keep at least top-3 chunks
    min_chunks_to_keep = min(3, len(chunks))

    truncated_chunks = chunks[:]
    while len(truncated_chunks) > min_chunks_to_keep:
        # Remove last (lowest-scoring) chunk
        truncated_chunks = truncated_chunks[:-1]

        # Reformat context with remaining chunks
        truncated_context = _format_context(truncated_chunks)
        truncated_prompt = prompt_template.replace(context_text, truncated_context)

        # Check if now within budget
        new_token_count = len(encoding.encode(truncated_prompt))
        if new_token_count <= max_tokens:
            logger.info(
                f"Context truncated to {len(truncated_chunks)} chunks "
                f"({new_token_count}/{max_tokens} tokens)"
            )
            return truncated_chunks, truncated_context

    # If still exceeded even with min chunks, return min chunks and warn
    truncated_context = _format_context(truncated_chunks)
    final_tokens = len(encoding.encode(prompt_template.replace(context_text, truncated_context)))
    logger.warning(
        f"Context still exceeds budget with minimum {min_chunks_to_keep} chunks "
        f"({final_tokens}/{max_tokens} tokens). Using minimum chunks anyway."
    )
    return truncated_chunks, truncated_context


TokenHandler = Callable[[str], None]


def answer_query(
    query: str,
    session_id: UUID | None = None,
    top_k: int = 10,
    stream_handler: TokenHandler | None = None,
    use_reranking: bool = False,
    safety_enabled: bool = True,
) -> AnswerResponse:
    """Run end-to-end RAG pipeline on user query with optional safety checks.

    Args:
        query: User question text
        session_id: Optional chat session ID for persistence
        top_k: Number of chunks to retrieve for context (default: 10)
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
    # Pass table_lookup to format tables as markdown in the prompt
    context = _format_context(chunks, table_lookup=table_lookup)
    user_content = f"""Question: {query}

RBA Document Excerpts:
{context}

Instructions:
- Provide a COMPREHENSIVE answer using ONLY the information from the excerpts above
- Include ALL relevant details: numbers, dates, trends, forecasts
- Cite document names and page numbers for each point
- If the excerpts don't contain specific information requested, acknowledge this limitation
- Synthesize information from multiple excerpts if available

Answer (provide 3-5 sentences with full details):"""

    # Phase 6: Validate context fits within token budget
    # Truncates low-scoring chunks if needed to prevent LLM errors
    settings = get_settings()
    chunks, context = _validate_context_budget(
        chunks, context, user_content, settings.max_context_tokens
    )

    # Rebuild user_content with potentially truncated context
    user_content = f"""Question: {query}

RBA Document Excerpts:
{context}

Instructions:
- Provide a COMPREHENSIVE answer using ONLY the information from the excerpts above
- Include ALL relevant details: numbers, dates, trends, forecasts
- Cite document names and page numbers for each point
- If the excerpts don't contain specific information requested, acknowledge this limitation
- Synthesize information from multiple excerpts if available

Answer (provide 3-5 sentences with full details):"""
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
