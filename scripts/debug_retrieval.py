"""Debug script to analyze retrieval quality for specific queries.

This helps diagnose:
1. What chunks are being retrieved
2. Whether they contain relevant information
3. If retrieval parameters need tuning
"""

import sys
from app.config import get_settings
from app.db.session import session_scope
from app.embeddings.client import EmbeddingClient
from app.rag.retriever import retrieve_similar_chunks
from app.db.models import Document

# Query to analyze
TEST_QUERY = "What is the inflation trend in Sydney in 2025"

def main():
    print("=" * 80)
    print(f"RETRIEVAL DEBUG FOR QUERY: {TEST_QUERY}")
    print("=" * 80)

    # Step 1: Check what documents we have
    print("\n## STEP 1: Available Documents")
    print("-" * 80)
    with session_scope() as session:
        docs = session.query(Document).order_by(Document.publication_date.desc()).all()
        print(f"Total documents: {len(docs)}\n")
        for doc in docs[:20]:  # Show latest 20
            print(f"  - [{doc.doc_type}] {doc.title}")
            print(f"    Date: {doc.publication_date}, Status: {doc.status}")

    # Step 2: Get embedding for query
    print("\n## STEP 2: Query Embedding")
    print("-" * 80)
    embedding_client = EmbeddingClient()
    query_embedding = embedding_client.embed([TEST_QUERY]).vectors[0]
    print(f"Query embedded successfully (dimension: {len(query_embedding)})")

    # Step 3: Test retrieval with different parameters
    print("\n## STEP 3: Retrieval Results")
    print("-" * 80)

    test_cases = [
        ("Default (top_k=6, no rerank)", 6, False),
        ("More chunks (top_k=10, no rerank)", 10, False),
        ("With reranking (top_k=6, rerank=True)", 6, True),
        ("More + reranking (top_k=10, rerank=True)", 10, True),
    ]

    for name, top_k, rerank in test_cases:
        print(f"\n### {name}")
        print(f"Parameters: top_k={top_k}, rerank={rerank}")

        with session_scope() as session:
            chunks = retrieve_similar_chunks(
                session,
                query_text=TEST_QUERY,
                query_embedding=query_embedding,
                limit=top_k,
                rerank=rerank,
            )

            print(f"Retrieved {len(chunks)} chunks:\n")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. [{chunk.doc_type}] {chunk.title}")
                print(f"     Pages: {chunk.page_start}-{chunk.page_end}")
                print(f"     Score: {chunk.score:.4f}")
                print(f"     Section: {chunk.section_hint or 'N/A'}")
                print(f"     Snippet: {chunk.text[:200]}...")
                print()

    # Step 4: Check for specific keywords in chunks
    print("\n## STEP 4: Keyword Analysis")
    print("-" * 80)
    keywords = ["inflation", "Sydney", "2025", "forecast", "trend", "price"]

    with session_scope() as session:
        chunks = retrieve_similar_chunks(
            session,
            query_text=TEST_QUERY,
            query_embedding=query_embedding,
            limit=10,
            rerank=True,
        )

        for keyword in keywords:
            matches = sum(1 for c in chunks if keyword.lower() in c.text.lower())
            print(f"  '{keyword}': found in {matches}/{len(chunks)} chunks")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
