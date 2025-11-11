from app.rag.pipeline import _compose_analysis
from app.rag.retriever import RetrievedChunk


def test_compose_analysis_handles_chunks():
    chunk = RetrievedChunk(
        chunk_id=1,
        document_id="doc-123",
        text="Sample text",
        doc_type="SMP",
        title="Statement on Monetary Policy – Nov 2025",
        publication_date="2025-11-04",
        page_start=2,
        page_end=3,
        section_hint=None,
        score=0.1,
    )
    analysis = _compose_analysis([chunk])
    assert "Statement on Monetary Policy" in analysis
    assert "pages 2-3" in analysis
