from app.pdf.chunker import chunk_pages


def test_chunking_orders_chunks():
    pages = ["word " * 200, "data " * 50]
    chunks = chunk_pages(pages, max_tokens=100)
    assert chunks, "Chunks should be produced"
    assert chunks[0].page_start == 0
    assert chunks[-1].page_end == len(pages) - 1
