"""PDF processing pipeline placeholder.

Steps to implement:
1. Fetch documents with status NEW from Postgres.
2. Download PDFs from MinIO.
3. Extract pages, clean text, create chunks.
4. Persist pages/chunks and update document status.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("PDF processing pipeline not implemented yet.")


if __name__ == "__main__":
    main()
