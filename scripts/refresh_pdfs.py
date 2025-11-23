"""Convenience entry point to refresh the corpus end-to-end."""

from __future__ import annotations

import logging

from scripts import build_embeddings, crawler_rba, ingest_documents


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting crawler...")
    crawler_rba.main()
    logging.info("Crawler finished. Ingesting documents (text + tables)...")
    ingest_documents.main()
    logging.info("Ingestion finished. Generating embeddings...")
    build_embeddings.main()
    logging.info("Refresh run complete.")


if __name__ == "__main__":
    main()
