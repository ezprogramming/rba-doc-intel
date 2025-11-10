"""Convenience entry point to refresh the corpus end-to-end."""

from __future__ import annotations

import logging

from scripts import build_embeddings, crawler_rba, process_pdfs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting crawler...")
    crawler_rba.main()
    logging.info("Crawler finished. Processing PDFs...")
    process_pdfs.main()
    logging.info("Processing finished. Generating embeddings...")
    build_embeddings.main()
    logging.info("Refresh run complete.")


if __name__ == "__main__":
    main()
