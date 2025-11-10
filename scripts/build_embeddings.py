"""CLI for backfilling embeddings."""

from __future__ import annotations

import time

from app.embeddings.indexer import generate_missing_embeddings


def main() -> None:
    total = 0
    while True:
        updated = generate_missing_embeddings()
        if updated == 0:
            break
        total += updated
        print(f"Embedded {updated} chunks in this pass (total={total}).", flush=True)
        time.sleep(0.5)
    print(f"Embedding backfill complete. Total updated: {total}.")


if __name__ == "__main__":
    main()
