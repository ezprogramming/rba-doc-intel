"""Create database schema if it does not already exist."""

from __future__ import annotations

from sqlalchemy import text

from app.db.models import Base
from app.db.session import get_engine


def main() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_url TEXT"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_length INTEGER"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS documents_content_hash_idx "
                "ON documents (content_hash)"
            )
        )
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'chat_sessions' AND column_name = 'metadata'
                    ) THEN
                        EXECUTE 'ALTER TABLE chat_sessions RENAME COLUMN metadata TO metadata_json';
                    END IF;
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'chat_messages' AND column_name = 'metadata'
                    ) THEN
                        EXECUTE 'ALTER TABLE chat_messages RENAME COLUMN metadata TO metadata_json';
                    END IF;
                END$$;
                """
            )
        )
    print("Database schema ensured.")


if __name__ == "__main__":
    main()
