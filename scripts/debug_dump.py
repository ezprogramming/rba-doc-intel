"""Quick inspection script for pipeline counts."""

from __future__ import annotations

from app.db.models import ChatMessage, ChatSession, Chunk, Document, Page
from app.db.session import session_scope


def main() -> None:
    with session_scope() as session:
        doc_count = session.query(Document).count()
        page_count = session.query(Page).count()
        chunk_count = session.query(Chunk).count()
        session_count = session.query(ChatSession).count()
        message_count = session.query(ChatMessage).count()

    print("Documents:", doc_count)
    print("Pages:", page_count)
    print("Chunks:", chunk_count)
    print("Chat Sessions:", session_count)
    print("Chat Messages:", message_count)


if __name__ == "__main__":
    main()
