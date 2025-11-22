"""Streamlit UI for the RAG experience with feedback collection.

Why collect feedback?
=====================
- Identify bad responses for improvement
- Create preference pairs for DPO/RLHF fine-tuning
- Track quality trends over time
- Prioritize which failures to fix first

Workflow:
1. User asks question → LLM generates answer
2. User clicks thumbs up/down button
3. Feedback stored in database with message_id link
4. Optional: User can add comment explaining what went wrong
5. Periodically: analyze feedback, create fine-tuning dataset
6. Fine-tune model on positive examples (SFT) or preference pairs (DPO)

Industry examples:
- ChatGPT: Thumbs up/down on every response
- Claude: User can rate responses and provide feedback
- Perplexity: Feedback buttons on all answers
- Stack Overflow: Upvote/downvote system drives quality
"""

from __future__ import annotations

from uuid import UUID, uuid4

import streamlit as st

from app.db.models import ChatMessage, Feedback
from app.db.session import session_scope
from app.rag.hooks import hooks
from app.rag.pipeline import answer_query

st.set_page_config(page_title="RBA Document Intelligence", layout="wide")
st.title("RBA Document Intelligence Platform")


def init_session() -> None:
    """Initialize Streamlit session state.

    Session state variables:
    - chat_session_id: UUID for current chat session
    - history: List of question/answer pairs with feedback status
    - streaming_answer: Buffer for streaming LLM responses
    - feedback_state: Dict mapping message_id -> feedback score (-1, 0, 1)

    Why session_state?
    - Persists across Streamlit reruns
    - Maintains chat history
    - Tracks user feedback
    """
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feedback_state" not in st.session_state:
        # Track feedback for each message: message_id -> score
        st.session_state.feedback_state = {}


def store_feedback(message_id: int, score: int, comment: str | None = None) -> None:
    """Store user feedback in database.

    Args:
        message_id: ID of the ChatMessage being rated
        score: Feedback score (-1 = thumbs down, +1 = thumbs up)
        comment: Optional user comment explaining the rating

    Why store feedback?
    - Track which responses users find helpful/unhelpful
    - Build training dataset for fine-tuning
    - Prioritize improvements based on user pain points

    Workflow:
    1. User clicks thumbs up/down
    2. Feedback stored with message_id link
    3. Later: Query feedback table to find negative examples
    4. Analyze patterns: What types of questions fail?
    5. Fine-tune model on positive examples or preference pairs
    """
    with session_scope() as session:
        # Check if feedback already exists for this message
        # Why check? User might change their mind (up → down or vice versa)
        existing = session.query(Feedback).filter(Feedback.chat_message_id == message_id).first()

        if existing:
            # Update existing feedback
            # Type ignore needed because SQLAlchemy Column types are complex
            existing.score = score  # type: ignore[assignment]
            if comment:
                existing.comment = comment  # type: ignore[assignment]
        else:
            # Create new feedback
            feedback = Feedback(chat_message_id=message_id, score=score, comment=comment)
            session.add(feedback)

    hooks.emit(
        "ui:feedback_recorded",
        message_id=message_id,
        score=score,
        comment=comment,
    )


def render_history() -> None:
    """Render chat history with feedback buttons.

    For each message:
    1. Show question and answer
    2. Show thumbs up/down buttons
    3. Highlight selected feedback
    4. Show evidence in expandable section

    Why feedback buttons?
    - Easy user feedback collection (no forms)
    - Inline with each response (context-aware)
    - Visual indication of rating status
    """
    for idx, entry in enumerate(st.session_state.history):
        # Display question
        st.markdown(f"**You:** {entry['question']}")

        # Display answer
        pending_flag = entry.get("pending")
        status_suffix = " _(generating...)_" if pending_flag else ""
        st.markdown(f"**Assistant:** {entry['answer']}{status_suffix}")
        if entry.get("error"):
            st.error(entry["error"])

        # Feedback buttons are only shown once the response is finalized
        col1, col2, col3 = st.columns([1, 1, 10])
        if pending_flag or not entry.get("message_id"):
            with col3:
                st.caption("Feedback available after the response is ready.")
        else:
            message_id = entry.get("message_id")
            current_feedback = st.session_state.feedback_state.get(message_id, 0)

            with col1:
                thumb_up_label = "👍" if current_feedback == 1 else "👍"
                if st.button(thumb_up_label, key=f"up_{idx}", disabled=current_feedback == 1):
                    if message_id:
                        store_feedback(message_id, score=1)
                        st.session_state.feedback_state[message_id] = 1
                        st.success("Thanks for your feedback!")
                        st.rerun()

            with col2:
                thumb_down_label = "👎" if current_feedback == -1 else "👎"
                if st.button(thumb_down_label, key=f"down_{idx}", disabled=current_feedback == -1):
                    if message_id:
                        store_feedback(message_id, score=-1)
                        st.session_state.feedback_state[message_id] = -1
                        st.warning("Feedback recorded. What went wrong?")
                        st.rerun()

            if current_feedback == 1:
                with col3:
                    st.caption("✓ Marked helpful")
            elif current_feedback == -1:
                with col3:
                    st.caption("✗ Marked unhelpful")

        # Evidence section (expandable)
        with st.expander("Evidence"):
            for evidence in entry["evidence"]:
                pages = evidence["pages"]
                page_label = (
                    f"pages {pages[0]}-{pages[1]}"
                    if all(p is not None for p in pages)
                    else "pages n/a"
                )
                section = f" · {evidence['section_hint']}" if evidence.get("section_hint") else ""
                st.write(f"- {evidence['doc_type']} · {evidence['title']}{section} · {page_label}")
                st.caption(evidence["snippet"])

        # Divider between messages
        st.divider()


def handle_submit(question: str, use_reranking: bool) -> None:
    """Handle user question submission.

    Workflow:
    1. Validate input
    2. Display question
    3. Stream LLM answer (with live updates)
    4. Retrieve message_id from database for feedback linking
    5. Store in session history with message_id
    6. Rerun to show feedback buttons

    Why retrieve message_id?
    - Links feedback to specific ChatMessage record
    - Enables feedback storage in database
    - Allows querying feedback metrics later
    """
    if not question.strip():
        st.warning("Please enter a question.")
        return

    hooks.emit(
        "ui:question_submitted",
        question=question,
        session_id=st.session_state.chat_session_id,
        rerank=use_reranking,
    )

    entry = {
        "question": question,
        "answer": "",
        "evidence": [],
        "message_id": None,
        "pending": True,
        "error": None,
    }
    st.session_state.history.append(entry)
    answer_placeholder = st.empty()

    def on_token(delta: str) -> None:
        entry["answer"] += delta
        answer_placeholder.markdown(f"**Assistant:** {entry['answer']} _(generating...)_")

    try:
        response = answer_query(
            question,
            session_id=st.session_state.chat_session_id,
            stream_handler=on_token,
            use_reranking=use_reranking,
        )
        entry["answer"] = response.answer
        entry["evidence"] = response.evidence
        hooks.emit(
            "ui:answer_rendered",
            session_id=st.session_state.chat_session_id,
            evidence_count=len(response.evidence),
            rerank=use_reranking,
        )

        # Retrieve message_id from database for feedback linkage
        with session_scope() as session:
            latest_message = (
                session.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == UUID(st.session_state.chat_session_id),
                    ChatMessage.role == "assistant",
                )
                .order_by(ChatMessage.created_at.desc())
                .first()
            )
            if latest_message:
                entry["message_id"] = latest_message.id
                hooks.emit(
                    "ui:message_committed",
                    session_id=st.session_state.chat_session_id,
                    message_id=latest_message.id,
                )

    except Exception as exc:  # noqa: BLE001
        entry["error"] = str(exc)
        if not entry["answer"]:
            entry["answer"] = "Encountered an error while generating a response."
    finally:
        entry["pending"] = False
        answer_placeholder.empty()


def main() -> None:
    init_session()
    use_reranking = st.sidebar.checkbox("Enable reranking", value=True)
    with st.form("chat-form"):
        question = st.text_area("Ask about Reserve Bank publications:", height=120)
        submitted = st.form_submit_button("Send")
        if submitted:
            handle_submit(question, use_reranking)
    st.divider()
    render_history()


if __name__ == "__main__":
    main()
