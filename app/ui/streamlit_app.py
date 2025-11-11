"""Streamlit UI for the RAG experience."""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

from app.rag.pipeline import answer_query


st.set_page_config(page_title="RBA Document Intelligence", layout="wide")
st.title("RBA Document Intelligence Platform")


def init_session() -> None:
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []
    if "streaming_answer" not in st.session_state:
        st.session_state.streaming_answer = ""


def render_history() -> None:
    for entry in st.session_state.history:
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**Assistant:** {entry['answer']}")
        with st.expander("Evidence"):
            for evidence in entry["evidence"]:
                pages = evidence["pages"]
                page_label = (
                    f"pages {pages[0]}-{pages[1]}"
                    if all(p is not None for p in pages)
                    else "pages n/a"
                )
                section = f" · {evidence['section_hint']}" if evidence.get("section_hint") else ""
                st.write(
                    f"- {evidence['doc_type']} · {evidence['title']}{section} · {page_label}"
                )
                st.caption(evidence["snippet"])


def handle_submit(question: str) -> None:
    if not question.strip():
        st.warning("Please enter a question.")
        return
    st.markdown(f"**You:** {question}")
    answer_placeholder = st.empty()

    def on_token(delta: str) -> None:
        st.session_state.streaming_answer += delta
        answer_placeholder.markdown(f"**Assistant:** {st.session_state.streaming_answer}")

    st.session_state.streaming_answer = ""
    response = answer_query(
        question,
        session_id=st.session_state.chat_session_id,
        stream_handler=on_token,
    )
    answer_placeholder.empty()
    st.session_state.streaming_answer = ""
    st.session_state.history.append(
        {
            "question": question,
            "answer": response.answer,
            "evidence": response.evidence,
        }
    )


def main() -> None:
    init_session()
    with st.form("chat-form"):
        question = st.text_area("Ask about Reserve Bank publications:", height=120)
        submitted = st.form_submit_button("Send")
        if submitted:
            handle_submit(question)
    st.divider()
    render_history()


if __name__ == "__main__":
    main()
