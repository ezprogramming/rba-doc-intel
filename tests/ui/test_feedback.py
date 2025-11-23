"""Basic tests for feedback helper logic."""

from types import SimpleNamespace

import pytest
from app.db.models import Feedback
from app.ui.streamlit_app import store_feedback


class DummySession:
    """Minimal stand-in for SQLAlchemy session used in tests."""

    def __init__(self, existing: object | None = None):
        self._existing = existing
        self.added = None

    def query(self, model):  # pragma: no cover - exercised via store_feedback
        assert model is Feedback

        class _Query:
            def __init__(self, existing):
                self._existing = existing

            def filter(self, *_, **__):
                return self

            def first(self):
                return self._existing

        return _Query(self._existing)

    def add(self, obj):
        self.added = obj


class DummyScope:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def patch_session(monkeypatch):
    def _patch(session: DummySession):
        monkeypatch.setattr(
            "app.ui.streamlit_app.session_scope",
            lambda: DummyScope(session),
        )

    return _patch


def test_store_feedback_updates_existing(patch_session):
    existing = SimpleNamespace(score=0, comment=None)
    session = DummySession(existing=existing)
    patch_session(session)

    store_feedback(message_id=1, score=1, comment="Great answer")

    assert existing.score == 1
    assert existing.comment == "Great answer"
    assert session.added is None


def test_store_feedback_creates_new_record(patch_session):
    session = DummySession(existing=None)
    patch_session(session)

    store_feedback(message_id=2, score=-1, comment=None)

    assert isinstance(session.added, Feedback)
    assert session.added.chat_message_id == 2
    assert session.added.score == -1
