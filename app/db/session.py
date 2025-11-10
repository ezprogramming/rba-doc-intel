\"\"\"Database session helpers.\"\"\"

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings


_engine = None
_SessionLocal = None


def _init_engine():
    global _engine, _SessionLocal
    settings = get_settings()
    _engine = create_engine(settings.database_url, future=True, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False, future=True)


def get_engine():
    if _engine is None:
        _init_engine()
    return _engine


def get_session_factory():
    if _SessionLocal is None:
        _init_engine()
    return _SessionLocal


@contextmanager
def session_scope():
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

