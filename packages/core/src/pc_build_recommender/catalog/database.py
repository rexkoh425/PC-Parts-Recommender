"""Database engine and session helpers with a local SQLite fallback."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .orm import Base

DEFAULT_DATABASE_URL = "sqlite:///./pc_build_recommender.db"


def get_database_url(database_url: str | None = None) -> str:
    """Resolve an explicit URL, then DATABASE_URL, then the local fallback."""

    return database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


def build_db_engine(
    database_url: str | None = None,
    *,
    echo: bool = False,
) -> Engine:
    url = get_database_url(database_url)
    options: dict[str, object] = {"echo": echo, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            options["poolclass"] = StaticPool
    engine = create_engine(url, **options)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(
    engine: Engine | None = None,
    *,
    database_url: str | None = None,
    echo: bool = False,
) -> sessionmaker[Session]:
    bind = engine or build_db_engine(database_url, echo=echo)
    return sessionmaker(bind=bind, class_=Session, expire_on_commit=False, autoflush=False)


def init_database(engine: Engine) -> None:
    """Create tables for local/dev use; production deployments should run Alembic."""

    from pc_build_recommender.annotation import orm as _annotation_orm

    _ = _annotation_orm
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Commit a unit of work or roll it back atomically on failure."""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Common aliases used by service layers.
create_engine_from_url = build_db_engine
create_tables = init_database
