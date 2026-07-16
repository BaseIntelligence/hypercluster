"""Async SQLAlchemy helpers for challenge-owned SQLite on `/data`."""

from __future__ import annotations

import os
import stat
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

PostInitHook = Callable[["Database"], Awaitable[Any]]


class Base(DeclarativeBase):
    """Base class for challenge-owned SQLAlchemy models."""


class Database:
    """Async SQLAlchemy database wrapper with init/close lifespan hooks."""

    def __init__(
        self,
        database_url: str,
        *,
        post_init_hooks: list[PostInitHook] | None = None,
    ) -> None:
        self.database_url = database_url
        self._initialized = False
        self._closed = False
        # Optional hooks after create_all (e.g. M11 price catalog seed-on-boot).
        self._post_init_hooks: list[PostInitHook] = list(post_init_hooks or [])
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            # Ensure parent directory exists for file-backed SQLite.
            path = _sqlite_path_from_url(database_url)
            if path is not None and path.parent and str(path.parent) not in {"", "."}:
                path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            connect_args=connect_args,
        )
        self._session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            autoflush=False,
        )

    def add_post_init_hook(self, hook: PostInitHook) -> None:
        """Register a coroutine run after successful ``init()`` create_all."""

        self._post_init_hooks.append(hook)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def closed(self) -> bool:
        return self._closed

    async def init(self) -> None:
        """Create challenge-owned tables and mark initialized.

        Runs any registered post-init hooks after ``create_all`` (e.g. optional
        ``HYPER_PRICE_SEED_ON_BOOT`` catalog seed). Hook failures are logged
        and must not prevent the database from marking initialized when tables
        already exist; individual hooks own their safety (seed is only_if_empty).
        """

        # Ensure model modules register metadata before create_all.
        import hypercluster.db.models  # noqa: F401

        async with self.engine.begin() as connection:
            if self.engine.url.get_backend_name().startswith("sqlite"):
                await connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            await connection.run_sync(Base.metadata.create_all)
        self._initialized = True
        self._closed = False
        for hook in list(self._post_init_hooks):
            try:
                await hook(self)
            except Exception:  # noqa: BLE001 — never block app boot on seed
                import logging

                logging.getLogger(__name__).exception(
                    "database post_init hook failed: %r",
                    hook,
                )

    async def close(self) -> None:
        """Dispose database connections."""

        await self.engine.dispose()
        self._closed = True

    async def healthcheck(self) -> bool:
        """Return True when the DB is usable (readiness probe).

        Fail closed when the SQLite parent directory is not writable (VAL-SCAF-035)
        or when a trivial connectivity probe fails. Never depends on master
        Postgres (the challenge uses CHALLENGE_DATABASE_URL only).
        """

        if not self._initialized or self._closed:
            return False
        if not self._sqlite_path_writable():
            return False
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def _sqlite_path_writable(self) -> bool:
        """Return False when a file-backed SQLite parent is not writable.

        Checks directory existence, owner-write mode bit (fail-closed even when
        the process is root and `os.access` would still return True), and DAC
        access for non-root runtimes (real containers as appuser).
        """

        path = _sqlite_path_from_url(self.database_url)
        if path is None:
            return True  # in-memory or non-file URL
        parent = path.parent
        if not parent.exists() or not parent.is_dir():
            return False
        try:
            mode = parent.stat().st_mode
        except OSError:
            return False
        # No owner-write bit → treat as unwritable (root + 0o500 mounts).
        if not (mode & stat.S_IWUSR):
            return False
        # Non-root still subject to DAC.
        if os.geteuid() != 0 and not os.access(parent, os.W_OK):
            return False
        return True

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield an async SQLAlchemy session."""

        async with self._session_factory() as session:
            yield session

    async def session_dependency(self) -> AsyncIterator[AsyncSession]:
        """FastAPI dependency wrapper for request-scoped sessions."""

        async with self.session() as session:
            yield session


def _sqlite_path_from_url(database_url: str) -> Path | None:
    """Extract a filesystem path from a sqlite URL, if present."""

    prefixes = (
        "sqlite+aiosqlite:///",
        "sqlite:///",
    )
    for prefix in prefixes:
        if database_url.startswith(prefix):
            raw = database_url[len(prefix) :]
            # Absolute paths use four slashes form: sqlite+aiosqlite:////data/x
            if raw.startswith("/"):
                return Path(raw)
            if raw in {":memory:", ""}:
                return None
            return Path(raw)
    return None


__all__ = ["Base", "Database"]
