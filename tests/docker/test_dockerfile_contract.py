"""Static Dockerfile contract checks (VAL-SCAF-018 / 020) without needing docker daemon."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert DOCKERFILE.is_file(), f"Dockerfile missing at {DOCKERFILE}"
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_defines_healthcheck_loopback_status_ok(dockerfile_text: str) -> None:
    """VAL-SCAF-018: HEALTHCHECK probes loopback /health and gates on status==ok."""

    assert "HEALTHCHECK" in dockerfile_text
    assert "http://127.0.0.1:8000/health" in dockerfile_text
    assert "status" in dockerfile_text
    assert "ok" in dockerfile_text
    # Template cadence.
    assert "--interval=30s" in dockerfile_text
    assert "--timeout=5s" in dockerfile_text
    assert "--start-period=10s" in dockerfile_text
    assert "--retries=3" in dockerfile_text


def test_dockerfile_exposes_cmd_and_nonroot_user(dockerfile_text: str) -> None:
    """VAL-SCAF-020: EXPOSE 8000, non-root appuser, uvicorn package app CMD."""

    assert "EXPOSE 8000" in dockerfile_text
    assert "appuser" in dockerfile_text
    assert "USER appuser" in dockerfile_text
    assert "uvicorn" in dockerfile_text
    assert "hypercluster.app:app" in dockerfile_text
    assert '"--host", "0.0.0.0"' in dockerfile_text or "--host 0.0.0.0" in dockerfile_text
    assert '"--port", "8000"' in dockerfile_text or "--port 8000" in dockerfile_text


def test_dockerfile_defaults_sqlite_data_path(dockerfile_text: str) -> None:
    """Image default CHALLENGE_DATABASE_URL is absolute SQLite under /data."""

    assert "sqlite+aiosqlite:////data/challenge.sqlite3" in dockerfile_text
    assert "CHALLENGE_PORT=8000" in dockerfile_text
