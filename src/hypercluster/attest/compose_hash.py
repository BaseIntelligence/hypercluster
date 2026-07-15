"""Deterministic app-compose hashing for offline TEE golden CLI (VAL-TEE-010).

Challenge-owned offline mixer: SHA-256 of raw file bytes (no timestamps, no
host-specific metadata). Output form matches the offline allowlist:
``sha256:<64-hex>``.

This is intentionally offline and pure — no dstack network, no docker daemon —
so CI stays green without hardware.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Pinned default golden relative path under tests/fixtures/tee (repo checkout).
GOLDEN_COMPOSE_RELPATH = "tests/fixtures/tee/golden_compose.yml"
GOLDEN_COMPOSE_HASH_RELPATH = "tests/fixtures/tee/golden_compose.sha256"


def hash_compose_bytes(raw: bytes) -> str:
    """Return ``sha256:<hex>`` for exact compose file bytes."""

    digest = hashlib.sha256(raw).hexdigest()
    return f"sha256:{digest}"


def hash_compose_file(path: str | Path) -> str:
    """Read compose file as raw bytes and return stable compose hash.

    Path is read once in binary mode so newline modes on disk are fixed for a
    given committed fixture. Two successive calls on the same file yield the
    same string (VAL-TEE-010).
    """

    p = Path(path)
    raw = p.read_bytes()
    return hash_compose_bytes(raw)


def load_golden_hash_file(path: str | Path) -> str:
    """Load a one-line ``sha256:...`` golden fixture; strip whitespace only."""

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"golden hash file empty: {path}")
    # Accept bare 64-hex or already-prefixed sha256: form.
    line = text.splitlines()[0].strip()
    if line.startswith("sha256:"):
        return line
    if len(line) == 64 and all(c in "0123456789abcdef" for c in line.lower()):
        return f"sha256:{line.lower()}"
    raise ValueError(f"golden hash file unparseable: {path!s}: {line!r}")


def ensure_golden_matches(
    compose_path: str | Path,
    golden_hash_path: str | Path,
) -> str:
    """Hash compose file and assert equality with golden hash file; return hash."""

    got = hash_compose_file(compose_path)
    expected = load_golden_hash_file(golden_hash_path)
    if got != expected:
        raise ValueError(
            f"compose-hash drift: got={got} expected={expected} "
            f"compose={compose_path} golden={golden_hash_path}"
        )
    return got


__all__ = [
    "GOLDEN_COMPOSE_HASH_RELPATH",
    "GOLDEN_COMPOSE_RELPATH",
    "ensure_golden_matches",
    "hash_compose_bytes",
    "hash_compose_file",
    "load_golden_hash_file",
]
