"""Raw hotkey weights for Base master aggregation.

Hypercluster never calls ``set_weights`` and never opens master Postgres.
Weights are the finite ≥0 hotkey map produced by windowed score aggregation
(architecture §10.2, VAL-SCORE-009/010/011). Empty participation is burn-safe
(``{}`` — never a NaN/poison payload).
"""

from __future__ import annotations

from typing import Any

from hypercluster.domain.aggregation import (
    build_leaderboard,
    compute_raw_weights,
    sanitize_weights_map,
)
from hypercluster.settings import HyperSettings, get_hyper_settings

# Optional process-level database handle set by create_app so the SDK
# zero-arg ``get_weights_fn`` can read the challenge SQLite store.
_database: Any | None = None
_hyper: HyperSettings | None = None


def bind_weights_runtime(
    database: Any | None,
    hyper: HyperSettings | None = None,
) -> None:
    """Register DB + product settings for process get_weights (app boot)."""

    global _database, _hyper
    _database = database
    _hyper = hyper


def clear_weights_runtime() -> None:
    """Drop bound runtime (tests)."""

    global _database, _hyper
    _database = None
    _hyper = None


async def get_weights() -> dict[str, float]:
    """Return raw hotkey → finite non-negative floats.

    Empty when no scores exist or the runtime database is unbound
    (VAL-SCORE-010 burn-safe).
    """

    return sanitize_weights_map(await load_raw_weights())


async def load_raw_weights(
    *,
    database: Any | None = None,
    hyper: HyperSettings | None = None,
) -> dict[str, float]:
    """Load windowed raw weights from the challenge DB."""

    db = database if database is not None else _database
    product = hyper if hyper is not None else (_hyper or get_hyper_settings())
    if db is None:
        return {}
    try:
        async with db.session() as session:
            return await compute_raw_weights(session, hyper=product)
    except Exception:  # noqa: BLE001 — never poison internal get_weights
        return {}


async def load_leaderboard(
    *,
    database: Any | None = None,
    hyper: HyperSettings | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregated leaderboard rows (empty-safe)."""

    db = database if database is not None else _database
    product = hyper if hyper is not None else (_hyper or get_hyper_settings())
    if db is None:
        return []
    try:
        async with db.session() as session:
            return await build_leaderboard(session, hyper=product, limit=limit)
    except Exception:  # noqa: BLE001
        return []


async def weight_preview_payload(
    *,
    database: Any | None = None,
    hyper: HyperSettings | None = None,
) -> dict[str, Any]:
    """Stable public shape for GET /v1/weight-preview."""

    weights = await load_raw_weights(database=database, hyper=hyper)
    return {
        "weights": weights,
        "count": len(weights),
        "empty": len(weights) == 0,
    }


__all__ = [
    "bind_weights_runtime",
    "clear_weights_runtime",
    "get_weights",
    "load_leaderboard",
    "load_raw_weights",
    "weight_preview_payload",
]
