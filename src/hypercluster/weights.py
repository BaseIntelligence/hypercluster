"""Raw hotkey weights for Base master aggregation.

Hypercluster never calls ``set_weights`` and never opens master Postgres.
Weights are the finite ≥0 hotkey map produced by windowed score aggregation
then M10 incentive sum-normalize (architecture §10.2, VAL-SCORE-009/010/011/016
+ VAL-WGT-010..014). Empty participation is burn-safe (``{}``).

Non-empty emission maps sum ≈ 1.0 within float tolerance. ``weight_snapshots``
retain pre-normalize absolute mass in ``raw_mass_json`` for audit.

``get_weights_fn`` and ``GET /v1/weight-preview`` share the same map family:
prefer the latest acked/pending snapshot when present, else live aggregation.
"""

from __future__ import annotations

from typing import Any

from hypercluster.domain.aggregation import (
    build_leaderboard,
    compute_mass_map,
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
    """Return hotkey → finite non-negative incentive floats.

    Empty when no scores exist or the runtime database is unbound
    (VAL-SCORE-010 / VAL-WGT-012 burn-safe). Non-empty maps are unit-sum under
    default M10 settings (VAL-WGT-014). Same family as weight-preview.
    """

    return sanitize_weights_map(await load_raw_weights())


async def load_raw_weights(
    *,
    database: Any | None = None,
    hyper: HyperSettings | None = None,
    prefer_snapshot: bool = True,
) -> dict[str, float]:
    """Load windowed raw weights from the challenge DB.

    When ``prefer_snapshot`` is true and a monochronic weight_snapshots row
    exists (pending or acked), return that map so get_weights / preview / push
    share one revision family (VAL-SCORE-016/028).
    """

    db = database if database is not None else _database
    product = hyper if hyper is not None else (_hyper or get_hyper_settings())
    if db is None:
        return {}
    try:
        async with db.session() as session:
            if prefer_snapshot:
                from hypercluster.weight_push import get_latest_snapshot

                snap = await get_latest_snapshot(session, prefer_acked=True)
                if snap is None:
                    # Fall back to any latest (including pending).
                    snap = await get_latest_snapshot(session, prefer_acked=False)
                if snap is not None:
                    # Prefer non-invalid window snapshots only.
                    if snap.push_status not in {"invalid_window", "rejected"}:
                        mapped = sanitize_weights_map(snap.weights_map())
                        if mapped:
                            return mapped
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
    """Stable public shape for GET /v1/weight-preview (VAL-SCORE-028).

    Returns pending or latest raw weight map when a snapshot exists; else the
    live aggregation window. Always finite ≥0; empty is burn-safe.
    """

    db = database if database is not None else _database
    product = hyper if hyper is not None else (_hyper or get_hyper_settings())
    snapshot_meta: dict[str, Any] | None = None
    weights: dict[str, float] = {}
    raw_mass: dict[str, float] = {}
    if db is not None:
        try:
            async with db.session() as session:
                from hypercluster.weight_push import get_latest_snapshot

                snap = await get_latest_snapshot(session, prefer_acked=False)
                if snap is not None and snap.push_status != "invalid_window":
                    weights = sanitize_weights_map(snap.weights_map())
                    raw_mass = sanitize_weights_map(snap.raw_mass_map())
                    snapshot_meta = {
                        "epoch": int(snap.epoch),
                        "revision": int(snap.revision),
                        "push_status": snap.push_status,
                        "payload_digest": snap.payload_digest,
                        "source": "snapshot",
                    }
                if not weights:
                    weights = await compute_raw_weights(session, hyper=product)
                    raw_mass = await compute_mass_map(session, hyper=product)
                    if snapshot_meta is None:
                        snapshot_meta = {"source": "aggregation"}
                elif not raw_mass:
                    # Legacy snapshot without raw_mass_json — live absolute mass.
                    raw_mass = await compute_mass_map(session, hyper=product)
        except Exception:  # noqa: BLE001
            weights = {}
            raw_mass = {}
    else:
        weights = await load_raw_weights(database=db, hyper=product)
    body: dict[str, Any] = {
        "weights": weights,
        "count": len(weights),
        "empty": len(weights) == 0,
        # VAL-WGT-013: expose pre-normalize mass when available for audit.
        "raw_mass": raw_mass,
    }
    if snapshot_meta is not None:
        body["snapshot"] = snapshot_meta
    return body


__all__ = [
    "bind_weights_runtime",
    "clear_weights_runtime",
    "get_weights",
    "load_leaderboard",
    "load_raw_weights",
    "weight_preview_payload",
]
