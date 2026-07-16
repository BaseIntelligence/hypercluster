"""Aggregation window, multi-hotkey ranking, self-deal damping, leaderboard.

Fulfills VAL-SCORE-008, 009, 010, 011, 012, 018, 022, 027, 029.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.db.models import Job, JobAttempt, Score, utc_now
from hypercluster.domain.aggregation import (
    SCORE_ROLES,
    HotkeyAggregate,
    apply_self_deal_damping,
    build_leaderboard,
    compute_raw_weights,
    finite_non_negative,
    list_scores_in_window,
    sanitize_weights_map,
    score_rows_bind_hotkey_role,
)
from hypercluster.domain.scoring_tee import persist_score_for_attempt
from hypercluster.settings import HyperSettings

# ----- helpers ---------------------------------------------------------------


def _hyper(**overrides: Any) -> HyperSettings:
    base: dict[str, Any] = {
        "tee_bonus_tdx": 1.08,
        "tee_bonus_tdx_gpu": 1.20,
        "efficiency_floor": 0.0,
        "score_window_attempts": 50,
        "self_deal_damping": 0.5,
    }
    base.update(overrides)
    return HyperSettings(**base)


def _job(*, hotkey: str, job_id: str | None = None) -> Job:
    return Job(
        id=job_id or str(uuid.uuid4()),
        submitter_hotkey=hotkey,
        status="succeeded",
        image_digest=("sha256:sim000000000000000000000000000000000000000000000000000000000001"),
        entrypoint_json=json.dumps(["python", "-m", "train"]),
        world_size=1,
        nnodes=1,
        nproc_per_node=1,
        backend="nccl",
        fabric_mode="auto",
        tee_mode="none",
        resource_json=json.dumps({"gpus": 1}),
        timeout_s=60,
    )


async def _seed_score(
    session: Any,
    *,
    hotkey: str,
    role: str,
    composite_efficiency: float,
    self_deal: bool = False,
    extra: dict[str, Any] | None = None,
) -> Score:
    """Persist a fully-gated score (composite ≈ efficiency * tee_bonus default 1)."""

    attempt_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    session.add(_job(hotkey=hotkey, job_id=job_id))
    session.add(
        JobAttempt(
            id=attempt_id,
            job_id=job_id,
            attempt_no=1,
            status="succeeded",
        )
    )
    await session.flush()
    details: dict[str, Any] = {}
    if self_deal:
        details["self_deal"] = True
    if extra:
        details.update(extra)
    row = await persist_score_for_attempt(
        session,
        attempt_id=attempt_id,
        hotkey=hotkey,
        role=role,
        correctness=1.0,
        efficiency=composite_efficiency,
        fabric_gate=1.0,
        proof=None,
        tee_mode="none",
        hyper=_hyper(),
        details=details or None,
    )
    return row


# ----- pure unit helpers -----------------------------------------------------


def test_score_roles_catalog_includes_demand_supply_joint() -> None:
    """VAL-SCORE-008: role set is demand|supply|joint."""

    assert "demand" in SCORE_ROLES
    assert "supply" in SCORE_ROLES
    assert "joint" in SCORE_ROLES


def test_sanitize_weights_map_rejects_nan_inf_negative() -> None:
    """VAL-SCORE-009 / VAL-SCORE-010: finite ≥ 0 only; empty burn-safe."""

    dirty = {
        "a": 1.5,
        "b": float("nan"),
        "c": float("inf"),
        "d": -3.0,
        "e": 0.0,
        "": 2.0,
    }
    clean = sanitize_weights_map(dirty)
    assert clean == {"a": 1.5, "e": 0.0}
    assert all(math.isfinite(v) and v >= 0.0 for v in clean.values())
    assert sanitize_weights_map({}) == {}
    assert sanitize_weights_map(None) == {}  # type: ignore[arg-type]


def test_self_deal_damping_reduces_mass_without_nan() -> None:
    """VAL-SCORE-012: soft self-deal penalty reduces mass, stays finite ≥0."""

    honest = apply_self_deal_damping(composite=10.0, self_deal=False, damping=0.5)
    collude = apply_self_deal_damping(composite=10.0, self_deal=True, damping=0.5)
    assert honest == pytest.approx(10.0)
    assert collude == pytest.approx(5.0)
    assert collude < honest
    assert math.isfinite(collude) and collude >= 0.0
    # damping 1.0 zeroes self-deal contribution; damping 0 keeps full mass
    assert apply_self_deal_damping(10.0, self_deal=True, damping=1.0) == 0.0
    assert apply_self_deal_damping(10.0, self_deal=True, damping=0.0) == 10.0
    # never produces NaN on weird inputs
    assert apply_self_deal_damping(float("nan"), self_deal=True, damping=0.5) == 0.0


def test_finite_non_negative_helper() -> None:
    assert finite_non_negative(1.2) == 1.2
    assert finite_non_negative(-1.0) == 0.0
    assert finite_non_negative(float("nan")) == 0.0
    assert finite_non_negative(float("inf")) == 0.0


# ----- VAL-SCORE-008 row binding ---------------------------------------------


@pytest.mark.asyncio
async def test_score_rows_bind_hotkey_and_role(settings_factory: Any, tmp_path: Path) -> None:
    """VAL-SCORE-008: scores bind hotkey + role; history lists role."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'role-bind.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    app = create_app(settings, hyper_settings=_hyper())
    hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            await _seed_score(session, hotkey=hotkey, role="demand", composite_efficiency=2.0)
            await _seed_score(session, hotkey=hotkey, role="supply", composite_efficiency=1.5)
            await session.commit()

            ok, missing = score_rows_bind_hotkey_role(
                await list_scores_in_window(session, window=50)
            )
            assert ok is True
            assert missing == []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/v1/scores/{hotkey}")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            items = body["items"]
            assert len(items) == 2
            roles = {item["role"] for item in items}
            assert roles == {"demand", "supply"}
            for item in items:
                assert item["hotkey"] == hotkey
                assert item["role"] in SCORE_ROLES


# ----- VAL-SCORE-009 / 011 multi-hotkey ranking ------------------------------


@pytest.mark.asyncio
async def test_multi_hotkey_ranking_reflects_composite_mass(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-SCORE-009 + VAL-SCORE-011: finite ≥0 map; heavier mass ranks higher."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rank-mass.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    hyper = _hyper(score_window_attempts=50)
    app = create_app(settings, hyper_settings=hyper)
    hotkey_a = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    hotkey_b = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            # A has more mass than B
            await _seed_score(session, hotkey=hotkey_a, role="demand", composite_efficiency=10.0)
            await _seed_score(session, hotkey=hotkey_a, role="demand", composite_efficiency=5.0)
            await _seed_score(session, hotkey=hotkey_b, role="demand", composite_efficiency=3.0)
            await session.commit()

            weights = await compute_raw_weights(session, hyper=hyper)
            assert all(math.isfinite(v) and v >= 0.0 for v in weights.values())
            assert weights[hotkey_a] > weights[hotkey_b]
            # M10 default: sum-normalize emission map (15+3=18 → unit sum).
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
            assert weights[hotkey_a] == pytest.approx(15.0 / 18.0)
            assert weights[hotkey_b] == pytest.approx(3.0 / 18.0)

            board = await build_leaderboard(session, hyper=hyper)
            assert board[0]["hotkey"] == hotkey_a
            assert board[1]["hotkey"] == hotkey_b
            # Leaderboard retains absolute aggregate mass for observability.
            assert board[0]["aggregate"] == pytest.approx(15.0)
            assert board[1]["aggregate"] == pytest.approx(3.0)
            assert board[0]["aggregate"] > board[1]["aggregate"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            lb = await client.get("/v1/leaderboard")
            assert lb.status_code == 200, lb.text
            rows = lb.json()["items"]
            assert len(rows) >= 2
            assert rows[0]["hotkey"] == hotkey_a
            assert rows[0]["aggregate"] >= rows[1]["aggregate"]
            preview = await client.get("/v1/weight-preview")
            assert preview.status_code == 200, preview.text
            wmap = preview.json()["weights"]
            assert wmap[hotkey_a] > wmap[hotkey_b]
            assert sum(float(v) for v in wmap.values()) == pytest.approx(1.0, abs=1e-6)
            assert all(math.isfinite(v) and v >= 0.0 for v in wmap.values())


# ----- VAL-SCORE-010 empty burn-safe -----------------------------------------


@pytest.mark.asyncio
async def test_empty_participation_burn_safe(settings_factory: Any, tmp_path: Path) -> None:
    """VAL-SCORE-010 + VAL-SCORE-029: vacant empty-safe, no NaN, no invented ranks."""

    from hypercluster.app import create_app
    from hypercluster.weights import get_weights

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'empty-burn.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    app = create_app(settings, hyper_settings=_hyper())

    async with app.router.lifespan_context(app):
        # No scores seeded — vacant first visit
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            lb = await client.get("/v1/leaderboard")
            assert lb.status_code == 200, lb.text
            body = lb.json()
            items = body.get("items")
            assert isinstance(items, list)
            assert items == []
            # no fabricated synthetic leaderboard of registry-only hotkeys
            assert "rank" not in body or body.get("count", 0) == 0

            missing = await client.get("/v1/scores/5CAbsentHotkeyDoesNotExist00000000000000000")
            assert missing.status_code == 200, missing.text
            mbody = missing.json()
            assert mbody["items"] == []
            assert mbody["count"] == 0

            preview = await client.get("/v1/weight-preview")
            assert preview.status_code == 200, preview.text
            wbody = preview.json()
            weights = wbody.get("weights") or wbody.get("map") or {}
            assert weights == {} or all(
                math.isfinite(float(v)) and float(v) >= 0.0 for v in weights.values()
            )
            # get_weights internal family
            raw = await get_weights()
            assert sanitize_weights_map(raw) == {}


# ----- VAL-SCORE-012 self-deal soft penalty ----------------------------------


@pytest.mark.asyncio
async def test_self_deal_soft_penalty_reduces_mass(settings_factory: Any, tmp_path: Path) -> None:
    """VAL-SCORE-012: same efficiency, self-deal mass < honest twin; finite ≥0."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'self-deal.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    hyper = _hyper(self_deal_damping=0.5, score_window_attempts=50)
    app = create_app(settings, hyper_settings=hyper)
    honest = "5DAAnrj7VHTznn2AaACRrN8iJZqK7PhB1aH6Yqz3G3eQnZf"
    collude = "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw"

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            await _seed_score(
                session,
                hotkey=honest,
                role="demand",
                composite_efficiency=8.0,
                self_deal=False,
            )
            await _seed_score(
                session,
                hotkey=collude,
                role="demand",
                composite_efficiency=8.0,
                self_deal=True,
            )
            await session.commit()

            weights = await compute_raw_weights(session, hyper=hyper)
            assert weights[collude] < weights[honest]
            # Absolute mass 4 honest-damped + 8 honest → unit shares 1/3 and 2/3.
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
            assert weights[collude] == pytest.approx(4.0 / 12.0)
            assert weights[honest] == pytest.approx(8.0 / 12.0)
            assert math.isfinite(weights[collude]) and weights[collude] >= 0.0


# ----- VAL-SCORE-022 window bounds -------------------------------------------


@pytest.mark.asyncio
async def test_score_window_bounds_contribution(settings_factory: Any, tmp_path: Path) -> None:
    """VAL-SCORE-022: only last N attempts contribute; old mass drops after recompute."""

    from datetime import timedelta

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'window.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    # Window of 2 keeps only the two newest scores.
    hyper = _hyper(score_window_attempts=2)
    app = create_app(settings, hyper_settings=hyper)
    hk = "5C4hrfj5Ji8aMa2ZxKs8n3tAVbCdEfGhIjKlMnOpQrStUvWx"

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            old = await _seed_score(session, hotkey=hk, role="demand", composite_efficiency=100.0)
            # Force old created_at into the past so window order is deterministic.
            old.created_at = utc_now() - timedelta(hours=3)
            mid = await _seed_score(session, hotkey=hk, role="demand", composite_efficiency=1.0)
            mid.created_at = utc_now() - timedelta(hours=2)
            await _seed_score(session, hotkey=hk, role="demand", composite_efficiency=2.0)
            await session.commit()

            window_rows = await list_scores_in_window(session, window=2)
            assert len(window_rows) == 2
            # Ancient 100-mass not in window → absolute mass is 1+2=3 not 103;
            # single-hotkey unit-sum emission is 1.0 (M10 default).
            from hypercluster.domain.aggregation import compute_mass_map

            mass = await compute_mass_map(session, hyper=hyper)
            assert mass[hk] == pytest.approx(3.0)
            weights = await compute_raw_weights(session, hyper=hyper)
            assert weights[hk] == pytest.approx(1.0)


# ----- VAL-SCORE-027 dual role same hotkey -----------------------------------


@pytest.mark.asyncio
async def test_dual_role_same_hotkey_aggregates_finite(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-SCORE-027: simultaneous demand+supply same hotkey → finite damped-safe mass."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'dual-role.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    hyper = _hyper()
    app = create_app(settings, hyper_settings=hyper)
    dual = "5HB2JzqvXuP8E3Wc9dY1mNrTsKqUvWxYzaB2cDefGhIjKlMn"

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            await _seed_score(session, hotkey=dual, role="demand", composite_efficiency=4.0)
            await _seed_score(session, hotkey=dual, role="supply", composite_efficiency=6.0)
            await session.commit()

            from hypercluster.domain.aggregation import compute_mass_map

            aggregates = await compute_raw_weights(session, hyper=hyper)
            assert dual in aggregates
            assert math.isfinite(aggregates[dual]) and aggregates[dual] >= 0.0
            # Dual-role absolute mass 4+6=10; unit-sum emission is 1.0.
            mass = await compute_mass_map(session, hyper=hyper)
            assert mass[dual] == pytest.approx(10.0)
            assert aggregates[dual] == pytest.approx(1.0)

            board = await build_leaderboard(session, hyper=hyper)
            assert len(board) == 1
            assert board[0]["hotkey"] == dual
            assert board[0]["aggregate"] == pytest.approx(10.0)
            # role breakdown visible for dual-role policy
            roles = board[0].get("roles") or {}
            assert roles.get("demand") == pytest.approx(4.0)
            assert roles.get("supply") == pytest.approx(6.0)


# ----- VAL-SCORE-018 leaderboard ordering ------------------------------------


@pytest.mark.asyncio
async def test_leaderboard_lists_composite_aggregates_descending(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-SCORE-018: GET /v1/leaderboard rows ordered desc by aggregate mass."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'lb-order.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    app = create_app(settings, hyper_settings=_hyper())
    h1 = "5E1RxL1vXqN7h9mT3bC2sA4dF6gH8jK0lM1nP2qR3sT4uV5w"
    h2 = "5E2GxL1vXqN7h9mT3bC2sA4dF6gH8jK0lM1nP2qR3sT4uV5x"
    h3 = "5E3HxL1vXqN7h9mT3bC2sA4dF6gH8jK0lM1nP2qR3sT4uV5y"

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            await _seed_score(session, hotkey=h2, role="demand", composite_efficiency=5.0)
            await _seed_score(session, hotkey=h1, role="demand", composite_efficiency=9.0)
            await _seed_score(session, hotkey=h3, role="supply", composite_efficiency=1.0)
            await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/v1/leaderboard")
            assert resp.status_code == 200, resp.text
            items = resp.json()["items"]
            assert [row["hotkey"] for row in items] == [h1, h2, h3]
            assert all(isinstance(row["aggregate"], (int, float)) for row in items)
            assert all(math.isfinite(row["aggregate"]) for row in items)


# ----- dataclass export smoke ------------------------------------------------


def test_hotkey_aggregate_dataclass() -> None:
    agg = HotkeyAggregate(
        hotkey="hk",
        aggregate=3.5,
        roles={"demand": 2.0, "supply": 1.5},
        score_count=2,
        self_deal_count=0,
    )
    public = agg.to_public(rank=1)
    assert public["rank"] == 1
    assert public["hotkey"] == "hk"
    assert public["aggregate"] == 3.5
