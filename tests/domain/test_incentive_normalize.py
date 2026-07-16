"""Incentive clamp + sum-normalize + snapshot raw mass + push (VAL-WGT-010..015, 022).

M10 default: finite ≥0 clamp; optional top-k / max-fraction; sum≈1 when mass>0;
empty → {}; weight_snapshots store normalized map + retain raw mass; mock-master
push payload is the unit-sum incentive map (VAL-WGT-015).
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from base.challenge_sdk.schemas import RawWeightPushRequest
from httpx import ASGITransport, AsyncClient

from hypercluster.db.models import Job, JobAttempt
from hypercluster.domain.incentive import (
    UNIT_SUM_TOLERANCE,
    apply_max_fraction,
    apply_top_k,
    clamp_mass_map,
    finalize_incentives,
    normalize_sum_to_unit,
    weight_sum,
)
from hypercluster.domain.scoring_tee import persist_score_for_attempt
from hypercluster.settings import HyperSettings
from hypercluster.sim.mock_master import app as mock_master_app
from hypercluster.sim.mock_master import configure_token, reset_store
from hypercluster.weight_push import (
    WeightPushClient,
    build_raw_weight_push_body,
    compute_payload_digest_for_body,
    create_pending_snapshot,
    get_snapshot_by_epoch_revision,
)
from hypercluster.weights import get_weights, weight_preview_payload

# Base-like ss58 keys for snapshot / get_weights integration.
HOTKEY_A = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
HOTKEY_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
HOTKEY_C = "5DAAnrj7VHTznn2AWG7Ym9yjT9LqVCzzSx5g5z3XU8MFGf6r"
TOKEN = "test-challenge-shared-token"
SLUG = "hypercluster"


def _hyper(**overrides: Any) -> HyperSettings:
    base: dict[str, Any] = {
        "tee_bonus_tdx": 1.08,
        "tee_bonus_tdx_gpu": 1.20,
        "efficiency_floor": 0.0,
        "score_window_attempts": 50,
        "self_deal_damping": 0.5,
        "incentive_sum_normalize": True,
        "weight_dust": 1e-12,
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
    efficiency: float,
    role: str = "demand",
    hyper: HyperSettings | None = None,
) -> None:
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
    await persist_score_for_attempt(
        session,
        attempt_id=attempt_id,
        hotkey=hotkey,
        role=role,
        correctness=1.0,
        efficiency=efficiency,
        fabric_gate=1.0,
        proof=None,
        tee_mode="none",
        hyper=hyper or _hyper(),
    )


# ----- pure unit: VAL-WGT-010 / 011 / 012 ------------------------------------


def test_clamp_drops_nan_inf_negative() -> None:
    """VAL-WGT-010: poison NaN/Inf/negative never reaches the weight map."""

    dirty = {
        "a": 2.0,
        "b": float("nan"),
        "c": float("inf"),
        "d": float("-inf"),
        "e": -5.0,
        "f": 0.0,
        "": 9.0,
    }
    clean = clamp_mass_map(dirty)
    assert clean == {"a": 2.0}
    assert all(math.isfinite(v) and v >= 0.0 for v in clean.values())


def test_sum_normalize_unit_sum_preserves_rank() -> None:
    """VAL-WGT-011: multi-hotkey positive mass normalizes to ~1.0 with rank order."""

    mass = {"a": 15.0, "b": 3.0, "c": 2.0}
    weights = normalize_sum_to_unit(mass)
    total = weight_sum(weights)
    assert total == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert weights["a"] > weights["b"] > weights["c"]
    assert weights["a"] == pytest.approx(15.0 / 20.0)
    assert weights["b"] == pytest.approx(3.0 / 20.0)
    assert weights["c"] == pytest.approx(2.0 / 20.0)


def test_empty_mass_burn_safe_empty_map() -> None:
    """VAL-WGT-012: no positive mass → {} (no fabricated equal weights)."""

    assert finalize_incentives({}) == {}
    assert finalize_incentives({"a": 0.0, "b": -1.0, "c": float("nan")}) == {}
    assert normalize_sum_to_unit({}) == {}
    assert normalize_sum_to_unit({"x": 0.0}) == {}


def test_finalize_clamps_then_normalizes() -> None:
    """VAL-WGT-010 + 011: finalize path is clamp → sum-normalize."""

    out = finalize_incentives(
        {"a": 10.0, "b": float("nan"), "c": -3.0, "d": 5.0},
        sum_normalize=True,
    )
    assert set(out) == {"a", "d"}
    assert weight_sum(out) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert out["a"] == pytest.approx(10.0 / 15.0)
    assert out["d"] == pytest.approx(5.0 / 15.0)


def test_finalize_can_skip_normalize_for_debug() -> None:
    """When sum-normalize is off, clamped absolute mass is returned."""

    out = finalize_incentives({"a": 10.0, "b": 5.0}, sum_normalize=False)
    assert out == {"a": 10.0, "b": 5.0}


def test_top_k_keeps_largest_then_renormalizes() -> None:
    """Optional simple top-k: keep largest, re-normalize (VAL-WGT-011)."""

    mass = {"a": 10.0, "b": 8.0, "c": 3.0, "d": 1.0}
    trimmed = apply_top_k(mass, k=2)
    assert set(trimmed) == {"a", "b"}
    out = finalize_incentives(mass, sum_normalize=True, top_k=2)
    assert set(out) == {"a", "b"}
    assert weight_sum(out) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert out["a"] > out["b"]


def test_max_fraction_clamps_share_then_renormalizes() -> None:
    """Optional max-fraction: cap absolute mass to max_fraction×total then re-norm.

    Simple policy (library/points-incentive.md): clamp each key's absolute mass
    to ``max_fraction * ΣM`` then sum-normalize. For 90:10 with max=0.5 →
    mass becomes 50:10 → unit shares 5/6 : 1/6.
    """

    mass = {"a": 90.0, "b": 10.0}
    capped = apply_max_fraction(mass, max_fraction=0.5)
    assert capped["a"] == pytest.approx(50.0)
    assert capped["b"] == pytest.approx(10.0)
    out = finalize_incentives(mass, sum_normalize=True, max_fraction=0.5)
    assert weight_sum(out) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert out["a"] == pytest.approx(50.0 / 60.0)
    assert out["b"] == pytest.approx(10.0 / 60.0)
    assert out["a"] > out["b"]


def test_dust_floor_drops_tiny_then_renormalizes() -> None:
    """Dust keys drop; remaining re-sum to 1 when mass remains."""

    mass = {"a": 1.0, "b": 1e-15, "c": 1.0}
    out = finalize_incentives(mass, sum_normalize=True, dust=1e-12)
    assert "b" not in out
    assert weight_sum(out) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert set(out) == {"a", "c"}


def test_four_factor_identity_untouched_by_incentive_module() -> None:
    """VAL-WGT-022: incentive module does not redefine composite product."""

    from hypercluster.domain.scoring import compute_four_factor

    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=1.0,
        tee_bonus=1.5,
    )
    assert breakdown.composite == pytest.approx(3.0)
    # Incentive only consumes mass maps; four-factor stays pure product.
    final = finalize_incentives({HOTKEY_A: breakdown.composite}, sum_normalize=True)
    assert final[HOTKEY_A] == pytest.approx(1.0)


# ----- integration: VAL-WGT-013 / 014 ----------------------------------------


@pytest.mark.asyncio
async def test_compute_raw_weights_unit_sum_default(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-WGT-011/014: emission path normalize sum ≈ 1 when multi-hotkey mass."""

    from hypercluster.app import create_app
    from hypercluster.domain.aggregation import compute_mass_map, compute_raw_weights

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'inc-unit.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = _hyper()
    app = create_app(settings, hyper_settings=hyper)

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            await _seed_score(session, hotkey=HOTKEY_A, efficiency=15.0, hyper=hyper)
            await _seed_score(session, hotkey=HOTKEY_B, efficiency=5.0, hyper=hyper)
            await session.commit()

            mass = await compute_mass_map(session, hyper=hyper)
            assert mass[HOTKEY_A] == pytest.approx(15.0)
            assert mass[HOTKEY_B] == pytest.approx(5.0)

            weights = await compute_raw_weights(session, hyper=hyper)
            assert weight_sum(weights) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
            assert weights[HOTKEY_A] == pytest.approx(0.75)
            assert weights[HOTKEY_B] == pytest.approx(0.25)
            assert weights[HOTKEY_A] > weights[HOTKEY_B]


@pytest.mark.asyncio
async def test_weight_snapshot_stores_normalized_and_raw_mass(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-WGT-013: snapshot weights unit-sum; raw_mass retained for audit."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'inc-snap.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = _hyper()
    app = create_app(settings, hyper_settings=hyper)

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            await _seed_score(session, hotkey=HOTKEY_A, efficiency=9.0, hyper=hyper)
            await _seed_score(session, hotkey=HOTKEY_B, efficiency=3.0, hyper=hyper)
            await session.commit()

        async with database.session() as session:
            snap = await create_pending_snapshot(
                session,
                challenge_slug=SLUG,
                epoch=99,
                hyper=hyper,
            )
            wmap = snap.weights_map()
            assert weight_sum(wmap) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
            assert wmap[HOTKEY_A] == pytest.approx(0.75)
            assert wmap[HOTKEY_B] == pytest.approx(0.25)

            raw = snap.raw_mass_map()
            assert raw[HOTKEY_A] == pytest.approx(9.0)
            assert raw[HOTKEY_B] == pytest.approx(3.0)
            # to_dict exposes both for audit (VAL-WGT-013).
            public = snap.to_dict()
            assert "weights" in public
            assert "raw_mass" in public
            assert weight_sum(public["weights"]) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
            assert public["raw_mass"][HOTKEY_A] == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_weight_preview_and_get_weights_unit_sum(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-WGT-014: preview + get_weights agree on unit-sum family when non-empty."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'inc-preview.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = _hyper()
    app = create_app(settings, hyper_settings=hyper)

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            await _seed_score(session, hotkey=HOTKEY_A, efficiency=6.0, hyper=hyper)
            await _seed_score(session, hotkey=HOTKEY_B, efficiency=2.0, hyper=hyper)
            await _seed_score(session, hotkey=HOTKEY_C, efficiency=2.0, hyper=hyper)
            await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            preview = await client.get("/v1/weight-preview")
            assert preview.status_code == 200, preview.text
            pmap = preview.json()["weights"]
            assert weight_sum(pmap) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
            assert all(math.isfinite(float(v)) and float(v) >= 0.0 for v in pmap.values())

        gw = await get_weights()
        assert weight_sum(gw) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
        # Same family (both unit-sum, same ranking).
        assert pmap[HOTKEY_A] == pytest.approx(gw[HOTKEY_A], abs=1e-9)
        assert pmap[HOTKEY_A] > pmap[HOTKEY_B]

        payload = await weight_preview_payload(database=database, hyper=hyper)
        assert weight_sum(payload["weights"]) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)


@pytest.mark.asyncio
async def test_empty_window_preview_get_weights_empty(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-WGT-012 surface: empty participation → {} on preview/get_weights."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'inc-empty.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    app = create_app(settings, hyper_settings=_hyper())

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            preview = await client.get("/v1/weight-preview")
            assert preview.status_code == 200
            assert preview.json()["weights"] == {}

        assert await get_weights() == {}


@pytest.mark.asyncio
async def test_poisoned_composites_never_yield_nan_weights(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-WGT-010: clamp at finalize so emission path stays finite ≥0."""

    from hypercluster.domain.incentive import finalize_incentives

    poisoned = {
        HOTKEY_A: float("nan"),
        HOTKEY_B: float("inf"),
        HOTKEY_C: -100.0,
    }
    out = finalize_incentives(poisoned, sum_normalize=True)
    assert out == {}
    # With one good key mixed in:
    mixed = {**poisoned, "5Gn8X3uE7vKqW9mPpR4sLtY2aBcDeFgHiJkLmNoPqRsTuVwXy": 4.0}
    out2 = finalize_incentives(mixed, sum_normalize=True)
    assert weight_sum(out2) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert all(math.isfinite(v) and v >= 0.0 for v in out2.values())


# ----- push payload: VAL-WGT-015 ---------------------------------------------


def test_build_push_body_normalizes_absolute_mass() -> None:
    """VAL-WGT-015: raw-weights builder coerces absolute mass → unit-sum on egress."""

    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).replace(microsecond=0)
    expires = now + timedelta(seconds=300)
    absolute = {HOTKEY_A: 15.0, HOTKEY_B: 5.0}
    payload, raw = build_raw_weight_push_body(
        challenge_slug=SLUG,
        epoch=77,
        revision=1,
        weights=absolute,
        nonce="n-unit-sum-push",
        computed_at=now,
        expires_at=expires,
    )
    body = json.loads(raw)
    wmap = {str(k): float(v) for k, v in body["weights"].items()}
    assert weight_sum(wmap) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert wmap[HOTKEY_A] == pytest.approx(0.75)
    assert wmap[HOTKEY_B] == pytest.approx(0.25)
    # Digest matches independent recompute over the normalized body.
    expected = compute_payload_digest_for_body(
        {k: v for k, v in body.items() if k != "payload_digest"}
    )
    assert payload.payload_digest == expected
    # Absolute mass must not appear as total\approx20 on the wire.
    assert weight_sum(wmap) != pytest.approx(20.0)


@pytest.mark.asyncio
async def test_push_to_mock_master_uses_normalized_weights(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-WGT-015: POST raw-weights body is unit-sum; digest matches; acked."""

    from hypercluster.app import create_app

    reset_store()
    configure_token(TOKEN)

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'inc-push-norm.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = _hyper()
    app = create_app(settings, hyper_settings=hyper)

    transport = httpx.ASGITransport(app=mock_master_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://mock-master.test"
    ) as master_http:
        async with app.router.lifespan_context(app):
            db = app.state.database
            async with db.session() as session:
                # Absolute mass 12 + 4 = 16; unit shares must be 0.75 / 0.25.
                await _seed_score(session, hotkey=HOTKEY_A, efficiency=12.0, hyper=hyper)
                await _seed_score(session, hotkey=HOTKEY_B, efficiency=4.0, hyper=hyper)
                await session.commit()

            client = WeightPushClient(
                database=db,
                challenge_slug=SLUG,
                master_base_url="http://mock-master.test",
                shared_token=TOKEN,
                hyper=hyper,
                http_client=master_http,
            )
            result = await client.push_once(epoch=515)
            assert result.status == "acknowledged", result.error
            assert result.push_status == "acked"
            assert result.payload_digest
            assert len(result.payload_digest) == 64

            # Capture from mock-master debug store (wire payload weights).
            listed = await master_http.get(f"/internal/v1/challenges/{SLUG}/raw-weights")
            assert listed.status_code == 200, listed.text
            items = listed.json()["items"]
            assert items, "mock-master must store accepted push"
            pushed = items[-1]
            wire_weights = {str(k): float(v) for k, v in pushed["weights"].items()}
            assert weight_sum(wire_weights) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
            assert wire_weights[HOTKEY_A] == pytest.approx(0.75)
            assert wire_weights[HOTKEY_B] == pytest.approx(0.25)
            # Absolute mass must not leak as the egress map sum.
            assert weight_sum(wire_weights) != pytest.approx(16.0)
            assert pushed["payload_digest"] == result.payload_digest
            assert int(pushed["epoch"]) == 515

            # Local snapshot: monochronic, unit-sum weights, raw mass retained.
            async with db.session() as session:
                row = await get_snapshot_by_epoch_revision(
                    session, epoch=515, revision=result.revision
                )
                assert row is not None
                assert row.push_status == "acked"
                assert row.payload_digest == result.payload_digest
                assert weight_sum(row.weights_map()) == pytest.approx(
                    1.0, abs=UNIT_SUM_TOLERANCE
                )
                raw_mass = row.raw_mass_map()
                assert raw_mass[HOTKEY_A] == pytest.approx(12.0)
                assert raw_mass[HOTKEY_B] == pytest.approx(4.0)
                # Canonical submit bytes recompute to same digest (VAL-WGT-015).
                if row.canonical_payload:
                    reparsed = RawWeightPushRequest.model_validate_json(row.canonical_payload)
                    assert reparsed.payload_digest == result.payload_digest
                    assert weight_sum(reparsed.weights) == pytest.approx(
                        1.0, abs=UNIT_SUM_TOLERANCE
                    )

            # Never challenge set_weights — push client has no chain setter.
            assert not hasattr(client, "set_weights")
