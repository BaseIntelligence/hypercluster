"""VAL-WGT-005 / 006 / 007: public points balance, list, and history routes.

M10 API surfaces:
- GET /v1/points/{hotkey} balance (never-seen → 0, empty-safe, never 500)
- GET /v1/points list balances (empty DB safe; seeded multi-hotkey)
- GET /v1/points/{hotkey}/history ordered ledger with attempt_id/score_id on earn
- No secrets / tokens / set_weights surface in responses
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.app import create_app
from hypercluster.db.models import PointsLedger, Score, utc_now
from hypercluster.domain.points import REASON_SCORE_EARN
from hypercluster.settings import HyperSettings

HOTKEY_A = "points-api-hotkey-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HOTKEY_B = "points-api-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
HOTKEY_NEVER = "points-api-hotkey-never-seen-zzzzzzzzzzzzzzzzzzzzzz"
SECRET_MARKERS = (
    "shared_token",
    "CHALLENGE_SHARED_TOKEN",
    "private_key",
    "BEGIN PRIVATE",
    "set_weights",
    "password",
    "api_key",
    "Authorization",
)


@pytest.fixture
async def points_client(settings_factory, tmp_path) -> AsyncIterator[tuple[AsyncClient, Any]]:
    """ASGI client with lifespan + app handle for DB seeding via state.database."""

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'points-api.sqlite3'}",
        shared_token="points-api-test-token-NOT-FOR-RESPONSES",
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        points_enabled=True,
        points_scale=1.0,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, app


def _assert_no_secrets(payload: Any) -> None:
    blob = json.dumps(payload) if not isinstance(payload, str) else payload
    lower = blob.lower()
    for marker in SECRET_MARKERS:
        assert marker.lower() not in lower, f"secret/forbidden marker {marker!r} in {blob[:400]}"


def _score(
    *,
    hotkey: str = HOTKEY_A,
    composite: float = 2.5,
    attempt_id: str | None = None,
    score_id: str | None = None,
    role: str = "demand",
) -> Score:
    return Score(
        id=score_id or str(uuid.uuid4()),
        attempt_id=attempt_id or str(uuid.uuid4()),
        hotkey=hotkey,
        role=role,
        correctness=1.0,
        efficiency=float(composite),
        fabric_gate=1.0,
        tee_bonus=1.0,
        composite=float(composite),
        details_json=None,
        created_at=utc_now(),
    )


async def _seed_earn(
    app: Any,
    *,
    hotkey: str,
    composite: float,
    attempt_id: str | None = None,
) -> Score:
    """Persist balance + ledger earn rows without full Job/Attempt FK graph."""

    from hypercluster.domain.points import (
        REASON_SCORE_EARN as _R,
    )
    from hypercluster.domain.points import (
        _upsert_balance,
        compute_score_earn_delta,
    )

    database = app.state.database
    score = _score(hotkey=hotkey, composite=composite, attempt_id=attempt_id)
    async with database.session() as session:
        hyper = app.state.hyper_settings
        scale = float(hyper.points_scale)
        delta = compute_score_earn_delta(composite, scale=scale)
        bal_after = await _upsert_balance(session, hotkey=hotkey, delta=delta)
        row = PointsLedger(
            id=str(uuid.uuid4()),
            hotkey=hotkey,
            role="demand",
            delta=float(delta),
            balance_after=float(bal_after),
            reason=_R,
            score_id=score.id,
            attempt_id=score.attempt_id,
            details_json=json.dumps({"composite": composite, "scale": scale}),
            created_at=utc_now(),
        )
        session.add(row)
        await session.commit()
    return score


# ---------------------------------------------------------------------------
# VAL-WGT-005: balance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_points_balance_never_seen_is_zero_empty_safe(
    points_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-WGT-005: unknown hotkey returns balance 0, not 404/500."""

    client, _app = points_client
    resp = await client.get(f"/v1/points/{HOTKEY_NEVER}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["hotkey"] == HOTKEY_NEVER
    assert float(body["balance"]) == 0.0
    assert math.isfinite(float(body["balance"]))
    assert float(body["balance"]) >= 0.0
    _assert_no_secrets(body)


@pytest.mark.asyncio
async def test_points_balance_matches_ledger_sum(
    points_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-WGT-005: known hotkey balance matches ledger sum of deltas."""

    client, app = points_client
    s1 = await _seed_earn(app, hotkey=HOTKEY_A, composite=2.0)
    s2 = await _seed_earn(app, hotkey=HOTKEY_A, composite=1.5)
    expected = 2.0 + 1.5

    resp = await client.get(f"/v1/points/{HOTKEY_A}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["hotkey"] == HOTKEY_A
    assert float(body["balance"]) == pytest.approx(expected)
    assert math.isfinite(float(body["balance"]))
    _assert_no_secrets(body)
    # Forensic: seed kept score/attempt refs in ledger; balance body itself public-safe.
    assert s1.attempt_id and s2.attempt_id


# ---------------------------------------------------------------------------
# VAL-WGT-006: list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_points_list_empty_db_safe(points_client: tuple[AsyncClient, Any]) -> None:
    """VAL-WGT-006: empty DB → items=[] count=0, not crash."""

    client, _app = points_client
    resp = await client.get("/v1/points")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"] == []
    assert body["count"] == 0
    assert body.get("empty") is True
    _assert_no_secrets(body)


@pytest.mark.asyncio
async def test_points_list_includes_seeded_hotkeys(
    points_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-WGT-006: multi-hotkey earns surface on list with stable shape."""

    client, app = points_client
    await _seed_earn(app, hotkey=HOTKEY_A, composite=3.0)
    await _seed_earn(app, hotkey=HOTKEY_B, composite=1.25)

    resp = await client.get("/v1/points")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] >= 2
    assert body.get("empty") is False
    by_hk = {row["hotkey"]: float(row["balance"]) for row in body["items"]}
    assert HOTKEY_A in by_hk
    assert HOTKEY_B in by_hk
    assert by_hk[HOTKEY_A] == pytest.approx(3.0)
    assert by_hk[HOTKEY_B] == pytest.approx(1.25)
    for row in body["items"]:
        assert "hotkey" in row and "balance" in row
        assert math.isfinite(float(row["balance"]))
        assert float(row["balance"]) >= 0.0
    _assert_no_secrets(body)


# ---------------------------------------------------------------------------
# VAL-WGT-007: history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_points_history_empty_safe(points_client: tuple[AsyncClient, Any]) -> None:
    """VAL-WGT-007: never-seen history → empty items, not 500."""

    client, _app = points_client
    resp = await client.get(f"/v1/points/{HOTKEY_NEVER}/history")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["hotkey"] == HOTKEY_NEVER
    assert body["items"] == []
    assert body["count"] == 0
    _assert_no_secrets(body)


@pytest.mark.asyncio
async def test_points_history_contains_earn_forensics(
    points_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-WGT-007: history includes score_earn rows with attempt_id/score_id."""

    client, app = points_client
    score = await _seed_earn(app, hotkey=HOTKEY_A, composite=4.0)
    resp = await client.get(f"/v1/points/{HOTKEY_A}/history")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["hotkey"] == HOTKEY_A
    assert body["count"] >= 1
    assert body["items"]
    earn = body["items"][0]
    assert earn["reason"] == REASON_SCORE_EARN
    assert earn["attempt_id"] == score.attempt_id
    assert earn["score_id"] == score.id
    assert float(earn["delta"]) == pytest.approx(4.0)
    assert float(earn["balance_after"]) == pytest.approx(4.0)
    _assert_no_secrets(body)


@pytest.mark.asyncio
async def test_points_history_ordered_newest_first(
    points_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-WGT-007: ledger history is ordered (newest first)."""

    client, app = points_client
    first = await _seed_earn(app, hotkey=HOTKEY_A, composite=1.0)
    second = await _seed_earn(app, hotkey=HOTKEY_A, composite=2.0)
    resp = await client.get(f"/v1/points/{HOTKEY_A}/history")
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert len(items) >= 2
    attempt_ids = [row["attempt_id"] for row in items]
    assert second.attempt_id in attempt_ids
    assert first.attempt_id in attempt_ids
    # Newest first: second earn comes before first.
    assert attempt_ids.index(second.attempt_id) < attempt_ids.index(first.attempt_id)


@pytest.mark.asyncio
async def test_points_routes_never_surface_set_weights(
    points_client: tuple[AsyncClient, Any],
) -> None:
    """Contract: points GET family does not expose set_weights or secrets."""

    client, app = points_client
    await _seed_earn(app, hotkey=HOTKEY_A, composite=1.0)
    for path in (
        f"/v1/points/{HOTKEY_A}",
        "/v1/points",
        f"/v1/points/{HOTKEY_A}/history",
    ):
        resp = await client.get(path)
        assert resp.status_code == 200, path
        _assert_no_secrets(resp.json())
        assert "set_weights" not in resp.text
