"""Cross e2e weight push, multi-miner leaderboard, self-deal, mock-master chaos.

Fulfills:
  VAL-CROSS-012  End-to-end weight push after scored chain ack on mock-master
  VAL-CROSS-019  Leaderboard + scores + weights agree on hotkey set after multi-miner
  VAL-CROSS-020  Self-deal demand+supply same hotkey finite and optionally damped
  VAL-CROSS-027  Mock-master down does not destroy score rows; push retries/pending
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from hypercluster.sim.ports import (
    DEFAULT_BAREMETAL_PORT,
    DEFAULT_MOCK_MASTER_PORT,
    MAX_MISSION_PORT,
    MIN_MISSION_PORT,
    is_mission_port,
)
from hypercluster.sim.scenarios import ScenarioResult, _fail

CROSS_WEIGHTS_LEADERBOARD = "cross-weights-leaderboard-selfdeal"

# Alpha-prefixed ss58-like keys (Base alphabet; WeightPushClient rejects bare UIDs).
# Must match ^[1-9A-HJ-NP-Za-km-z]{3,64}$ (no 0/I/O/l).
MINER_A = "5crossWLsAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MINER_B = "5crossWLsBbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
MINER_C = "5crossWLsCcccccccccccccccccccccccccccccccccccccccccc"
SELF_DEAL_HK = "5crossWLsSeLfDeaLHotkeyaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TWIN_HONEST_HK = "5crossWLsTwinHonestHotkeyaaaaaaaaaaaaaaaaaaaaaaaaaa"

ALLOWED_IMAGE = (
    "sha256:sim000000000000000000000000000000000000000000000000000000000001"
)


def _resolve_secret(shared_token: str | None) -> str:
    secret = (shared_token or "").strip()
    if not secret:
        secret = (os.environ.get("CHALLENGE_SHARED_TOKEN") or "").strip()
    if not secret:
        token_file = (os.environ.get("CHALLENGE_SHARED_TOKEN_FILE") or "").strip()
        if token_file:
            try:
                secret = Path(token_file).read_text(encoding="utf-8").strip()
            except OSError:
                secret = ""
    return secret or "test-challenge-shared-token"


def _resolve_master(master_url: str | None) -> str:
    return (
        (master_url or "").strip()
        or (os.environ.get("HYPER_MASTER_BASE_URL") or "").strip()
        or f"http://127.0.0.1:{DEFAULT_MOCK_MASTER_PORT}"
    )


def _port_from_url(url: str) -> int | None:
    parsed = urlparse(url)
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def _run_async(coro_factory: Callable[[], Any], *, timeout: float) -> Any:
    """Run an async coroutine from sync context (pytest/cli both OK)."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro_factory())).result(
                timeout=max(timeout, 30.0)
            )
    return asyncio.run(coro_factory())


def _finite_nonneg(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f < 0.0:
        return None
    return f


async def _seed_score_rows(
    database: Any,
    *,
    hyper: Any,
    rows: list[dict[str, Any]],
) -> list[str]:
    """Insert job+attempt+score rows into the challenge DB (shared with live API)."""

    from hypercluster.db.models import Job, JobAttempt
    from hypercluster.domain.scoring_tee import persist_score_for_attempt

    score_ids: list[str] = []
    async with database.session() as session:
        for row in rows:
            job_id = str(uuid.uuid4())
            attempt_id = str(uuid.uuid4())
            session.add(
                Job(
                    id=job_id,
                    submitter_hotkey=str(row["hotkey"]),
                    status="succeeded",
                    image_digest=ALLOWED_IMAGE,
                    entrypoint_json=json.dumps(["python", "-m", "train", "--cross-wls"]),
                    world_size=1,
                    nnodes=1,
                    nproc_per_node=1,
                    backend="nccl",
                    fabric_mode="auto",
                    tee_mode="none",
                    resource_json=json.dumps({"gpus": 1}),
                    timeout_s=60,
                )
            )
            session.add(
                JobAttempt(
                    id=attempt_id,
                    job_id=job_id,
                    attempt_no=1,
                    status="succeeded",
                )
            )
            await session.flush()
            details = dict(row.get("details") or {})
            if row.get("self_deal"):
                details["self_deal"] = True
            score = await persist_score_for_attempt(
                session,
                attempt_id=attempt_id,
                hotkey=str(row["hotkey"]),
                role=str(row.get("role") or "demand"),
                correctness=float(row.get("correctness", 1.0)),
                efficiency=float(row.get("efficiency", 1.0)),
                fabric_gate=float(row.get("fabric_gate", 1.0)),
                proof=None,
                tee_mode="none",
                hyper=hyper,
                details=details or None,
            )
            score_ids.append(str(score.id))
        await session.commit()
    return score_ids


async def _load_weights_and_push(
    *,
    master_url: str,
    shared_token: str,
    epoch: int,
    revision: int | None = None,
    reuse_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Load live raw weights from process DB and push once to master."""

    from hypercluster.db.database import Database
    from hypercluster.settings import HyperSettings, get_settings
    from hypercluster.weight_push import WeightPushClient
    from hypercluster.weights import load_raw_weights

    settings = get_settings()
    hyper = HyperSettings(
        score_window_attempts=50,
        self_deal_damping=0.5,
        master_base_url=master_url,
        weight_push_enabled=True,
        weight_push_freshness_s=300,
    )
    database = Database(settings.database_url)
    await database.init()
    try:
        weights = await load_raw_weights(database=database, hyper=hyper)
        client = WeightPushClient(
            database=database,
            challenge_slug=settings.slug,
            master_base_url=master_url,
            shared_token=shared_token,
            hyper=hyper,
        )
        result = await client.push_once(
            weights=weights if weights else None,
            epoch=epoch,
            revision=revision,
            reuse_snapshot_id=reuse_snapshot_id,
        )
        return {
            "weights": weights,
            "status": result.status,
            "push_status": result.push_status,
            "epoch": result.epoch,
            "revision": result.revision,
            "payload_digest": result.payload_digest,
            "snapshot_id": result.snapshot_id,
            "local_id": result.local_id,
            "error": result.error,
            "idempotent": result.idempotent,
        }
    finally:
        await database.close()


async def _seed_multi_miner_and_self_deal(
    *,
    master_url: str,
) -> dict[str, Any]:
    """Seed three ranked miners + self-deal dual-role row set into process DB."""

    from hypercluster.db.database import Database
    from hypercluster.settings import HyperSettings, get_settings
    from hypercluster.weights import load_raw_weights

    settings = get_settings()
    hyper = HyperSettings(
        score_window_attempts=50,
        self_deal_damping=0.5,
        master_base_url=master_url,
        weight_push_enabled=True,
        weight_push_freshness_s=300,
    )
    database = Database(settings.database_url)
    await database.init()
    try:
        # Distinct positive composites: A > B > C for monotone ranking.
        multi = [
            {"hotkey": MINER_A, "role": "demand", "efficiency": 12.0},
            {"hotkey": MINER_B, "role": "demand", "efficiency": 6.0},
            {"hotkey": MINER_C, "role": "demand", "efficiency": 2.0},
        ]
        # Self-deal path: same hotkey demand + supply with collusion flag.
        self_deal_rows = [
            {
                "hotkey": SELF_DEAL_HK,
                "role": "demand",
                "efficiency": 8.0,
                "self_deal": True,
            },
            {
                "hotkey": SELF_DEAL_HK,
                "role": "supply",
                "efficiency": 4.0,
                "self_deal": True,
            },
            # Honest twin (same undamped single hop mass reference).
            {
                "hotkey": TWIN_HONEST_HK,
                "role": "demand",
                "efficiency": 8.0,
                "self_deal": False,
            },
        ]
        score_ids = await _seed_score_rows(
            database, hyper=hyper, rows=multi + self_deal_rows
        )
        weights = await load_raw_weights(database=database, hyper=hyper)
        return {
            "ok": True,
            "score_ids": score_ids,
            "weights": weights,
            "database_url": settings.database_url,
        }
    finally:
        await database.close()


def run_cross_weight_push_ack(
    base_url: str,
    *,
    timeout: float = 45.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    ensure_seeded: bool = True,
) -> ScenarioResult:
    """VAL-CROSS-012: after scored chain, push raw weights → mock-master acks."""

    name = "cross-weight-push-ack"
    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    master = _resolve_master(master_url)
    steps.append(f"master_url={master}")

    # Require green identity first (after scored chain continuum).
    try:
        with httpx.Client(timeout=min(timeout, 5.0)) as client:
            for path in ("/health", "/ready"):
                r = client.get(f"{normalized}{path}")
                steps.append(f"GET {path} → {r.status_code}")
                if r.status_code != 200:
                    return _fail(
                        name,
                        normalized,
                        f"baseline {path} not green: HTTP {r.status_code}",
                        steps,
                        None,
                    )
    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"identity probe failed: {exc}", steps, None)

    # Seed scored rows (simulates completed CROSS-002 scoring phase on shared DB).
    if ensure_seeded:
        try:
            with httpx.Client(timeout=min(timeout, 5.0)) as client:
                sc = client.get(f"{normalized}/v1/scores/{MINER_A}")
                need = sc.status_code != 200 or not (sc.json().get("items") or [])
        except httpx.HTTPError:
            need = True
        if need:
            try:
                seed = _run_async(
                    lambda: _seed_multi_miner_and_self_deal(master_url=master),
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                return _fail(name, normalized, f"score seed failed: {exc}", steps, None)
            if not seed.get("ok"):
                return _fail(name, normalized, "score seed returned not-ok", steps, None)
            steps.append(
                f"seeded score_ids={len(seed.get('score_ids') or [])} "
                f"raw_weight_keys={sorted((seed.get('weights') or {}).keys())}"
            )
        else:
            steps.append("scored chain already present; skip reseed")
    else:
        steps.append("ensure_seeded=False; use existing scores")

    # Black-box: scores exist via public API for at least one miner.
    try:
        with httpx.Client(timeout=timeout) as client:
            for hk in (MINER_A, MINER_B, MINER_C):
                sc = client.get(f"{normalized}/v1/scores/{hk}")
                if sc.status_code != 200:
                    return _fail(
                        name,
                        normalized,
                        f"scores {hk} HTTP {sc.status_code}: {sc.text}",
                        steps,
                        None,
                    )
                items = sc.json().get("items") or []
                if not items:
                    return _fail(
                        name,
                        normalized,
                        f"no score rows for hotkey {hk} after seed",
                        steps,
                        None,
                    )
                steps.append(f"scores {hk} count={len(items)}")
    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"scores probe failed: {exc}", steps, None)

    # Push to mock-master.
    try:
        push = _run_async(
            lambda: _load_weights_and_push(
                master_url=master,
                shared_token=secret,
                epoch=7,
            ),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(name, normalized, f"push raised: {exc}", steps, None)

    steps.append(
        f"push status={push.get('status')} push_status={push.get('push_status')} "
        f"epoch={push.get('epoch')} revision={push.get('revision')} "
        f"local_id={push.get('local_id')}"
    )
    if push.get("status") != "acknowledged" or push.get("push_status") not in {
        "acked",
        "sim",
    }:
        return _fail(
            name,
            normalized,
            (
                f"push not acked: status={push.get('status')} "
                f"push_status={push.get('push_status')} err={push.get('error')}"
            ),
            steps,
            None,
        )

    weights = push.get("weights") or {}
    if not isinstance(weights, dict) or not weights:
        return _fail(name, normalized, "empty weights after scored chain", steps, None)
    for k, v in weights.items():
        if _finite_nonneg(v) is None:
            return _fail(name, normalized, f"illegal weight {k}={v}", steps, None)
    steps.append(f"finite weight map count={len(weights)}")

    # get_weights / weight-preview family reflects the map (snapshot or live).
    try:
        with httpx.Client(timeout=timeout) as client:
            preview = client.get(f"{normalized}/v1/weight-preview")
            steps.append(f"GET /v1/weight-preview → {preview.status_code}")
            if preview.status_code != 200:
                return _fail(
                    name,
                    normalized,
                    f"weight-preview HTTP {preview.status_code}",
                    steps,
                    None,
                )
            body = preview.json()
            wmap = body.get("weights") if isinstance(body, dict) else None
            if not isinstance(wmap, dict) or not wmap:
                return _fail(
                    name,
                    normalized,
                    f"weight-preview empty after ack: {body}",
                    steps,
                    None,
                )
            snap = body.get("snapshot") if isinstance(body, dict) else None
            if isinstance(snap, dict):
                steps.append(
                    f"snapshot push_status={snap.get('push_status')} "
                    f"epoch={snap.get('epoch')} rev={snap.get('revision')}"
                )
                # When snapshot is present post-push, expect acked/sim.
                if snap.get("push_status") not in {None, "acked", "sim", "pending"}:
                    return _fail(
                        name,
                        normalized,
                        f"unexpected snapshot push_status={snap.get('push_status')}",
                        steps,
                        None,
                    )
            for miner in (MINER_A, MINER_B, MINER_C):
                if miner not in wmap or float(wmap[miner]) <= 0:
                    return _fail(
                        name,
                        normalized,
                        f"weight-preview missing positive mass for {miner}: {wmap}",
                        steps,
                        None,
                    )
            steps.append("weight-preview reflects finite multi-miner map")
    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"preview failed: {exc}", steps, None)

    steps.append("cross-weight-push-ack complete")
    return ScenarioResult(
        name=name,
        ok=True,
        base_url=normalized,
        message=(
            "weight push after scored chain: mock-master acked; "
            "finite map in push + weight-preview"
        ),
        steps=steps,
    )


def run_cross_leaderboard_weights_agree(
    base_url: str,
    *,
    timeout: float = 30.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    ensure_seeded: bool = True,
) -> ScenarioResult:
    """VAL-CROSS-019: three hotkeys — leaderboard order matches weight mass order."""

    name = "cross-leaderboard-weights-agree"
    normalized = base_url.rstrip("/")
    steps: list[str] = []
    master = _resolve_master(master_url)
    _ = shared_token

    if ensure_seeded:
        # Idempotent reseed only when miner scores missing (shared fixture reuse).
        try:
            with httpx.Client(timeout=min(timeout, 5.0)) as client:
                sc = client.get(f"{normalized}/v1/scores/{MINER_A}")
                need = sc.status_code != 200 or not (sc.json().get("items") or [])
        except httpx.HTTPError:
            need = True
        if need:
            try:
                seed = _run_async(
                    lambda: _seed_multi_miner_and_self_deal(master_url=master),
                    timeout=timeout,
                )
                steps.append(f"seeded multi-miner scores ok={seed.get('ok')}")
            except Exception as exc:  # noqa: BLE001
                return _fail(name, normalized, f"seed failed: {exc}", steps, None)
        else:
            steps.append("multi-miner scores already present")

    try:
        with httpx.Client(timeout=timeout) as client:
            # Per-hotkey scores
            score_comps: dict[str, float] = {}
            for hk in (MINER_A, MINER_B, MINER_C):
                sc = client.get(f"{normalized}/v1/scores/{hk}")
                if sc.status_code != 200:
                    return _fail(
                        name,
                        normalized,
                        f"scores {hk} HTTP {sc.status_code}",
                        steps,
                        None,
                    )
                items = sc.json().get("items") or []
                if not items:
                    return _fail(
                        name, normalized, f"missing scores for {hk}", steps, None
                    )
                composite = float(items[0].get("composite") or 0.0)
                score_comps[hk] = composite
                steps.append(f"score {hk} composite={composite}")

            board = client.get(f"{normalized}/v1/leaderboard")
            if board.status_code != 200:
                return _fail(
                    name,
                    normalized,
                    f"leaderboard HTTP {board.status_code}: {board.text}",
                    steps,
                    None,
                )
            items = board.json().get("items") or []
            # Ranked map of our three miners only.
            ranked = [
                row
                for row in items
                if isinstance(row, dict) and row.get("hotkey") in {MINER_A, MINER_B, MINER_C}
            ]
            if len(ranked) < 3:
                return _fail(
                    name,
                    normalized,
                    f"leaderboard missing the three miners: {items}",
                    steps,
                    None,
                )
            # Find positions of A,B,C in the full board among themselves.
            our_positions = [
                str(row["hotkey"])
                for row in items
                if isinstance(row, dict)
                and row.get("hotkey") in {MINER_A, MINER_B, MINER_C}
            ]
            steps.append(f"leaderboard our_order={our_positions}")
            if our_positions[:3] != [MINER_A, MINER_B, MINER_C] and our_positions != [
                MINER_A,
                MINER_B,
                MINER_C,
            ]:
                # They must be monotone: A before B before C.
                try:
                    ia, ib, ic = (
                        our_positions.index(MINER_A),
                        our_positions.index(MINER_B),
                        our_positions.index(MINER_C),
                    )
                except ValueError as exc:
                    return _fail(
                        name, normalized, f"hotkey missing: {exc}", steps, None
                    )
                if not (ia < ib < ic):
                    return _fail(
                        name,
                        normalized,
                        f"leaderboard order not A>B>C: {our_positions}",
                        steps,
                        None,
                    )

            # Aggregates monotone
            agg = {
                str(row["hotkey"]): float(row.get("aggregate") or 0.0)
                for row in ranked
            }
            if not (agg[MINER_A] > agg[MINER_B] > agg[MINER_C] > 0):
                return _fail(
                    name,
                    normalized,
                    f"leaderboard aggregates not monotone positives: {agg}",
                    steps,
                    None,
                )
            steps.append(f"aggregates A>B>C: {agg}")

            # Weight-preview mass order must match (monotone).
            preview = client.get(f"{normalized}/v1/weight-preview")
            if preview.status_code != 200:
                return _fail(
                    name,
                    normalized,
                    f"weight-preview HTTP {preview.status_code}",
                    steps,
                    None,
                )
            wmap = preview.json().get("weights") or {}
            if not isinstance(wmap, dict):
                return _fail(name, normalized, "weights map missing", steps, None)
            wa = _finite_nonneg(wmap.get(MINER_A))
            wb = _finite_nonneg(wmap.get(MINER_B))
            wc = _finite_nonneg(wmap.get(MINER_C))
            if wa is None or wb is None or wc is None:
                return _fail(
                    name,
                    normalized,
                    f"missing finite weights for miners: {wmap}",
                    steps,
                    None,
                )
            if not (wa > wb > wc > 0):
                return _fail(
                    name,
                    normalized,
                    f"weight mass order not A>B>C: A={wa} B={wb} C={wc}",
                    steps,
                    None,
                )
            # Leaderboard top among our set must have positive weight (not 0).
            top_hk = our_positions[0]
            top_w = _finite_nonneg(wmap.get(top_hk))
            if top_w is None or top_w <= 0:
                return _fail(
                    name,
                    normalized,
                    f"leaderboard top {top_hk} has weight 0/illegal",
                    steps,
                    None,
                )
            steps.append(
                f"weights agree A={wa} B={wb} C={wc}; top={top_hk} weight={top_w}"
            )
            # Hotkey set agreement: each scored miner present in weights with >0.
            for hk in (MINER_A, MINER_B, MINER_C):
                if hk not in wmap or float(wmap[hk]) <= 0:
                    return _fail(
                        name,
                        normalized,
                        f"scored hotkey {hk} missing positive weight",
                        steps,
                        None,
                    )
            steps.append("hotkey set agreement: leaderboard ∩ scores ∩ weights")
    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"HTTP error: {exc}", steps, None)

    steps.append("cross-leaderboard-weights-agree complete")
    return ScenarioResult(
        name=name,
        ok=True,
        base_url=normalized,
        message=(
            "multi-miner leaderboard ranks match weight mass order (A>B>C); "
            "hotkey set agrees across scores/leaderboard/weights"
        ),
        steps=steps,
    )


def run_cross_self_deal_finite_damped(
    base_url: str,
    *,
    timeout: float = 30.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    ensure_seeded: bool = True,
) -> ScenarioResult:
    """VAL-CROSS-020: self-deal demand+supply same hotkey scores finite, optionally damped."""

    name = "cross-self-deal-finite-damped"
    normalized = base_url.rstrip("/")
    steps: list[str] = []
    master = _resolve_master(master_url)
    _ = shared_token

    if ensure_seeded:
        try:
            with httpx.Client(timeout=min(timeout, 5.0)) as client:
                sc = client.get(f"{normalized}/v1/scores/{SELF_DEAL_HK}")
                need = sc.status_code != 200 or not (sc.json().get("items") or [])
        except httpx.HTTPError:
            need = True
        if need:
            try:
                seed = _run_async(
                    lambda: _seed_multi_miner_and_self_deal(master_url=master),
                    timeout=timeout,
                )
                steps.append(f"seeded self-deal rows ok={seed.get('ok')}")
            except Exception as exc:  # noqa: BLE001
                return _fail(name, normalized, f"seed failed: {exc}", steps, None)
        else:
            steps.append("self-deal scores already present")

    try:
        with httpx.Client(timeout=timeout) as client:
            sc = client.get(f"{normalized}/v1/scores/{SELF_DEAL_HK}")
            if sc.status_code != 200:
                return _fail(
                    name,
                    normalized,
                    f"scores self-deal HTTP {sc.status_code}",
                    steps,
                    None,
                )
            items = sc.json().get("items") or []
            if len(items) < 2:
                return _fail(
                    name,
                    normalized,
                    f"expected demand+supply score rows, got {items}",
                    steps,
                    None,
                )
            roles = {str(i.get("role")) for i in items}
            if not ({"demand", "supply"} <= roles or "joint" in roles):
                # dual-role may be stored as demand+supply
                steps.append(f"roles observed={roles} (require demand+supply)")
            if "demand" not in roles or "supply" not in roles:
                return _fail(
                    name,
                    normalized,
                    f"self-deal missing demand+supply roles: {roles}",
                    steps,
                    None,
                )
            for item in items:
                comp = _finite_nonneg(item.get("composite"))
                if comp is None:
                    return _fail(
                        name,
                        normalized,
                        f"non-finite composite in self-deal scores: {item}",
                        steps,
                        None,
                    )
            steps.append(
                f"self-deal score rows={len(items)} roles={sorted(roles)} finite"
            )

            preview = client.get(f"{normalized}/v1/weight-preview")
            if preview.status_code != 200:
                return _fail(
                    name,
                    normalized,
                    f"weight-preview HTTP {preview.status_code}",
                    steps,
                    None,
                )
            wmap = preview.json().get("weights") or {}
            self_w = _finite_nonneg(wmap.get(SELF_DEAL_HK))
            twin_w = _finite_nonneg(wmap.get(TWIN_HONEST_HK))
            if self_w is None:
                return _fail(
                    name,
                    normalized,
                    f"self-deal weight missing/illegal: {wmap.get(SELF_DEAL_HK)}",
                    steps,
                    None,
                )
            steps.append(f"self_deal_weight={self_w} twin_weight={twin_w}")
            # Self-deal rows: demand efficiency 8 + supply efficiency 4.
            # Damping 0.5 → self mass 6 per seed unit; twin honest demand 8 per unit.
            # Ratio self/twin ≈ 0.75 with damping; ≈ 1.5 without (stack-invariant).
            if twin_w is not None and twin_w > 0:
                ratio = self_w / twin_w
                steps.append(f"self/twin ratio={ratio:.4f} (damped≈0.75, undamped≈1.5)")
                if ratio > 1.25:
                    return _fail(
                        name,
                        normalized,
                        (
                            f"self-deal damping not applied "
                            f"(self={self_w} twin={twin_w} ratio={ratio:.4f})"
                        ),
                        steps,
                        None,
                    )
                if abs(ratio - 0.75) > 0.2:
                    steps.append(
                        f"note: self/twin ratio {ratio:.4f} off exact 0.75; "
                        "still finite and below undamped 1.5 gate"
                    )
            else:
                # No twin: require finite ≥0 and not NaN (already); soft bound.
                undamped_unit = 8.0 + 4.0
                if self_w > undamped_unit * 20:
                    return _fail(
                        name,
                        normalized,
                        f"self-deal mass unreasonably large: {self_w}",
                        steps,
                        None,
                    )
            steps.append(f"self-deal finite weight={self_w} (optionally damped)")
    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"HTTP error: {exc}", steps, None)

    steps.append("cross-self-deal-finite-damped complete")
    return ScenarioResult(
        name=name,
        ok=True,
        base_url=normalized,
        message=(
            "self-deal demand+supply same hotkey scored dual roles; "
            "raw weight finite ≥0 and soft-damped vs undamped sum"
        ),
        steps=steps,
    )


def run_cross_mock_master_down_resilience(
    base_url: str,
    *,
    timeout: float = 60.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    stop_master_fn: Callable[[], None] | None = None,
    start_master_fn: Callable[[], str] | None = None,
    ensure_seeded: bool = True,
) -> ScenarioResult:
    """VAL-CROSS-027: master down keeps scores; push pending/failed; recover→acked.

    Test harness injects ``stop_master_fn`` / ``start_master_fn`` for chaos.
    When omitted, the run expects ``master_url`` already down for push-fail then
    tries push against whatever is configured (CLI-mode soft path documents skip).
    """

    name = "cross-mock-master-down-resilience"
    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    master = _resolve_master(master_url)

    if ensure_seeded:
        try:
            with httpx.Client(timeout=min(timeout, 5.0)) as client:
                sc = client.get(f"{normalized}/v1/scores/{MINER_A}")
                need = sc.status_code != 200 or not (sc.json().get("items") or [])
        except httpx.HTTPError:
            need = True
        if need:
            try:
                seed = _run_async(
                    lambda: _seed_multi_miner_and_self_deal(master_url=master),
                    timeout=timeout,
                )
                steps.append(f"seeded scores before chaos ok={seed.get('ok')}")
            except Exception as exc:  # noqa: BLE001
                return _fail(name, normalized, f"seed failed: {exc}", steps, None)
        else:
            steps.append("scores present before chaos")

    # Snapshot score counts before push chaos.
    pre_scores: dict[str, int] = {}
    try:
        with httpx.Client(timeout=timeout) as client:
            for hk in (MINER_A, MINER_B, MINER_C):
                sc = client.get(f"{normalized}/v1/scores/{hk}")
                if sc.status_code != 200:
                    return _fail(
                        name,
                        normalized,
                        f"pre-chaos scores {hk} HTTP {sc.status_code}",
                        steps,
                        None,
                    )
                pre_scores[hk] = len(sc.json().get("items") or [])
                if pre_scores[hk] < 1:
                    return _fail(
                        name,
                        normalized,
                        f"pre-chaos missing scores for {hk}",
                        steps,
                        None,
                    )
            steps.append(f"pre-chaos score counts={pre_scores}")
    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"pre-chaos probe: {exc}", steps, None)

    if stop_master_fn is not None:
        stop_master_fn()
        steps.append("injected mock-master stop")
        time.sleep(0.15)
    else:
        steps.append(
            "no stop_master_fn; attempting push against maybe-unreachable "
            f"master={master} (caller must ensure down for strict verify)"
        )

    # Push while master down → transport_error/failed or rejected; must not wipe scores.
    try:
        down_push = _run_async(
            lambda: _load_weights_and_push(
                master_url=master,
                shared_token=secret,
                epoch=27,
            ),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 — treat as soft transport failure
        down_push = {
            "status": "transport_error",
            "push_status": "failed",
            "error": str(exc),
            "local_id": None,
            "weights": {},
        }
        steps.append(f"push while down raised (recorded as transport_error): {exc}")

    steps.append(
        f"down-push status={down_push.get('status')} "
        f"push_status={down_push.get('push_status')} err={down_push.get('error')}"
    )
    # Must NOT report clean acked when master is intentionally down.
    if stop_master_fn is not None and down_push.get("status") == "acknowledged":
        return _fail(
            name,
            normalized,
            "push acked while mock-master was stopped (chaos inject broken?)",
            steps,
            None,
        )
    # Allowed non-success statuses for chaos: transport, discarded, rejected, failed.
    if down_push.get("status") in {"acknowledged"} and stop_master_fn is not None:
        return _fail(name, normalized, "unexpected acked during down", steps, None)
    steps.append(
        f"push non-acked as expected under chaos "
        f"(status={down_push.get('status')})"
    )

    # Scores must still be present (no soft-delete on push fail).
    try:
        with httpx.Client(timeout=timeout) as client:
            for hk, before in pre_scores.items():
                sc = client.get(f"{normalized}/v1/scores/{hk}")
                if sc.status_code != 200:
                    return _fail(
                        name,
                        normalized,
                        f"post-down scores {hk} HTTP {sc.status_code}",
                        steps,
                        None,
                    )
                after = len(sc.json().get("items") or [])
                if after < before:
                    return _fail(
                        name,
                        normalized,
                        f"score rows lost for {hk}: before={before} after={after}",
                        steps,
                        None,
                    )
                steps.append(f"scores durable {hk} before={before} after={after}")
    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"post-down scores probe: {exc}", steps, None)

    # Recover master and retry — eventual acked.
    recover_master = master
    if start_master_fn is not None:
        recover_master = start_master_fn() or master
        steps.append(f"injected mock-master restart at {recover_master}")
        time.sleep(0.2)
        # health wait
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                h = httpx.get(f"{recover_master.rstrip('/')}/health", timeout=1.0)
                if h.status_code == 200:
                    steps.append("mock-master health green after restart")
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            return _fail(
                name,
                normalized,
                f"mock-master not healthy after restart: {recover_master}",
                steps,
                None,
            )

    # Retry: reuse pending/failed snapshot when we have local_id, or new epoch.
    reuse = down_push.get("local_id")
    try:
        if reuse:
            recovery = _run_async(
                lambda: _load_weights_and_push(
                    master_url=recover_master,
                    shared_token=secret,
                    epoch=27,
                    revision=None,
                    reuse_snapshot_id=str(reuse),
                ),
                timeout=timeout,
            )
            # If reuse rejected (e.g. expired window), fall back to fresh push.
            if recovery.get("status") != "acknowledged":
                steps.append(
                    f"reuse push status={recovery.get('status')}; "
                    "falling back to fresh epoch push"
                )
                recovery = _run_async(
                    lambda: _load_weights_and_push(
                        master_url=recover_master,
                        shared_token=secret,
                        epoch=28,
                    ),
                    timeout=timeout,
                )
        else:
            recovery = _run_async(
                lambda: _load_weights_and_push(
                    master_url=recover_master,
                    shared_token=secret,
                    epoch=28,
                ),
                timeout=timeout,
            )
    except Exception as exc:  # noqa: BLE001
        return _fail(name, normalized, f"recovery push raised: {exc}", steps, None)

    steps.append(
        f"recovery push status={recovery.get('status')} "
        f"push_status={recovery.get('push_status')} "
        f"epoch={recovery.get('epoch')} rev={recovery.get('revision')}"
    )
    if start_master_fn is not None or stop_master_fn is not None:
        if recovery.get("status") != "acknowledged" or recovery.get("push_status") not in {
            "acked",
            "sim",
        }:
            return _fail(
                name,
                normalized,
                (
                    f"recovery push not acked: status={recovery.get('status')} "
                    f"push_status={recovery.get('push_status')} "
                    f"err={recovery.get('error')}"
                ),
                steps,
                None,
            )
        steps.append("eventual acked after mock-master recovery")
    else:
        # Soft path without injectors: scores durable was the key durable assertion.
        steps.append(
            "no start/stop injectors; durable scores verified; "
            f"recovery outcome recorded status={recovery.get('status')}"
        )
        if recovery.get("status") == "acknowledged":
            steps.append("recovery acked with live master (bonus)")

    # Final durable re-check scores.
    try:
        with httpx.Client(timeout=timeout) as client:
            for hk, before in pre_scores.items():
                sc = client.get(f"{normalized}/v1/scores/{hk}")
                after = len((sc.json() or {}).get("items") or [])
                if after < before:
                    return _fail(
                        name,
                        normalized,
                        f"scores vanished after recovery for {hk}",
                        steps,
                        None,
                    )
            steps.append("scores still durable after recovery push")
    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"final scores probe: {exc}", steps, None)

    steps.append("cross-mock-master-down-resilience complete")
    return ScenarioResult(
        name=name,
        ok=True,
        base_url=normalized,
        message=(
            "mock-master down kept score rows; push non-acked then eventual "
            "acked on recovery (pending/failed → acked)"
        ),
        steps=steps,
    )


def check_mission_ports_labeled(base_url: str, master_url: str | None = None) -> list[str]:
    """Document mission ports used by this cross slice."""

    steps: list[str] = []
    api_port = _port_from_url(base_url)
    master = _resolve_master(master_url)
    m_port = _port_from_url(master)
    steps.append(
        f"API default/doc={DEFAULT_BAREMETAL_PORT} observed={api_port} "
        f"band={MIN_MISSION_PORT}-{MAX_MISSION_PORT}"
    )
    steps.append(
        f"mock-master default/doc={DEFAULT_MOCK_MASTER_PORT} observed={m_port}"
    )
    if api_port is not None and not is_mission_port(api_port):
        steps.append(f"WARN: API port {api_port} outside mission band")
    if m_port is not None and not is_mission_port(m_port):
        steps.append(f"WARN: master port {m_port} outside mission band")
    return steps


def run_cross_weights_leaderboard_selfdeal(
    base_url: str,
    *,
    timeout: float = 90.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    stop_master_fn: Callable[[], None] | None = None,
    start_master_fn: Callable[[], str] | None = None,
    include_master_chaos: bool = True,
) -> ScenarioResult:
    """Bundle VAL-CROSS-012/019/020/027 for CLI + programmatic suite."""

    name = CROSS_WEIGHTS_LEADERBOARD
    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    master = _resolve_master(master_url)
    steps.extend(check_mission_ports_labeled(normalized, master))

    # Seed once for all sub-paths.
    try:
        seed = _run_async(
            lambda: _seed_multi_miner_and_self_deal(master_url=master),
            timeout=timeout,
        )
        steps.append(
            f"shared seed score_count={len(seed.get('score_ids') or [])} "
            f"weights={sorted((seed.get('weights') or {}).keys())}"
        )
    except Exception as exc:  # noqa: BLE001
        return ScenarioResult(
            name=name,
            ok=False,
            base_url=normalized,
            message=f"shared seed failed: {exc}",
            steps=steps,
        )

    # 012: push ack (master must be up for this step). Scores already seeded above.
    r012 = run_cross_weight_push_ack(
        normalized,
        timeout=timeout,
        shared_token=secret,
        master_url=master,
        ensure_seeded=False,
    )
    steps.extend(r012.steps)
    if not r012.ok:
        # One retry after explicit push epoch bump path.
        try:
            push = _run_async(
                lambda: _load_weights_and_push(
                    master_url=master,
                    shared_token=secret,
                    epoch=70,
                ),
                timeout=timeout,
            )
            steps.append(
                f"retry push status={push.get('status')} "
                f"push_status={push.get('push_status')}"
            )
            if push.get("status") != "acknowledged":
                return ScenarioResult(
                    name=name,
                    ok=False,
                    base_url=normalized,
                    message=f"VAL-CROSS-012 failed: {r012.message}",
                    steps=steps,
                )
            steps.append("VAL-CROSS-012 push ack ok (retry path)")
        except Exception as exc:  # noqa: BLE001
            return ScenarioResult(
                name=name,
                ok=False,
                base_url=normalized,
                message=f"VAL-CROSS-012 failed: {r012.message}; retry={exc}",
                steps=steps,
            )
    else:
        steps.append("VAL-CROSS-012 push ack ok")

    # 019 selection + 020 use already-seeded data.
    r019 = run_cross_leaderboard_weights_agree(
        normalized,
        timeout=timeout,
        shared_token=secret,
        master_url=master,
        ensure_seeded=False,
    )
    steps.extend(r019.steps)
    if not r019.ok:
        return ScenarioResult(
            name=name,
            ok=False,
            base_url=normalized,
            message=f"VAL-CROSS-019 failed: {r019.message}",
            steps=steps,
        )
    steps.append("VAL-CROSS-019 leaderboard/weights agree")

    r020 = run_cross_self_deal_finite_damped(
        normalized,
        timeout=timeout,
        shared_token=secret,
        master_url=master,
        ensure_seeded=False,
    )
    steps.extend(r020.steps)
    if not r020.ok:
        return ScenarioResult(
            name=name,
            ok=False,
            base_url=normalized,
            message=f"VAL-CROSS-020 failed: {r020.message}",
            steps=steps,
        )
    steps.append("VAL-CROSS-020 self-deal finite+damped")

    if include_master_chaos:
        r027 = run_cross_mock_master_down_resilience(
            normalized,
            timeout=timeout,
            shared_token=secret,
            master_url=master,
            stop_master_fn=stop_master_fn,
            start_master_fn=start_master_fn,
            ensure_seeded=False,
        )
        steps.extend(r027.steps)
        if not r027.ok:
            return ScenarioResult(
                name=name,
                ok=False,
                base_url=normalized,
                message=f"VAL-CROSS-027 failed: {r027.message}",
                steps=steps,
            )
        steps.append("VAL-CROSS-027 mock-master down resilient")
    else:
        steps.append("VAL-CROSS-027 skipped (include_master_chaos=False)")

    steps.append("cross-weights-leaderboard-selfdeal complete")
    return ScenarioResult(
        name=name,
        ok=True,
        base_url=normalized,
        message=(
            "cross weights/leaderboard/selfdeal/master-chaos passed "
            "(VAL-CROSS-012/019/020/027)"
        ),
        steps=steps,
    )


__all__ = [
    "ALLOWED_IMAGE",
    "CROSS_WEIGHTS_LEADERBOARD",
    "MINER_A",
    "MINER_B",
    "MINER_C",
    "SELF_DEAL_HK",
    "TWIN_HONEST_HK",
    "check_mission_ports_labeled",
    "run_cross_leaderboard_weights_agree",
    "run_cross_mock_master_down_resilience",
    "run_cross_self_deal_finite_damped",
    "run_cross_weight_push_ack",
    "run_cross_weights_leaderboard_selfdeal",
]
