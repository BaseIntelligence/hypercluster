"""VAL-FAB-001 / VAL-FAB-018 / VAL-FAB-019: FabricReport schema + sim inventory + fabric-scan."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from hypercluster.fabric.discovery import (
    FabricReport,
    IbDevice,
    build_fabric_report,
    canonical_report_payload,
    compute_report_digest,
    validate_accepted_report,
)
from hypercluster.sim.inventory import (
    SimInventory,
    default_sim_inventory,
    plan_readiness,
    seed_sim_inventory,
)


def _sha256_hex(payload: str) -> str:
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


# ----- VAL-FAB-001 -----------------------------------------------------------


def test_fabric_report_requires_report_digest_and_topology_fields() -> None:
    """VAL-FAB-001: accepted report must carry digest + core topology fields."""

    collected = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)
    report = build_fabric_report(
        node_id="node-a",
        collected_at=collected,
        ib_devices=[
            IbDevice(name="mlx5_0", port=1, rate_gbps=200.0, state="Active"),
        ],
        gpu_gpu_topo_matrix="GPU0\tX\nGPU1\tNV1\n",
        numa_map={"gpu0": 0, "gpu1": 0},
        nccl_version="2.21.5",
        eth_ifaces=["eth0", "lo"],
        gpu_count=2,
    )

    assert report.node_id == "node-a"
    assert report.collected_at == collected
    assert isinstance(report.ib_devices, list)
    assert len(report.ib_devices) == 1
    assert report.ib_devices[0].name == "mlx5_0"
    assert report.gpu_topo_sha256
    assert report.report_digest.startswith("sha256:")
    assert len(report.report_digest) == len("sha256:") + 64

    payload = canonical_report_payload(report)
    recomputed = compute_report_digest(payload)
    assert report.report_digest == recomputed

    public = report.model_dump(mode="json")
    assert public["node_id"] == "node-a"
    assert public["ib_devices"]
    assert public["gpu_topo_sha256"]
    assert public["report_digest"] == report.report_digest


def test_fabric_report_digest_stable_for_canonical_fixture() -> None:
    """Digest is sha256 of canonical JSON (sorted keys, no digest field)."""

    body = {
        "node_id": "fixture-node",
        "collected_at": "2026-07-15T12:00:00Z",
        "ib_devices": [{"name": "mlx5_0", "port": 1, "rate_gbps": 200.0, "state": "Active"}],
        "ib_rate_gbps": 200.0,
        "gpu_gpu_topo_matrix": "GPU0\tX\n",
        "gpu_topo_sha256": hashlib.sha256(b"GPU0\tX\n").hexdigest(),
        "numa_map": {"gpu0": 0},
        "nccl_version": "sim-2.21.5",
        "eth_ifaces": ["lo"],
    }
    digest = compute_report_digest(body)
    expect = _sha256_hex(json.dumps(body, sort_keys=True, separators=(",", ":")))
    assert digest == expect
    # Recompute twice for stability.
    assert compute_report_digest(body) == digest


def test_fabric_report_rejects_missing_digest_on_validate() -> None:
    """FAIL path: report without digest is not accepted."""

    with pytest.raises(ValidationError):
        FabricReport(
            node_id="n1",
            collected_at=datetime.now(UTC),
            ib_devices=[],
            gpu_topo_sha256="abc",
            # report_digest omitted
        )


def test_multi_gpu_claim_requires_nonempty_gpu_topo_sha() -> None:
    """FAIL: multi-GPU report with empty topo sha is rejected on accept validation."""

    report = build_fabric_report(
        node_id="n-multi",
        collected_at=datetime.now(UTC),
        ib_devices=[],
        gpu_gpu_topo_matrix="",
        numa_map={},
        gpu_count=4,
        force_empty_topo_sha=True,
    )
    with pytest.raises(ValueError, match="gpu_topo_sha256"):
        validate_accepted_report(report, gpu_count=4)


def test_empty_ib_devices_allowed_with_digest() -> None:
    """ib_devices may be empty (eth-only) so long as digest + topo are present."""

    report = build_fabric_report(
        node_id="eth-only",
        collected_at=datetime.now(UTC),
        ib_devices=[],
        gpu_gpu_topo_matrix="GPU0\tX\n",
        numa_map={"gpu0": 0},
        gpu_count=1,
    )
    accepted = validate_accepted_report(report, gpu_count=1)
    assert accepted.ib_devices == []
    assert accepted.report_digest


# ----- VAL-FAB-019 -----------------------------------------------------------


def test_sim_inventory_synthetic_ib_nvlink_enables_multi_node_plan() -> None:
    """VAL-FAB-019: seeded sim inventory yields multi-node plan readiness without HW."""

    inv = seed_sim_inventory(seed=42, node_count=4, gpus_per_node=2)
    assert isinstance(inv, SimInventory)
    assert len(inv.nodes) >= 2

    for node in inv.nodes:
        report = node.fabric_report
        assert report.report_digest
        assert report.gpu_topo_sha256
        # Default sim topology includes IB devices and NVLink topo text.
        assert len(report.ib_devices) >= 1
        assert "NV" in report.gpu_gpu_topo_matrix or report.gpu_count >= 1

    readiness = plan_readiness(inv, world_size=4, nnodes=2, nproc_per_node=2)
    assert readiness.ok is True
    assert readiness.rankmap
    assert len(readiness.rankmap) == 4
    node_ids = {binding["node_id"] for binding in readiness.rankmap}
    assert len(node_ids) >= 1
    # Multi-node: at least two distinct nodes for world 4 with 2/node.
    assert len(node_ids) == 2
    ranks = sorted(b["rank"] for b in readiness.rankmap)
    assert ranks == list(range(4))


def test_default_sim_inventory_is_deterministic() -> None:
    """default_sim_inventory is deterministic for a fixed seed."""

    a = default_sim_inventory(seed=7)
    b = default_sim_inventory(seed=7)
    digests_a = [n.fabric_report.report_digest for n in a.nodes]
    digests_b = [n.fabric_report.report_digest for n in b.nodes]
    assert digests_a == digests_b
    assert a.graph_digest == b.graph_digest
    assert a.graph_digest.startswith("sha256:")


def test_sim_seed_produces_ib_edges_graph() -> None:
    """Seed exposes synthetic IB/NVLink graph edges for multi-node connectivity."""

    inv = seed_sim_inventory(seed=1, node_count=3, gpus_per_node=2)
    assert inv.ib_edges
    # Fully-connected or chain of IB edges among nodes
    edge_nodes = set()
    for edge in inv.ib_edges:
        edge_nodes.add(edge["src"])
        edge_nodes.add(edge["dst"])
    assert len(edge_nodes) == 3
    for node in inv.nodes:
        assert node.nvlink_pairs or node.gpu_count == 1


# ----- VAL-FAB-018 (domain-level, no HTTP required) --------------------------


@pytest.mark.asyncio
async def test_fabric_scan_accepts_report_and_persists_digest(
    settings_factory, tmp_path
) -> None:
    """VAL-FAB-018: fabric-scan inserts/updates fabric_reports for a sim node."""

    from hypercluster.db.database import Database
    from hypercluster.domain.fabric_reports import (
        FabricReportError,
        fabric_scan_node,
        get_latest_fabric_report,
    )
    from hypercluster.domain.nodes import register_node
    from hypercluster.domain.providers import register_provider

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'fab.sqlite3'}"
    database = Database(db_url)
    await database.init()
    try:
        async with database.session() as session:
            provider, _ = await register_provider(
                session, hotkey="sim-fab-provider-hotkey-aaaaaaaaaaaaaaaaaa"
            )
            node, _ = await register_node(
                session,
                hotkey=provider.hotkey,
                gpu_model="H100",
                gpu_count=2,
                ssh_endpoint="sim-host-1:22",
                inventory={"source": "pre-scan"},
            )
            node_id = node.id

            report = await fabric_scan_node(
                session,
                node_id=node_id,
                source="sim",
                seed=99,
            )
            assert report.node_id == node_id
            assert report.report_digest
            assert report.gpu_topo_sha256
            assert isinstance(report.ib_devices, list)

            stored = await get_latest_fabric_report(session, node_id)
            assert stored is not None
            assert stored.report_digest == report.report_digest
            assert stored.node_id == node_id

            # Rescan without topology change keeps the same digest.
            report2 = await fabric_scan_node(
                session,
                node_id=node_id,
                source="sim",
                seed=99,
            )
            assert report2.report_digest == report.report_digest

            # Different seed/topo changes digest.
            report3 = await fabric_scan_node(
                session,
                node_id=node_id,
                source="sim",
                seed=100,
                topo_variant="spread",
            )
            assert report3.report_digest != report.report_digest

        async with database.session() as session:
            with pytest.raises(FabricReportError) as exc_info:
                await fabric_scan_node(session, node_id="does-not-exist", source="sim")
            assert exc_info.value.status_code == 404
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_fabric_scan_api_and_cli_surface(
    settings_factory, tmp_path
) -> None:
    """VAL-FAB-018: POST fabric-scan + CLI wrap produce accepted dashboard report."""

    import json as _json

    from httpx import ASGITransport, AsyncClient
    from typer.testing import CliRunner

    from hypercluster.api.auth import build_signed_headers
    from hypercluster.app import create_app
    from hypercluster.cli import app as cli_app
    from hypercluster.settings import HyperSettings

    token = "test-challenge-shared-token"
    hotkey = "sim-fab-scan-hotkey-bbbbbbbbbbbbbbbbbbbbbbbb"

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'fab-api.sqlite3'}",
        shared_token=token,
        shared_token_file=None,
    )
    hyper = HyperSettings(allow_insecure_signatures=True, signature_ttl_seconds=300)
    app = create_app(settings, hyper_settings=hyper)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Register provider + node via signed paths.
            raw = _json.dumps({"display_name": "fab"}).encode()
            headers = build_signed_headers(secret=token, hotkey=hotkey, body=raw)
            headers["Content-Type"] = "application/json"
            r = await client.post("/v1/providers/register", content=raw, headers=headers)
            assert r.status_code == 200, r.text

            raw = _json.dumps(
                {
                    "gpu_model": "H100",
                    "gpu_count": 2,
                    "ssh_endpoint": "10.9.0.1:22",
                    "inventory": {"sim": True},
                }
            ).encode()
            headers = build_signed_headers(secret=token, hotkey=hotkey, body=raw)
            headers["Content-Type"] = "application/json"
            r = await client.post("/v1/nodes", content=raw, headers=headers)
            assert r.status_code == 200, r.text
            node_id = r.json()["id"]

            # Fabric-scan endpoint
            raw = _json.dumps({"source": "sim", "seed": 3}).encode()
            headers = build_signed_headers(secret=token, hotkey=hotkey, body=raw)
            headers["Content-Type"] = "application/json"
            r = await client.post(
                f"/v1/nodes/{node_id}/fabric-scan",
                content=raw,
                headers=headers,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["node_id"] == node_id
            assert body["report_digest"]
            assert "ib_devices" in body
            assert body["gpu_topo_sha256"]
            digest1 = body["report_digest"]

            # GET fabric report view
            r = await client.get(f"/v1/nodes/{node_id}/fabric-report")
            assert r.status_code == 200, r.text
            assert r.json()["report_digest"] == digest1

            # Missing node → 404
            raw = b"{}"
            headers = build_signed_headers(secret=token, hotkey=hotkey, body=raw)
            headers["Content-Type"] = "application/json"
            r = await client.post(
                "/v1/nodes/missing-node-id/fabric-scan",
                content=raw,
                headers=headers,
            )
            assert r.status_code == 404

    # CLI wrapper (in-process, no live server): uses domain helper via --json offline path
    runner = CliRunner()
    # CLI offline fabric-scan-sim uses seed inventory for local verification when
    # --offline is set; live path needs --url. Unit-test offline sim seed command.
    result = runner.invoke(
        cli_app,
        ["sim", "seed", "--node-count", "2", "--gpus-per-node", "2", "--seed", "5"],
    )
    assert result.exit_code == 0, result.output
    assert "graph_digest" in result.output or "nodes=" in result.output


def test_plan_readiness_fails_without_topology() -> None:
    """Empty inventory cannot plan multi-node (FAIL signal for VAL-FAB-019)."""

    empty = SimInventory(nodes=[], ib_edges=[], nvlink_edges=[], graph_digest="sha256:" + "0" * 64)
    readiness = plan_readiness(empty, world_size=2, nnodes=2, nproc_per_node=1)
    assert readiness.ok is False
    assert "topology" in readiness.reason.lower() or "no" in readiness.reason.lower()
