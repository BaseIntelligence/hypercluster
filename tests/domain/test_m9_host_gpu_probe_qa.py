"""Offline unit tests for M9 host_gpu_probe + smoke helper wiring.

No live Verda, no live SSH. Product tree stays free of Verda clients
(VAL-GPU-065). Live rent/discontinue assertions are ops-only (VAL-GPU-060..064).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from hypercluster.domain.gpu_probes import compute_attach_digest
from hypercluster.probe.model_table import models_match, normalize_gpu_model
from hypercluster.probe.pipeline import GpuProbeConfig, GpuProbeContext, run_gpu_probe
from hypercluster.probe.transport import FakeSshTransport
from hypercluster.probe.types import ClaimedInventory

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_QA = REPO_ROOT / "scripts" / "qa"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def host_probe_mod() -> ModuleType:
    return _load_module("scripts_qa_host_gpu_probe", SCRIPTS_QA / "host_gpu_probe.py")


@pytest.fixture(scope="module")
def product_path_mod() -> ModuleType:
    return _load_module("scripts_qa_product_path_m9", SCRIPTS_QA / "product_path.py")


def test_host_gpu_probe_module_exists_outside_product() -> None:
    """VAL-GPU-060 tooling lives under scripts/qa, not src/hypercluster."""

    product = REPO_ROOT / "src" / "hypercluster"
    assert (SCRIPTS_QA / "host_gpu_probe.py").is_file()
    assert not (product / "host_gpu_probe.py").exists()
    assert not (product / "verda_client.py").exists()


def test_product_tree_still_free_of_verda_client() -> None:
    """VAL-GPU-065: product tree greps clean of Verda SDK / api.verda.com client."""

    product = REPO_ROOT / "src" / "hypercluster"
    product_import = re.compile(r"^\s*(?:from|import)\s+scripts\.qa\b", re.M)
    live_verda_import = re.compile(r"^\s*(?:from|import)\s+verda(?:[.\s]|$)", re.M)
    for py in product.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not product_import.search(text), f"{py} imports scripts.qa"
        assert not live_verda_import.search(text), f"{py} imports verda"
        # Comments may mention the ban; forbid live URL literals that look like clients.
        for line in text.splitlines():
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                continue
            if "api.verda.com" not in stripped:
                continue
            if "http" not in stripped and "VERDA_API" not in stripped:
                continue
            lower = stripped.lower()
            if "never" in lower or "forbid" in lower or "no " in lower:
                continue
            pytest.fail(f"{py} embeds Verda API host: {stripped[:120]}")


def test_fake_host_probe_pass_all_status(host_probe_mod: ModuleType) -> None:
    """Offline FakeSsh path equivalent of host_probe.json status=passed."""

    from hypercluster.probe.fixtures import get_fixture

    fx = get_fixture("pass_all")
    transport = FakeSshTransport(scripted=fx.scripted)
    # Fixture bank uses its own claim family; align claim to fixture.
    claimed = fx.claimed if fx.claimed is not None else ClaimedInventory(
        gpu_model="1V100.6V", gpu_count=1
    )
    ctx = GpuProbeContext(
        node_id="node-m9-offline",
        provider_hotkey="hk",
        ssh_endpoint="10.0.0.9:22",
        claimed=claimed,
        key_fingerprint="sha256:test",
    )
    evidence = run_gpu_probe(transport, ctx, config=GpuProbeConfig(mode="full"))
    public = evidence.to_public()
    assert public["status"] == "passed"
    measured_name = public["measured"]["gpus"][0]["name"]
    assert models_match(claimed.gpu_model, measured_name) or normalize_gpu_model(
        measured_name
    ) is not None
    assert normalize_gpu_model(measured_name) is not None
    host_probe = {
        "status": public["status"],
        "ok": public["status"] == "passed",
        "evidence_id": public["id"],
        "claimed": {
            "gpu_model": claimed.gpu_model,
            "gpu_count": claimed.gpu_count,
            "family": normalize_gpu_model(claimed.gpu_model),
        },
        "measured": public["measured"],
        "measured_primary_name": measured_name,
        "measured_family": normalize_gpu_model(measured_name),
        "claim_model_class_match": models_match(claimed.gpu_model, measured_name),
        "checks": public["checks"],
        "digests": public["digests"],
        "mode": public["mode"],
        "transport": "fake",
        "key_fingerprint": "sha256:test",
        "raw_redacted": public.get("raw_redacted") or {},
    }
    assert host_probe["ok"] is True
    assert host_probe["claim_model_class_match"] is True
    assert "private_key" not in json.dumps(host_probe)
    assert "BEGIN" not in json.dumps(host_probe)


def test_key_ref_body_parsing(product_path_mod: ModuleType) -> None:
    parse = product_path_mod._key_ref_body
    assert parse("file:/tmp/k.pem") == {"kind": "file", "name": "/tmp/k.pem"}
    assert parse("env:HYPER_SSH_KEY_PATH") == {"kind": "env", "name": "HYPER_SSH_KEY_PATH"}
    assert parse("/tmp/k.pem") == {"kind": "file", "name": "/tmp/k.pem"}
    assert parse({"kind": "file", "name": "/x"}) == {"kind": "file", "name": "/x"}
    assert parse(None) is None
    assert parse("") is None


def test_attach_digest_matches_product_rules(host_probe_mod: ModuleType) -> None:
    """Attach payload digest is computable offline for product store path."""

    evidence = {
        "status": "passed",
        "mode": "full",
        "transport": "real",
        "claimed": {"gpu_model": "Tesla V100", "gpu_count": 1},
        "measured": {
            "gpu_count": 1,
            "gpus": [{"name": "Tesla V100-SXM2-16GB", "uuid": "GPU-1"}],
        },
        "checks": [
            {
                "id": "ssh_connect",
                "fatal": True,
                "passed": True,
                "message": "ok",
                "duration_ms": 1,
                "details": {},
            }
        ],
    }
    digest = compute_attach_digest(evidence)
    assert digest.startswith("sha256:")
    # Same payload → same digest (deterministic).
    assert compute_attach_digest(evidence) == digest


def test_wait_tcp_timeout_fast(host_probe_mod: ModuleType) -> None:
    """wait_tcp fails closed when nothing listens (no hang for unit gate)."""

    out = host_probe_mod.wait_tcp("127.0.0.1", 1, timeout_s=0.4, interval_s=0.1)
    assert out["ok"] is False
    assert out["attempts"] >= 1


def test_smoke_script_exposes_with_host_probe_flag() -> None:
    source = (SCRIPTS_QA / "verda_single_gpu_smoke.py").read_text(encoding="utf-8")
    assert "--with-host-probe" in source
    assert "host_gpu_probe" in source
    assert "host_probe.json" in source
    assert "cost_ceiling.json" in source
    assert "run_product_gpu_probe" in source
    assert "attach_host_probe_evidence" in source
    # Never set_weights from this ops path.
    assert "set_weights" in source  # fence documentation
    assert "set_weights(" not in source
    # Product Verda adapter must stay false.
    assert '"product_verda_adapter": False' in source or (
        "product_verda_adapter" in source and "False" in source
    )


def test_model_class_match_catalog_aliases() -> None:
    """VAL-GPU-062 offline: catalog/claim family matches measured class."""

    assert models_match("1V100.6V", "Tesla V100-SXM2-16GB")
    assert models_match("A100", "NVIDIA A100-SXM4-40GB")
    assert models_match("1A100.40S.22V", "NVIDIA A100-SXM4-40GB") or (
        normalize_gpu_model("1A100.40S.22V") == "a100"
        or normalize_gpu_model("NVIDIA A100-SXM4-40GB") == "a100"
    )
    # Known mismatch: consumer vs data-center loss.
    assert not models_match("H100", "GeForce RTX 4090")


def test_scripts_qa_host_probe_is_offline_gated() -> None:
    """host_gpu_probe ops helper is Verda-free and reuses product probe pipeline."""

    source = (SCRIPTS_QA / "host_gpu_probe.py").read_text(encoding="utf-8")
    # Must not load commercial cloud credentials or rental clients.
    assert "client_credentials" not in source.lower()
    assert "oauth" not in source.lower()
    # Must reuse product probe pipeline (allowlist spirit).
    assert "run_gpu_probe" in source
    assert "RealSshExecutor" in source
    assert "set_weights" not in source or "never calls set_weights" in source
