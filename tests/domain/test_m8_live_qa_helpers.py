"""Unit tests for M8 external QA helpers (no live Verda network).

Helpers live under scripts/qa (outside product package). Gated pytest
must never require VERDA_* credentials (VAL-LIVE-010).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

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
def verda_client_mod() -> ModuleType:
    return _load_module("scripts_qa_verda_client", SCRIPTS_QA / "verda_client.py")


def test_scripts_qa_not_in_product_package() -> None:
    """VAL-LIVE-001: external QA scripts stay outside src/hypercluster."""

    product = REPO_ROOT / "src" / "hypercluster"
    assert not (product / "verda_client.py").exists()
    assert SCRIPTS_QA.is_dir()
    assert (SCRIPTS_QA / "verda_single_gpu_smoke.py").is_file()
    # Product package must not import scripts.qa helpers (regex avoids comment hits).
    import re

    product_import = re.compile(r"^\s*(?:from|import)\s+scripts\.qa\b", re.M)
    live_verda_import = re.compile(r"^\s*(?:from|import)\s+verda(?:[.\s]|$)", re.M)
    for py in product.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not product_import.search(text), f"{py} imports scripts.qa"
        assert not live_verda_import.search(text), f"{py} imports verda"


def test_pick_cheapest_single_gpu_respects_caps(verda_client_mod: ModuleType) -> None:
    pick = verda_client_mod.pick_cheapest_single_gpu
    types = [
        {
            "instance_type": "1A6000.10V",
            "model": "RTX A6000",
            "price_per_hour": "0.6100",
            "gpu": {"number_of_gpus": 1},
            "gpu_memory": {"size_in_gigabytes": 48},
            "cpu": {"number_of_cores": 10},
            "memory": {"size_in_gigabytes": 64},
        },
        {
            "instance_type": "2A100.80V",
            "model": "A100",
            "price_per_hour": "0.20",
            "gpu": {"number_of_gpus": 2},
        },
        {
            "instance_type": "1H100.80S.30V",
            "model": "H100",
            "price_per_hour": "3.25",
            "gpu": {"number_of_gpus": 1},
        },
        {
            "instance_type": "CPU.4V.16G",
            "model": "CPU",
            "price_per_hour": "0.05",
            "gpu": {"number_of_gpus": 0},
        },
    ]
    availability = [
        {"location_code": "FIN-01", "availabilities": ["1A6000.10V", "CPU.4V.16G"]},
        {"location_code": "FIN-03", "availabilities": ["1H100.80S.30V", "2A100.80V"]},
    ]
    choice = pick(types, availability, max_rate_usd=1.50)
    assert choice is not None
    assert choice.instance_type == "1A6000.10V"
    assert choice.gpu_count == 1
    assert choice.price_per_hour == 0.61
    assert choice.location_code == "FIN-01"

    # Rate cap excludes expensive H100-only market.
    none_choice = pick(
        types,
        [{"location_code": "FIN-03", "availabilities": ["1H100.80S.30V"]}],
        max_rate_usd=1.0,
    )
    assert none_choice is None


def test_estimate_cost_and_cap(verda_client_mod: ModuleType) -> None:
    cost = verda_client_mod.estimate_cost_usd(
        price_per_hour=0.61,
        start_unix=1_000.0,
        end_unix=1_000.0 + 600.0,  # 10 minutes
        min_billable_minutes=1.0,
    )
    assert cost == pytest.approx(0.61 * (600.0 / 3600.0), rel=1e-5)
    assert verda_client_mod.cost_within_hard_cap(cost, 5.0)
    assert not verda_client_mod.cost_within_hard_cap(6.0, 5.0)


def test_redact_secrets_strips_tokens(verda_client_mod: ModuleType) -> None:
    payload = {
        "access_token": "super-secret",
        "nested": {"client_secret": "x", "ok": 1},
        "authorization": "Bearer abc",
    }
    red = verda_client_mod.redact_secrets(payload)
    assert red["access_token"] == "***REDACTED***"
    assert red["nested"]["client_secret"] == "***REDACTED***"
    assert red["nested"]["ok"] == 1
    assert red["authorization"] == "***REDACTED***"


def test_live_verda_marker_is_opt_in_only() -> None:
    """VAL-LIVE-010: live_verda marker exists; this module is offline-only.

    Collection of this file must not require external credentials files.
    """

    assert not (REPO_ROOT / "src" / "hypercluster" / "verda_client.py").exists()
    # Ensure pytest config documents the mark.
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "live_verda" in text
    lower = text.lower()
    assert "never default" in lower or "never selects" in lower or "opt-in" in lower
    # Offline unit helpers must not open env by themselves at import time.
    source = (SCRIPTS_QA / "verda_client.py").read_text(encoding="utf-8")
    assert "class VerdaClient" in source
    # Module import of verda_client does not require network (constructor does).
    assert "urlopen" in source


def test_discontinue_safe_parse_already_gone(verda_client_mod: ModuleType) -> None:
    """VAL-LIVE-014 unit: client.discontinue treats not-found as ok/idempotent.

    Stubs HTTP to avoid network.
    """

    client_cls = verda_client_mod.VerdaClient

    class Stub(client_cls):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:  # noqa: D401
            self.client_id = "x"
            self.client_secret = "y"
            self.api_base = "https://example.invalid"
            self.timeout_s = 1.0
            self.user_agent = "test"
            self.cf_connecting_ip = "127.0.0.1"
            self._token = "t"
            self._token_expires_at = 1e18

        def instance_action(self, **kwargs):  # type: ignore[no-untyped-def]
            raise verda_client_mod.VerdaOpsError("HTTP 404 not found discontinued")

    stub = Stub()
    out = stub.discontinue("abc-123")
    assert out["ok"] is True
    assert out.get("idempotent") is True
