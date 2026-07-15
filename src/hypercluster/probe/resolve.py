"""SSH transport resolution with production fail-closed FakeSsh fence.

VAL-GPU-028: production settings must not silently fake silicon.
``HYPER_SSH_TRANSPORT=fake`` requires ``HYPER_ALLOW_FAKE_SSH=true`` (tests/CI
only). Real transport is opt-in to a later allowlist executor feature; until
then ``real`` also fails closed rather than falling back to FakeSsh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hypercluster.probe.fixtures import (
    FakeSshFixture,
    get_fixture,
    load_fixture_json,
    load_named_fixture,
)
from hypercluster.probe.transport import FakeSshTransport, SshTransport
from hypercluster.settings import HyperSettings, get_hyper_settings

# Public error code seen by API layers (503) when product cannot probe.
SSH_TRANSPORT_UNAVAILABLE = "ssh_transport_unavailable"
FAKE_SSH_NOT_ALLOWED = "fake_ssh_not_allowed"


class TransportConfigError(Exception):
    """Configuration / policy error resolving a probe transport."""

    def __init__(self, code: str, message: str, *, status_code: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def resolve_ssh_transport(
    settings: HyperSettings | None = None,
    *,
    fixture_name: str | None = None,
    fixture_path: str | Path | None = None,
    scripted: dict[str, Any] | None = None,
    real_transport: SshTransport | None = None,
) -> SshTransport:
    """Resolve FakeSsh or RealSsh according to HYPER_* policy.

    * ``ssh_transport=fake`` only when ``allow_fake_ssh`` is true.
    * ``ssh_transport=real`` uses ``real_transport`` when injected (unit
      tests / future executor); otherwise fails closed with
      ``ssh_transport_unavailable`` — **never** silent-fake.
    """

    cfg = settings if settings is not None else get_hyper_settings()
    kind = (cfg.ssh_transport or "real").strip().lower()

    if kind in {"fake", "fakessh", "sim"}:
        if not cfg.allow_fake_ssh:
            raise TransportConfigError(
                FAKE_SSH_NOT_ALLOWED,
                "FakeSsh is disabled; set HYPER_ALLOW_FAKE_SSH=true for tests/CI only",
                status_code=503,
            )
        path = fixture_path if fixture_path is not None else cfg.fake_ssh_script
        name = fixture_name if fixture_name is not None else cfg.fake_ssh_fixture
        fx = _select_fake_fixture(
            fixture_name=name,
            fixture_path=path,
            scripted=scripted,
        )
        return FakeSshTransport(scripted=fx.scripted, name="fake")

    if kind in {"real", "ssh", "allowlist"}:
        if real_transport is not None:
            return real_transport
        raise TransportConfigError(
            SSH_TRANSPORT_UNAVAILABLE,
            "real SSH transport is not configured (no key/executor); refuse silent FakeSsh",
            status_code=503,
        )

    raise TransportConfigError(
        SSH_TRANSPORT_UNAVAILABLE,
        f"unknown HYPER_SSH_TRANSPORT={cfg.ssh_transport!r}",
        status_code=503,
    )


def resolve_fake_fixture(
    settings: HyperSettings | None = None,
    *,
    fixture_name: str | None = None,
    fixture_path: str | Path | None = None,
) -> FakeSshFixture:
    """Load the FakeSsh fixture context (claimed/occupied) for a CI probe run.

    Does not construct the transport; use :func:`resolve_ssh_transport` for that.
    Still enforces the FakeSsh allow fence when settings demand fake mode.
    """

    cfg = settings if settings is not None else get_hyper_settings()
    kind = (cfg.ssh_transport or "real").strip().lower()
    if kind in {"fake", "fakessh", "sim"} and not cfg.allow_fake_ssh:
        raise TransportConfigError(
            FAKE_SSH_NOT_ALLOWED,
            "FakeSsh is disabled; set HYPER_ALLOW_FAKE_SSH=true for tests/CI only",
            status_code=503,
        )
    return _select_fake_fixture(fixture_name=fixture_name, fixture_path=fixture_path)


def _select_fake_fixture(
    *,
    fixture_name: str | None,
    fixture_path: str | Path | None,
    scripted: dict[str, Any] | None = None,
) -> FakeSshFixture:
    if fixture_path is not None:
        return load_fixture_json(fixture_path)
    if scripted is not None:
        # Minimal wrapper: caller already built outcomes.
        from hypercluster.probe.transport import FakeOutcome
        from hypercluster.probe.types import ClaimedInventory

        normalized: dict[str, FakeOutcome] = {}
        for cid, val in scripted.items():
            if isinstance(val, FakeOutcome):
                normalized[str(cid)] = val
            elif isinstance(val, dict):
                from hypercluster.probe.fixtures import outcome_from_dict

                normalized[str(cid)] = outcome_from_dict(val)
            else:
                raise TypeError(f"unsupported scripted value for {cid!r}")
        return FakeSshFixture(
            name="custom",
            scripted=normalized,
            claimed=ClaimedInventory(gpu_model="1V100.6V", gpu_count=1),
        )
    name = fixture_name or "pass_all"
    try:
        return load_named_fixture(name, prefer_json=True)
    except KeyError:
        return get_fixture("pass_all") if name in {"", "pass_all"} else get_fixture(name)


def fake_ssh_permitted(settings: HyperSettings | None = None) -> bool:
    """Whether FakeSsh is explicitly allowed under current settings."""

    cfg = settings if settings is not None else get_hyper_settings()
    return bool(cfg.allow_fake_ssh) and (cfg.ssh_transport or "").lower() in {
        "fake",
        "fakessh",
        "sim",
    }


__all__ = [
    "FAKE_SSH_NOT_ALLOWED",
    "SSH_TRANSPORT_UNAVAILABLE",
    "TransportConfigError",
    "fake_ssh_permitted",
    "resolve_fake_fixture",
    "resolve_ssh_transport",
]
