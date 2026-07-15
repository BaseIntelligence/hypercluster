"""Offline TEE quote envelopes for CI (no hardware / network).

Fixtures are JSON envelopes packaged as base64 for TeeVerifyRequest.quote_b64.
Layout is challenge-owned offline_fixture v1 (not a raw Intel quote re-parse).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hypercluster.attest.report_data import REPORT_DATA_SIZE, build_report_data

OFFLINE_FIXTURE_KIND = "hypercluster.offline_quote.v1"


class OfflineQuoteEnvelope(BaseModel):
    """In-process offline TDX quote stand-in for CI.

    Fields mirror the subset aleged by live dstack-verifier responses so the
    same TeeVerifyResult shape is produced offline and online.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["hypercluster.offline_quote.v1"] = "hypercluster.offline_quote.v1"
    quote_version: int = 1
    tee_type: str = "tdx"
    compose_hash: str
    # Optional measurement target used to surface compose_hash_mismatch when
    # the envelope's actual compose_hash differs from the expected pin.
    expected_compose_hash: str | None = None
    report_data_hex: str
    tcb_status: str = "UpToDate"
    advisory_ids: list[str] = Field(default_factory=list)
    os_image_hash: str | None = None
    event_log: str | None = None
    event_log_ok: bool = True
    quote_sig_ok: bool = True
    job_id: str | None = None
    image_digest: str | None = None
    nonce: str | None = None
    vm_config: dict[str, Any] | None = None
    gpu_evidence: dict[str, Any] | None = None
    fixture_id: str | None = None

    @field_validator("compose_hash")
    @classmethod
    def _compose_shape(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("compose_hash is required")
        return v

    @field_validator("report_data_hex")
    @classmethod
    def _report_data_hex_shape(cls, value: str) -> str:
        # Soft shape only — full layout reject lives in parse_report_data /
        # verify so truncated/extra fixtures still package and fail closed.
        return (value or "").strip().lower()


def package_quote_b64(envelope: OfflineQuoteEnvelope) -> str:
    """Serialize envelope as standard base64 (UTF-8 JSON body)."""

    body = envelope.model_dump(mode="json")
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode("ascii")


def unpack_quote_b64(quote_b64: str) -> OfflineQuoteEnvelope:
    """Decode quote_b64 into OfflineQuoteEnvelope; raises ValueError on junk."""

    if not quote_b64 or not str(quote_b64).strip():
        raise ValueError("quote_b64 is empty")
    try:
        raw = base64.b64decode(quote_b64.encode("ascii"), validate=False)
    except Exception as exc:  # noqa: BLE001 — surface as quote invalid
        raise ValueError(f"quote_b64 is not valid base64: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"quote payload is not offline fixture JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("quote payload must be a JSON object")
    kind = payload.get("kind")
    if kind != OFFLINE_FIXTURE_KIND:
        raise ValueError(f"unknown offline quote kind: {kind!r}")
    return OfflineQuoteEnvelope.model_validate(payload)


def load_quote_fixture(path: str | Path) -> OfflineQuoteEnvelope:
    """Load OfflineQuoteEnvelope from a JSON fixture file."""

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"fixture {p} root must be object")
    # Support both bare envelope and {"envelope": {...}} wrappers.
    if "envelope" in data and isinstance(data["envelope"], dict):
        data = data["envelope"]
    if "kind" not in data:
        data = {**data, "kind": OFFLINE_FIXTURE_KIND}
    return OfflineQuoteEnvelope.model_validate(data)


def make_offline_envelope(
    *,
    compose_hash: str,
    report_data: bytes | None = None,
    tcb_status: str = "UpToDate",
    advisory_ids: list[str] | None = None,
    job_id: str | None = None,
    image_digest: str | None = None,
    nonce: str | None = None,
    expected_compose_hash: str | None = None,
    event_log_ok: bool = True,
    quote_sig_ok: bool = True,
    os_image_hash: str | None = None,
    event_log: str | None = None,
    gpu_evidence: dict[str, Any] | None = None,
    fixture_id: str | None = None,
    vm_config: dict[str, Any] | None = None,
) -> OfflineQuoteEnvelope:
    """Build a consistent offline envelope (for tests + sim)."""

    if report_data is None:
        if not (job_id and image_digest and nonce):
            raise ValueError("report_data or (job_id, image_digest, nonce) required")
        report_data = build_report_data(
            job_id=job_id, image_digest=image_digest, nonce=nonce
        )
    if len(report_data) != REPORT_DATA_SIZE:
        # Still package — verify will fail layout (used for negative fixtures).
        pass
    return OfflineQuoteEnvelope(
        compose_hash=compose_hash,
        expected_compose_hash=expected_compose_hash or compose_hash,
        report_data_hex=report_data.hex(),
        tcb_status=tcb_status,
        advisory_ids=list(advisory_ids or []),
        job_id=job_id,
        image_digest=image_digest,
        nonce=nonce,
        event_log_ok=event_log_ok,
        quote_sig_ok=quote_sig_ok,
        os_image_hash=os_image_hash,
        event_log=event_log,
        gpu_evidence=gpu_evidence,
        fixture_id=fixture_id,
        vm_config=vm_config,
    )


def write_quote_fixture(path: str | Path, envelope: OfflineQuoteEnvelope) -> Path:
    """Write envelope JSON to disk (stable sorted keys)."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = envelope.model_dump(mode="json")
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


__all__ = [
    "OFFLINE_FIXTURE_KIND",
    "OfflineQuoteEnvelope",
    "load_quote_fixture",
    "make_offline_envelope",
    "package_quote_b64",
    "unpack_quote_b64",
    "write_quote_fixture",
]
