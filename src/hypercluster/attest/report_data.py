"""report_data layout: job_id ‖ image_digest ‖ nonce → 64-byte TDX field.

Architecture §9: bind report_data as job digest + nonce (left-aligned zero pad).
Canonical stitch used by offline fixtures and live path:

  job_digest   = SHA256( "hypercluster-job-v1" ‖ job_id ‖ image_digest )   # 32 B
  nonce_digest = SHA256( "hypercluster-nonce-v1" ‖ nonce )                # 32 B
  report_data  = job_digest ‖ nonce_digest                                  # 64 B

Parser fail-closed on truncated / extra trailing / non-hex payloads.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

REPORT_DATA_SIZE = 64
JOB_DIGEST_SIZE = 32
NONCE_DIGEST_SIZE = 32

_TAG_JOB = b"hypercluster-job-v1"
_TAG_NONCE = b"hypercluster-nonce-v1"


class ReportDataLayoutError(ValueError):
    """Malformed report_data length/layout (VAL-TEE-011)."""


@dataclass(slots=True, frozen=True)
class ParsedReportData:
    """Parsed 64-byte report_data field."""

    job_digest: bytes
    nonce_digest: bytes
    raw: bytes

    def matches(self, other: ParsedReportData) -> bool:
        return self.job_digest == other.job_digest and self.nonce_digest == other.nonce_digest


def build_job_digest(*, job_id: str, image_digest: str) -> bytes:
    """32-byte job binding digest (stable for fixed job_id + image_digest)."""

    if not job_id or not str(job_id).strip():
        raise ValueError("job_id is required")
    if not image_digest or not str(image_digest).strip():
        raise ValueError("image_digest is required")
    preimage = _TAG_JOB + b"\x00" + job_id.encode() + b"\x00" + image_digest.encode()
    return hashlib.sha256(preimage).digest()


def build_nonce_digest(*, nonce: str) -> bytes:
    if not nonce or not str(nonce).strip():
        raise ValueError("nonce is required")
    return hashlib.sha256(_TAG_NONCE + b"\x00" + nonce.encode()).digest()


def build_report_data(*, job_id: str, image_digest: str, nonce: str) -> bytes:
    """Build the canonical 64-byte report_data field."""

    jd = build_job_digest(job_id=job_id, image_digest=image_digest)
    nd = build_nonce_digest(nonce=nonce)
    assert len(jd) == JOB_DIGEST_SIZE
    assert len(nd) == NONCE_DIGEST_SIZE
    return jd + nd


def parse_report_data(value: bytes | bytearray | str) -> ParsedReportData:
    """Parse and validate exact 64-byte report_data (fail-closed layout).

    Accepts raw bytes or hex string. Rejects truncated (too short) and extra
    trailing (too long) before any crypto accept.
    """

    if isinstance(value, str):
        hex_s = value.strip().lower().removeprefix("0x")
        if len(hex_s) % 2 != 0:
            raise ReportDataLayoutError("report_data hex has odd length (layout)")
        try:
            raw = bytes.fromhex(hex_s)
        except ValueError as exc:
            raise ReportDataLayoutError(f"report_data hex is malformed: {exc}") from exc
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raise TypeError(f"report_data must be bytes or hex str, got {type(value).__name__}")

    if len(raw) < REPORT_DATA_SIZE:
        raise ReportDataLayoutError(
            f"report_data truncated: got {len(raw)} bytes, need {REPORT_DATA_SIZE}"
        )
    if len(raw) > REPORT_DATA_SIZE:
        raise ReportDataLayoutError(
            f"report_data extra trailing: got {len(raw)} bytes, need {REPORT_DATA_SIZE}"
        )

    return ParsedReportData(
        job_digest=raw[:JOB_DIGEST_SIZE],
        nonce_digest=raw[JOB_DIGEST_SIZE:],
        raw=raw,
    )


def report_data_hex(*, job_id: str, image_digest: str, nonce: str) -> str:
    return build_report_data(job_id=job_id, image_digest=image_digest, nonce=nonce).hex()


__all__ = [
    "JOB_DIGEST_SIZE",
    "NONCE_DIGEST_SIZE",
    "REPORT_DATA_SIZE",
    "ParsedReportData",
    "ReportDataLayoutError",
    "build_job_digest",
    "build_nonce_digest",
    "build_report_data",
    "parse_report_data",
    "report_data_hex",
]
