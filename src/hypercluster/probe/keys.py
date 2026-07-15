"""SSH key_ref resolution without ever placing raw PEM into evidence/DB.

Rules (VAL-GPU-031 / design §3.1):
- Resolve keys from file (0600) or env ref name only.
- Surface at most key_fingerprint + key_ref {kind, name}.
- API request bodies cannot carry private_key / pem fields.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

KeyRefKind = Literal["env", "file", "agent"]


class KeyMaterialError(Exception):
    """Key resolution / request-wash failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class KeyRef:
    kind: KeyRefKind
    name: str

    def to_public(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name}


@dataclass(slots=True)
class ResolvedKey:
    """In-process only; never serialize into GpuHostEvidence."""

    path: Path | None
    pem_bytes: bytes
    fingerprint: str
    ref: KeyRef


_BODY_FORBIDDEN = frozenset(
    {
        "private_key",
        "privatekey",
        "key_pem",
        "ssh_private_key",
        "ssh_key_pem",
        "pem",
        "id_rsa",
        "openssh_private_key",
    }
)


def reject_body_private_key_fields(body: dict[str, Any] | None) -> None:
    """Reject probe API bodies that smuggle raw private key material (422 path)."""

    if not body:
        return
    for key, value in body.items():
        lower = str(key).lower().replace("-", "_")
        if lower in _BODY_FORBIDDEN or (
            "private" in lower and "key" in lower
        ) or lower.endswith("_pem"):
            raise KeyMaterialError(
                "private_key_not_allowed",
                f"request field {key!r} is not allowed; use key_ref only",
            )
        if isinstance(value, str) and _looks_like_pem(value):
            raise KeyMaterialError(
                "private_key_not_allowed",
                f"request field {key!r} contains PEM private key material",
            )
        if isinstance(value, dict):
            # Nested wash (e.g. {key: {pem: "..."}})
            reject_body_private_key_fields(value)


def _looks_like_pem(text: str) -> bool:
    upper = text.upper()
    return "BEGIN" in upper and "PRIVATE KEY" in upper


def compute_key_fingerprint(material: bytes | str) -> str:
    """Stable fingerprint for evidence: sha256 of key material (public or private bytes).

    We hash whatever bytes are used for auth so reloads share a stable id
    without storing the private key itself. Prefer hashing the public key
    when available; here file contents are the only material.
    """

    data = material if isinstance(material, bytes) else material.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def resolve_key_ref(ref: KeyRef) -> ResolvedKey:
    """Resolve file/env key_ref to path (+ load for fingerprint). Agent reserved."""

    if ref.kind == "file":
        path = Path(ref.name).expanduser()
        if not path.is_file():
            raise KeyMaterialError("key_not_found", f"SSH key file not found: {path}")
        pem = path.read_bytes()
        return ResolvedKey(
            path=path,
            pem_bytes=pem,
            fingerprint=compute_key_fingerprint(pem),
            ref=ref,
        )

    if ref.kind == "env":
        raw = os.environ.get(ref.name)
        if raw is None or raw == "":
            raise KeyMaterialError(
                "key_not_found",
                f"environment variable {ref.name!r} is empty / unset",
            )
        # Env may hold a path to a key file (preferred) or raw PEM for tests.
        candidate = Path(raw).expanduser()
        if candidate.is_file() and not _looks_like_pem(raw):
            pem = candidate.read_bytes()
            return ResolvedKey(
                path=candidate,
                pem_bytes=pem,
                fingerprint=compute_key_fingerprint(pem),
                ref=ref,
            )
        pem_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if not _looks_like_pem(raw):
            # Treat as opaque path that didn't exist
            raise KeyMaterialError(
                "key_not_found",
                f"env {ref.name!r} is not a readable key path and not PEM",
            )
        return ResolvedKey(
            path=None,
            pem_bytes=pem_bytes,
            fingerprint=compute_key_fingerprint(pem_bytes),
            ref=ref,
        )

    if ref.kind == "agent":
        raise KeyMaterialError(
            "key_agent_unsupported",
            "ssh-agent key_ref is reserved; use kind=file|env for v1",
        )

    raise KeyMaterialError("invalid_key_ref", f"unknown key_ref.kind={ref.kind!r}")


def load_private_key_material(ref: KeyRef) -> ResolvedKey:
    """Alias used by tests/ops; same as :func:`resolve_key_ref`."""

    return resolve_key_ref(ref)


def public_key_meta_for_evidence(
    ref: KeyRef,
    *,
    fingerprint: str,
) -> dict[str, Any]:
    """Metadata safe to store on evidence / logs (no PEM)."""

    return {
        "key_ref": ref.to_public(),
        "key_fingerprint": fingerprint,
    }


def key_ref_from_settings(
    *,
    ssh_key_path: str | None,
    ssh_key_env: str | None = None,
) -> KeyRef | None:
    """Build a KeyRef from process settings (product default)."""

    if ssh_key_path:
        return KeyRef(kind="file", name=ssh_key_path)
    if ssh_key_env:
        return KeyRef(kind="env", name=ssh_key_env)
    # Conventional fall-backs (do not invent PEM)
    env_path = os.environ.get("HYPER_SSH_KEY_PATH")
    if env_path:
        return KeyRef(kind="env", name="HYPER_SSH_KEY_PATH")
    return None


__all__ = [
    "KeyMaterialError",
    "KeyRef",
    "KeyRefKind",
    "ResolvedKey",
    "compute_key_fingerprint",
    "key_ref_from_settings",
    "load_private_key_material",
    "public_key_meta_for_evidence",
    "reject_body_private_key_fields",
    "resolve_key_ref",
]
