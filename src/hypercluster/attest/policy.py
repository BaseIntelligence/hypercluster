"""TEE verify policy: compose allowlist + TCB enforce (fail-closed defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class TeeVerifyPolicy:
    """Challenge-pinned offline/live verification policy.

    Defaults match CI enforce path:
    - compose allowlist fail-closed (empty = accept nothing)
    - tcb_enforce True rejects non-acceptable TCB statuses
    - disallowed_advisory_ids reject when any present under enforce-on
    """

    compose_allowlist: frozenset[str] = field(default_factory=frozenset)
    tcb_enforce: bool = True
    acceptable_tcb_statuses: frozenset[str] = field(default_factory=lambda: frozenset({"UpToDate"}))
    disallowed_advisory_ids: frozenset[str] = field(default_factory=frozenset)

    def is_compose_allowed(self, compose_hash: str | None) -> bool:
        if not compose_hash:
            return False
        if not self.compose_allowlist:
            return False
        return compose_hash in self.compose_allowlist


def default_policy_from_settings() -> TeeVerifyPolicy:
    """Build policy from HyperSettings (HYPER_* env)."""

    from hypercluster.settings import get_hyper_settings

    hs = get_hyper_settings()
    allow = frozenset(hs.compose_hash_allowlist_set())
    disallowed = frozenset(hs.tee_disallowed_advisories_set())
    acceptable = frozenset(hs.tee_acceptable_tcb_set())
    return TeeVerifyPolicy(
        compose_allowlist=allow,
        tcb_enforce=bool(hs.tee_tcb_enforce),
        acceptable_tcb_statuses=acceptable or frozenset({"UpToDate"}),
        disallowed_advisory_ids=disallowed,
    )


# Canonical default golden compose hash used by offline fixtures / tests.
DEFAULT_COMPOSE_HASH_GOLDEN = (
    "sha256:0c0ffeec0a5eabcdef0123456789abcdef0123456789abcdef0123456789ab"
)


__all__ = [
    "DEFAULT_COMPOSE_HASH_GOLDEN",
    "TeeVerifyPolicy",
    "default_policy_from_settings",
]
