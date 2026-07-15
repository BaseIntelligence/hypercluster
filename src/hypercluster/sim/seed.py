"""Deterministic sim seed helpers (VAL-CLI-013).

``hypercluster sim seed`` and library callers share one resolution path so the
same seed via ``--seed`` or ``HYPER_SIM_SEED`` always rebuilds identical node
IDs / digests (never random UUIDs).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Any

from hypercluster.fabric.discovery import DIGEST_PREFIX, canonical_json
from hypercluster.sim.inventory import SimInventory

DEFAULT_SIM_SEED = 0
SIM_SEED_ENV = "HYPER_SIM_SEED"


def resolve_sim_seed(
    seed: int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    default: int = DEFAULT_SIM_SEED,
) -> int:
    """Resolve inventory seed from explicit value, then env, then default.

    Priority:
    1. Explicit *seed* when not ``None`` (CLI ``--seed`` after Typer fills)
    2. ``HYPER_SIM_SEED`` environment variable (integer string)
    3. *default* (``0``)

    When Typer provides a default of 0 and env is set, callers should pass the
    env-aware Option default via :func:`cli_seed_option_default` so env wins
    over the literal zero when the user omitted ``--seed``.
    """

    if seed is not None:
        return int(seed)
    env = environ if environ is not None else os.environ
    raw = (env.get(SIM_SEED_ENV) or "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw, 10)
    except ValueError as exc:
        raise ValueError(
            f"{SIM_SEED_ENV}={raw!r} is not an integer seed"
        ) from exc


def cli_seed_option_default() -> int:
    """Typer default factory: prefer ``HYPER_SIM_SEED`` over fixed 0."""

    return resolve_sim_seed(None)


def inventory_shape_digest(inventory: SimInventory) -> str:
    """Stable digest over counts, node IDs, graph digest, and report digests.

    Used by tests and operator output to prove two ``sim seed`` runs matched
    without comparing full public payloads.
    """

    body: dict[str, Any] = {
        "seed": inventory.seed,
        "node_count": len(inventory.nodes),
        "node_ids": [n.node_id for n in inventory.nodes],
        "ib_edge_count": len(inventory.ib_edges),
        "nvlink_edge_count": len(inventory.nvlink_edges),
        "graph_digest": inventory.graph_digest,
        "report_digests": [n.fabric_report.report_digest for n in inventory.nodes],
    }
    return DIGEST_PREFIX + hashlib.sha256(
        canonical_json(body).encode()
    ).hexdigest()


__all__ = [
    "DEFAULT_SIM_SEED",
    "SIM_SEED_ENV",
    "cli_seed_option_default",
    "inventory_shape_digest",
    "resolve_sim_seed",
]
