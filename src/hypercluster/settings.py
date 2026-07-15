"""Challenge settings (`CHALLENGE_*`) and product knobs (`HYPER_*`)."""

from __future__ import annotations

from functools import lru_cache

from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.version import API_VERSION, SDK_CONTRACT_VERSION
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Canonical absolute SQLite URL under the challenge data volume.
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:////data/challenge.sqlite3"


class Settings(ChallengeSettings):
    """Hypercluster challenge settings.

    Env prefix remains `CHALLENGE_` for Base SDK compatibility (host, port,
    database_url, shared token). Default DB URL is challenge SQLite on `/data`.
    Product knobs live on :class:`HyperSettings` under the `HYPER_` prefix so
    they never collide with Base `CHALLENGE_*` identity fields.
    """

    model_config = SettingsConfigDict(env_prefix="CHALLENGE_", extra="forbid")

    slug: str = "hypercluster"
    name: str = "Hypercluster"
    version: str = "0.1.0"
    api_version: str = API_VERSION
    sdk_version: str = SDK_CONTRACT_VERSION
    database_url: str = DEFAULT_DATABASE_URL
    # Local/dev default allows env-only configuration; containers should mount
    # the shared file at the default Base secret path.
    shared_token_file: str | None = Field(
        default="/run/secrets/base/challenge_token",
        repr=False,
    )


class HyperSettings(BaseSettings):
    """Hypercluster-only product knobs (`HYPER_*` env).

    These alter background work, scoring windows, and TEE mode but must never
    rename or relocate Base `/health` `/ready` `/version` contracts.
    """

    model_config = SettingsConfigDict(env_prefix="HYPER_", extra="ignore")

    combined_worker: bool = False
    combined_worker_interval_seconds: float = Field(default=5.0, ge=0.05)
    tee_live: bool = False
    tee_bonus_tdx: float = Field(default=1.08, ge=1.0)
    tee_bonus_tdx_gpu: float = Field(default=1.20, ge=1.0)
    weight_push_interval_s: float = Field(default=120.0, ge=1.0)
    score_window_attempts: int = Field(default=50, ge=1)
    efficiency_floor: float = Field(default=0.0, ge=0.0)
    # Signed miner auth (marketplace write routes). Insecure HMAC mode is for
    # local/tests (matches peer's allow_insecure_signatures pattern); set false
    # in production so only substrate hotkey signatures verify.
    allow_insecure_signatures: bool = True
    signature_ttl_seconds: int = Field(default=300, ge=30)
    node_liveness_seconds: int = Field(default=120, ge=5)
    # Marketplace offer hard caps (VAL-MKT-010 / VAL-MKT-011).
    # Env: HYPER_MAX_OFFER_PRICE_PER_HOUR, HYPER_MAX_OFFER_LIFETIME_HOURS.
    max_offer_price_per_hour: float = Field(default=1000.0, gt=0)
    max_offer_lifetime_hours: float = Field(default=720.0, gt=0)
    # Job admit static gates (VAL-JOB-001..003). Comma-separated digests allowlist.
    # Env: HYPER_JOB_IMAGE_ALLOWLIST, HYPER_MAX_JOB_* .
    job_image_allowlist: str = Field(
        default=(
            "sha256:sim000000000000000000000000000000000000000000000000000000000001,"
            "sha256:cafebabecafebabecafebabecafebabecafebabecafebabecafebabecafebabe,"
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
    )
    max_job_world_size: int = Field(default=64, ge=1)
    max_job_nnodes: int = Field(default=16, ge=1)
    max_job_nproc_per_node: int = Field(default=8, ge=1)
    max_job_timeout_s: int = Field(default=86400, ge=1)
    max_job_gpu_budget: int = Field(default=128, ge=1)
    # Local sim job lifecycle (VAL-JOB-006/008). Zero delays for unit tests;
    # combined worker applies run sleep so cancel/timeout races are testable.
    sim_job_step_delay_s: float = Field(default=0.0, ge=0.0)
    sim_job_run_sleep_s: float = Field(default=0.0, ge=0.0)
    # Capacity binding + queue scaling (VAL-JOB-013..019, 022..024).
    # When True (default), sim invents a synthetic lease/pod bind so lifecycle
    # smoke works without a marketplace rental. When False, jobs without an
    # explicit valid lease/pod stay placing until capacity or capacity fail.
    sim_auto_capacity: bool = True
    # Forced launch failure for integrity tests (VAL-JOB-018).
    sim_launch_fail: bool = False
    # Launcher honesty injects (VAL-FAB-013/014/015/025).
    sim_launch_timeout: bool = False
    sim_launch_inject_sleep_s: float = Field(default=0.0, ge=0.0)
    sim_honesty_level: str = Field(default="l1")  # l0|l1|l2
    sim_inventory_spoof: bool = False
    # Concurrent multi-node / large-job budget (VAL-JOB-015).
    max_concurrent_large_jobs: int = Field(default=4, ge=1)
    large_job_world_size_threshold: int = Field(default=4, ge=1)
    # Aggregate world_size across concurrently running jobs.
    max_concurrent_world_size_budget: int = Field(default=64, ge=1)
    # How long a job may wait in placing without capacity (seconds).
    capacity_wait_timeout_s: float = Field(default=2.0, ge=0.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache process challenge settings (env-driven CHALLENGE_*)."""

    return Settings()


@lru_cache(maxsize=1)
def get_hyper_settings() -> HyperSettings:
    """Load and cache Hypercluster product knobs (env-driven HYPER_*)."""

    return HyperSettings()


def clear_settings_cache() -> None:
    """Drop settings caches (tests / reconfigure)."""

    get_settings.cache_clear()
    get_hyper_settings.cache_clear()


__all__ = [
    "DEFAULT_DATABASE_URL",
    "HyperSettings",
    "Settings",
    "clear_settings_cache",
    "get_hyper_settings",
    "get_settings",
]
