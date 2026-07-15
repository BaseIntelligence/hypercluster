"""Challenge settings (`CHALLENGE_*` / future `HYPER_*` knobs)."""

from __future__ import annotations

from functools import lru_cache

from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.version import API_VERSION, SDK_CONTRACT_VERSION
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Settings(ChallengeSettings):
    """Hypercluster challenge settings.

    Env prefix remains `CHALLENGE_` for Base SDK compatibility. Product knobs
    that are Hypercluster-specific will land in later milestones (often `HYPER_*`
    via a nested model). Default DB URL is challenge SQLite under `/data`.
    """

    model_config = SettingsConfigDict(env_prefix="CHALLENGE_", extra="forbid")

    slug: str = "hypercluster"
    name: str = "Hypercluster"
    version: str = "0.1.0"
    api_version: str = API_VERSION
    sdk_version: str = SDK_CONTRACT_VERSION
    database_url: str = "sqlite+aiosqlite:////data/challenge.sqlite3"
    # Local/dev default allows env-only configuration; containers should mount
    # the shared file at the default Base secret path.
    shared_token_file: str | None = Field(
        default="/run/secrets/base/challenge_token",
        repr=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache process settings (env-driven)."""

    return Settings()


def clear_settings_cache() -> None:
    """Drop the settings cache (tests / reconfigure)."""

    get_settings.cache_clear()


__all__ = ["Settings", "clear_settings_cache", "get_settings"]
