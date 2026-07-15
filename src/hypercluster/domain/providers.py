"""Provider register / list / heartbeat domain service."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import Provider, utc_now


class ProviderError(Exception):
    """Domain error for provider operations."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


async def register_provider(
    session: AsyncSession,
    *,
    hotkey: str,
    display_name: str | None = None,
) -> tuple[Provider, bool]:
    """Create or return existing provider for hotkey (idempotent).

    Returns ``(provider, created)``. Same hotkey maps to a single row; second
    register is idempotent (same id, 2xx) as required by VAL-MKT-001.
    """

    hotkey = hotkey.strip()
    if not hotkey:
        raise ProviderError("invalid_hotkey", "hotkey must be non-empty", status_code=400)

    result = await session.execute(select(Provider).where(Provider.hotkey == hotkey))
    existing = result.scalar_one_or_none()
    if existing is not None:
        # Idempotent: optionally refresh display_name if provided and empty before.
        if display_name and not existing.display_name:
            existing.display_name = display_name
            existing.updated_at = utc_now()
            await session.commit()
            await session.refresh(existing)
        return existing, False

    now = utc_now()
    provider = Provider(
        id=str(uuid.uuid4()),
        hotkey=hotkey,
        display_name=display_name,
        status="active",
        last_seen_at=now,
        created_at=now,
        updated_at=now,
    )
    session.add(provider)
    await session.commit()
    await session.refresh(provider)
    return provider, True


async def list_providers(
    session: AsyncSession,
    *,
    hotkey: str | None = None,
) -> list[Provider]:
    """List providers; when hotkey given, restrict to that owner."""

    stmt = select(Provider).order_by(Provider.created_at.asc())
    if hotkey:
        stmt = stmt.where(Provider.hotkey == hotkey)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_provider_by_hotkey(session: AsyncSession, hotkey: str) -> Provider | None:
    result = await session.execute(select(Provider).where(Provider.hotkey == hotkey))
    return result.scalar_one_or_none()


async def get_provider(session: AsyncSession, provider_id: str) -> Provider | None:
    result = await session.execute(select(Provider).where(Provider.id == provider_id))
    return result.scalar_one_or_none()


async def provider_heartbeat(
    session: AsyncSession,
    *,
    hotkey: str,
) -> Provider:
    """Advance liveness for the provider bound to hotkey.

    Does not mutate ``id`` or ``hotkey`` (VAL-MKT-003).
    """

    provider = await get_provider_by_hotkey(session, hotkey)
    if provider is None:
        raise ProviderError(
            "provider_not_found",
            "provider not registered for hotkey",
            status_code=404,
        )

    # Capture identity so we never reassign identity fields.
    identity_id = provider.id
    identity_hotkey = provider.hotkey

    now = utc_now()
    provider.last_seen_at = now
    provider.updated_at = now
    # Explicitly keep status active on healthy heartbeat when not banned.
    if provider.status not in {"suspended", "banned"}:
        provider.status = "active"

    await session.commit()
    await session.refresh(provider)

    # Invariant: heartbeat never mutates identity.
    if provider.id != identity_id or provider.hotkey != identity_hotkey:
        raise ProviderError(
            "identity_mutated",
            "heartbeat must not change provider id or hotkey",
            status_code=500,
        )
    return provider


def provider_to_public(provider: Provider) -> dict[str, Any]:
    return provider.to_dict()


__all__ = [
    "ProviderError",
    "get_provider",
    "get_provider_by_hotkey",
    "list_providers",
    "provider_heartbeat",
    "provider_to_public",
    "register_provider",
]
