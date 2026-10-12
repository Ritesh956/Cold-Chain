"""Retention for the readings table.

The cut-off is relative to the NEWEST stored reading (event time), not to the wall clock, so replaying an
old history (the simulator, the demo stack) never purges data that just arrived. Alerts and incident
reports are small and are kept. The floor on `days` keeps every window a consumer needs: the breach
model reads 25 hours of history and the drift monitor 24 hours.
"""

from __future__ import annotations

from datetime import timedelta

from coldchain_core.repo import Repo

MIN_RETENTION_DAYS = 3


def validate(days: int) -> int:
    if days != 0 and days < MIN_RETENTION_DAYS:
        raise ValueError(
            f"RETENTION_DAYS must be 0 (keep everything) or at least {MIN_RETENTION_DAYS}, got {days}: "
            "the scorers and the drift monitor read up to a day of history"
        )
    return days


async def purge_once(repo: Repo, days: int) -> int:
    """Delete readings older than `days` before the newest reading. Returns rows removed."""
    if days <= 0:
        return 0
    newest = None
    for zone in await repo.zones():
        latest = await repo.latest(zone)
        if latest is not None and (newest is None or latest.ts > newest):
            newest = latest.ts
    if newest is None:
        return 0
    return await repo.purge_readings_before(newest - timedelta(days=days))
