from __future__ import annotations

from datetime import timedelta

import pytest
from coldchain_core.config import Settings
from coldchain_core.repo import MemoryRepo
from coldchain_core.retention import MIN_RETENTION_DAYS, purge_once, validate
from core_helpers import T0, reading

DAY = 1440


def day_rows(zone: str, days: int):
    return [reading(zone, i) for i in range(0, days * DAY, 60)]  # hourly is enough for a retention test


async def test_cutoff_is_relative_to_the_newest_reading_not_the_wall_clock():
    repo = MemoryRepo()
    await repo.upsert_readings(day_rows("A", 10))  # readings from 2025, long before "now"
    removed = await purge_once(repo, 5)
    assert removed > 0  # data from 2025 is only old relative to its own newest reading
    first = (await repo.window("A", T0 + timedelta(days=10), 10 * DAY))[0]
    newest = await repo.latest("A")
    assert newest.ts - first.ts <= timedelta(days=5)


async def test_uses_the_newest_reading_across_all_zones():
    repo = MemoryRepo()
    await repo.upsert_readings([*day_rows("A", 4), *day_rows("B", 10)])
    await purge_once(repo, 5)
    # B's newest reading is about day 10, so the cut-off is about day 5: all of zone A (days 0-4) is past it
    assert await repo.latest("A") is None
    oldest_b = (await repo.window("B", T0 + timedelta(days=10), 10 * DAY))[0].ts
    assert oldest_b >= T0 + timedelta(days=4, hours=23)


async def test_zero_days_keeps_everything_and_empty_store_is_a_no_op():
    repo = MemoryRepo()
    assert await purge_once(repo, 7) == 0
    await repo.upsert_readings(day_rows("A", 6))
    assert await purge_once(repo, 0) == 0
    assert len(await repo.window("A", T0 + timedelta(days=6), 10 * DAY)) == 6 * 24


def test_retention_must_be_zero_or_long_enough_for_the_scorers():
    assert validate(0) == 0 and validate(MIN_RETENTION_DAYS) == MIN_RETENTION_DAYS
    for bad in (1, 2, -1):
        with pytest.raises(ValueError, match="at least"):
            validate(bad)


def test_settings_reject_a_dangerous_retention_from_the_environment(monkeypatch):
    monkeypatch.setenv("RETENTION_DAYS", "1")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("RETENTION_DAYS", "14")
    assert Settings.from_env().retention_days == 14
