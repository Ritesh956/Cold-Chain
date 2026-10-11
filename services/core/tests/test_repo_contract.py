"""The same contract runs against MemoryRepo and (when TEST_DATABASE_URL is set) real Postgres,
so the in-memory fake used everywhere else is proven to behave like the real thing."""

from __future__ import annotations

import os

import pytest
from coldchain_core.pg import PgRepo
from coldchain_core.repo import MemoryRepo
from coldchain_schemas import Alert, AlertKind
from core_helpers import MIN, T0, history, reading

PG_URL = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.integration)])
async def repo(request):
    if request.param == "memory":
        r = MemoryRepo()
        await r.init()
        yield r
        return
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg

    with psycopg.connect(PG_URL, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS readings CASCADE; DROP TABLE IF EXISTS zones, alerts, scoring_state CASCADE")
    r = PgRepo(PG_URL)
    await r.init()
    yield r
    await r.close()


async def test_ping_and_empty_state(repo):
    assert await repo.ping()
    assert await repo.zones() == [] and await repo.latest("A") is None
    assert await repo.open_alerts() == [] and await repo.list_alerts(False, None, 10) == []


async def test_upsert_is_idempotent_and_reports_new_rows(repo):
    rows = history("A", 5)
    assert await repo.upsert_readings(rows) == 5
    assert await repo.upsert_readings(rows) == 0  # full replay: nothing new
    assert await repo.upsert_readings([*rows[3:], reading("A", 5)]) == 1  # 2 replayed + 1 new
    assert await repo.upsert_readings([]) == 0


async def test_first_write_wins_on_conflicting_duplicate(repo):
    await repo.upsert_readings([reading("A", 0, temp=-25.0)])
    await repo.upsert_readings([reading("A", 0, temp=99.0)])
    assert (await repo.latest("A")).temp_c == -25.0


async def test_zones_latest_and_null_temperature_roundtrip(repo):
    await repo.upsert_readings([*history("B", 3), reading("A", 7, temp=None, on=False, cur=0.3)])
    assert await repo.zones() == ["A", "B"]
    a = await repo.latest("A")
    assert (a.ts, a.temp_c, a.compressor_on, a.compressor_current_a) == (T0 + 7 * MIN, None, False, 0.3)
    assert (await repo.latest("B")).ts == T0 + 2 * MIN
    assert a.ts.tzinfo is not None  # timezone-aware, UTC


async def test_window_is_inclusive_ascending_and_adds_one_prior_row(repo):
    await repo.upsert_readings([r for r in history("A", 30) if r.ts != T0 + 9 * MIN])  # 9 missing
    w = await repo.window("A", T0 + 19 * MIN, 10)  # range 10..19
    assert [r.ts for r in w] == [T0 + i * MIN for i in range(8, 20) if i != 9]  # 8 (prior) then 10..19
    assert w[0].ts == T0 + 8 * MIN


async def test_window_with_no_earlier_data_and_other_zone_isolation(repo):
    await repo.upsert_readings([*history("A", 10), *history("B", 10, temp=-10.0)])
    w = await repo.window("A", T0 + 9 * MIN, 10)
    assert [r.ts for r in w] == [T0 + i * MIN for i in range(10)]
    assert all(r.temp_c == -25.0 for r in w)
    assert await repo.window("C", T0 + 9 * MIN, 10) == []


async def test_scoring_state_roundtrip_and_update(repo):
    assert await repo.get_scoring_state("A", "anomaly") is None
    await repo.set_scoring_state("A", "anomaly", T0)
    await repo.set_scoring_state("A", "breach", T0 + MIN)
    await repo.set_scoring_state("A", "anomaly", T0 + 5 * MIN)  # update, not duplicate
    assert await repo.get_scoring_state("A", "anomaly") == T0 + 5 * MIN
    assert await repo.get_scoring_state("A", "breach") == T0 + MIN
    assert await repo.get_scoring_state("B", "anomaly") is None


def _alert(zone="A", kind=AlertKind.ANOMALY, i=0, **kw):
    return Alert(
        zone=zone,
        kind=kind,
        severity="warning",
        opened_at=T0 + i * MIN,
        peak_score=3.0,
        summary="s",
        model_version="1",
        **kw,
    )


async def test_alert_insert_update_and_filters(repo):
    a = await repo.save_alert(_alert(i=1))
    assert a.id is not None and a.closed_at is None
    b = await repo.save_alert(_alert("B", AlertKind.BREACHED, i=2))
    c = await repo.save_alert(_alert(i=0))

    closed = a.model_copy(update={"closed_at": T0 + 10 * MIN, "peak_score": 8.5})
    assert (await repo.save_alert(closed)).id == a.id  # update in place, no new row

    assert {x.id for x in await repo.open_alerts()} == {b.id, c.id}
    everything = await repo.list_alerts(False, None, 10)
    assert [x.id for x in everything] == [b.id, a.id, c.id]  # newest opened_at first
    stored = next(x for x in everything if x.id == a.id)
    assert stored.closed_at == T0 + 10 * MIN and stored.peak_score == 8.5
    assert [x.id for x in await repo.list_alerts(True, "A", 10)] == [c.id]
    assert len(await repo.list_alerts(False, None, 2)) == 2
    assert stored.kind is AlertKind.ANOMALY and stored.opened_at.tzinfo is not None
