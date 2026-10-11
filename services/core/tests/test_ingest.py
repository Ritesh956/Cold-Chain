from __future__ import annotations

import asyncio

import pytest
from coldchain_core.cache import MemoryCache
from coldchain_core.events import Broadcaster
from coldchain_core.ingest import IngestWorker
from coldchain_core.metrics import Metrics
from coldchain_core.repo import MemoryRepo
from conftest import FakeSource, history, reading


def val(counter, *labels) -> float:
    return (counter.labels(*labels) if labels else counter)._value.get()


class LoggingRepo(MemoryRepo):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    async def upsert_readings(self, rows):
        self.log.append("upsert")
        return await super().upsert_readings(rows)


class FailingRepo(MemoryRepo):
    async def upsert_readings(self, rows):
        raise ConnectionError("db down")


class FailingCache(MemoryCache):
    async def set_latest(self, reading):
        raise ConnectionError("redis down")


def payloads(rows):
    return [r.model_dump_json().encode() for r in rows]


def worker(source, repo=None, cache=None, bcast=None, metrics=None):
    m = metrics or Metrics()
    return IngestWorker(source, repo or MemoryRepo(), cache or MemoryCache(), bcast or Broadcaster(), m), m


async def test_messages_are_acked_only_after_the_database_commit():
    log: list[str] = []
    repo = LoggingRepo(log)
    w, _ = worker(FakeSource([payloads(history("A", 3))], log), repo)
    assert await w.process_once() == 3
    assert log == ["upsert", "ack", "ack", "ack"]
    assert len(await repo.window("A", history("A", 3)[-1].ts, 3)) == 3


async def test_redelivered_messages_are_idempotent_and_counted_as_duplicates():
    log: list[str] = []
    repo = MemoryRepo()
    batch = payloads(history("A", 4))
    w, m = worker(FakeSource([batch, batch], log), repo)
    await w.process_once()
    await w.process_once()  # at-least-once: the same messages arrive again
    assert val(m.ingested, "new") == 4 and val(m.ingested, "duplicate") == 4
    assert len(await repo.window("A", history("A", 4)[-1].ts, 10)) == 4
    assert log == ["ack"] * 8


async def test_poison_messages_are_terminated_not_redelivered_and_do_not_block_good_ones():
    log: list[str] = []
    good = payloads(history("A", 2))
    bad = [b"not json", b'{"zone": "A"}', b'{"ts": "2025-01-01T00:00:00Z", "zone": "A", "temp_c": "hot"}']
    w, m = worker(FakeSource([[*bad, *good]], log))
    await w.process_once()
    assert sorted(log) == ["ack", "ack", "term", "term", "term"]
    assert val(m.poison) == 3 and val(m.ingested, "new") == 2


async def test_database_failure_naks_everything_and_raises():
    log: list[str] = []
    w, m = worker(FakeSource([payloads(history("A", 3))], log), FailingRepo())
    with pytest.raises(ConnectionError):
        await w.process_once()
    assert log == ["nak", "nak", "nak"] and val(m.ingest_errors, "database") == 1


async def test_cache_failure_does_not_prevent_ack_or_storage():
    log: list[str] = []
    repo = MemoryRepo()
    w, m = worker(FakeSource([payloads(history("A", 2))], log), repo, FailingCache())
    await w.process_once()
    assert log == ["ack", "ack"] and val(m.ingest_errors, "cache") == 1
    assert await repo.latest("A") is not None


async def test_latest_per_zone_goes_to_cache_and_live_stream():
    rows = [*history("A", 3), *history("B", 5)]
    cache, bcast = MemoryCache(), Broadcaster()
    w, _ = worker(FakeSource([payloads(rows)], []), cache=cache, bcast=bcast)
    async with bcast.subscribe() as q:
        await w.process_once()
        events = [q.get_nowait() for _ in range(q.qsize())]
    assert {e["reading"]["zone"] for e in events} == {"A", "B"}
    assert len(events) == 2  # one per zone, not one per message
    assert (await cache.get_latest("B")).ts == rows[-1].ts


async def test_out_of_order_replay_does_not_regress_the_cache():
    cache = MemoryCache()
    newer, older = reading("A", 10), reading("A", 2)
    w, _ = worker(FakeSource([payloads([newer]), payloads([older])], []), cache=cache)
    await w.process_once()
    await w.process_once()
    assert (await cache.get_latest("A")).ts == newer.ts


async def test_run_loop_survives_a_failing_batch_and_keeps_consuming():
    log: list[str] = []
    repo = MemoryRepo()
    calls = {"n": 0}
    real = repo.upsert_readings

    async def flaky(rows):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("blip")
        return await real(rows)

    repo.upsert_readings = flaky
    batch = payloads(history("A", 2))
    w, _ = worker(FakeSource([batch, batch], log), repo)
    task = asyncio.create_task(w.run(backoff_s=0.01))
    for _ in range(100):
        if log.count("ack") == 2:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert log == ["nak", "nak", "ack", "ack"] or log[:2] == ["nak", "nak"]
    assert await repo.latest("A") is not None
