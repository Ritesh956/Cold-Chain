from __future__ import annotations

from dataclasses import replace

from coldchain_core.alerts import AlertEngine
from coldchain_core.config import Settings
from coldchain_core.events import BroadcastAlertPublisher, Broadcaster
from coldchain_core.metrics import Metrics
from coldchain_core.mlclient import MlApiError
from coldchain_core.repo import MemoryRepo
from coldchain_core.scoring import Scorer
from coldchain_schemas import AlertKind
from core_helpers import MIN, T0, FakeMl, history, reading

SETTINGS = Settings(score_every_min=1, max_chunk_min=30, min_coverage=0.95, alert_close_after_min=3)


def val(counter, *labels) -> float:
    return counter.labels(*labels)._value.get()


async def make(n_minutes=200, ml=None, settings=SETTINGS, repo=None):
    repo = repo or MemoryRepo()
    await repo.upsert_readings(history("A", n_minutes))
    ml = ml or FakeMl()
    bcast, metrics = Broadcaster(), Metrics()
    scorer = Scorer(
        repo,
        ml,
        AlertEngine(settings.alert_close_after_min),
        BroadcastAlertPublisher(bcast),
        metrics,
        settings,
    )
    await scorer.start()
    return scorer, repo, ml, bcast, metrics


async def test_first_sight_scores_only_the_newest_minute_with_required_history():
    scorer, repo, ml, _, _ = await make()
    await scorer.tick()
    by_kind = {k: (n, rl) for k, _, n, rl in ml.calls}
    assert by_kind == {"anomaly": (10, 1), "breach": (20, 1)}  # min_history + return_last - 1, no backfill
    assert await repo.get_scoring_state("A", "anomaly") == T0 + 199 * MIN


async def test_catches_up_in_chunks_and_never_skips_a_minute():
    scorer, repo, ml, _, _ = await make()
    await scorer.tick()
    await repo.upsert_readings(history("A", 100, start=200))  # 100 new minutes arrive
    ml.calls.clear()
    await scorer.tick()  # max_chunk_min=30 -> 30 minutes this tick
    assert [(k, rl) for k, _, _, rl in ml.calls] == [("anomaly", 30), ("breach", 30)]
    assert await repo.get_scoring_state("A", "anomaly") == T0 + 229 * MIN
    for _ in range(3):
        await scorer.tick()
    assert await repo.get_scoring_state("A", "anomaly") == T0 + 299 * MIN  # fully caught up
    assert sum(rl for k, _, _, rl in ml.calls if k == "anomaly") == 100  # exactly the 100 new minutes


async def test_nothing_new_means_no_ml_call():
    scorer, _, ml, _, _ = await make()
    await scorer.tick()
    ml.calls.clear()
    await scorer.tick()
    assert ml.calls == []


async def test_ml_outage_delays_but_never_skips_then_recovers():
    scorer, repo, ml, _, m = await make()
    await scorer.tick()
    await repo.upsert_readings(history("A", 5, start=200))
    ml.fail = MlApiError("down", retryable=True)
    await scorer.tick()
    assert await repo.get_scoring_state("A", "anomaly") == T0 + 199 * MIN  # position unchanged
    assert val(m.scored, "anomaly", "ml_error_retry") == 1
    ml.fail = None
    await scorer.tick()
    assert await repo.get_scoring_state("A", "anomaly") == T0 + 204 * MIN
    assert [rl for k, _, _, rl in ml.calls if k == "anomaly"][-1] == 5  # all 5 missed minutes scored


async def test_rejected_request_is_counted_distinctly():
    scorer, repo, ml, _, m = await make()
    await scorer.tick()
    await repo.upsert_readings(history("A", 2, start=200))
    ml.fail = MlApiError("422", retryable=False)
    await scorer.tick()
    assert val(m.scored, "anomaly", "ml_error_rejected") == 1


async def test_insufficient_history_is_skipped_not_retried_forever():
    scorer, repo, ml, _, m = await make(n_minutes=5)  # anomaly needs 10, breach needs 20
    await scorer.tick()
    assert ml.calls == []
    assert val(m.scored, "anomaly", "skipped_insufficient_data") == 1
    assert await repo.get_scoring_state("A", "anomaly") == T0 + 4 * MIN  # moved on
    await repo.upsert_readings(history("A", 40, start=5))  # latest is now minute 44
    await scorer.tick()  # a 30-minute chunk needs a window reaching before minute 0: skipped again
    assert ml.calls == [] and await repo.get_scoring_state("A", "anomaly") == T0 + 34 * MIN
    await scorer.tick()  # the remaining 10 minutes have a fully covered window and are scored
    assert [(k, n, rl) for k, _, n, rl in ml.calls if k == "anomaly"] == [("anomaly", 19, 10)]


async def test_a_gap_inside_the_window_is_sent_as_null_temperatures():
    repo = MemoryRepo()
    await repo.upsert_readings([r for r in history("A", 100) if not 95 <= (r.ts - T0) / MIN < 97])
    sent: list[list[dict]] = []

    class Capture(FakeMl):
        async def anomaly(self, zone, readings, return_last):
            sent.append(readings)
            return await super().anomaly(zone, readings, return_last)

    scorer, *_ = await make(
        ml=Capture(), repo=repo, n_minutes=0, settings=replace(SETTINGS, min_coverage=0.7)
    )
    await scorer.tick()
    assert [r["temp_c"] for r in sent[0]].count(None) == 2 and len(sent[0]) == 10


async def test_alarm_opens_one_alert_and_closes_it_with_hysteresis():
    alarm = lambda ts, kind: kind == "anomaly" and T0 + 205 * MIN <= ts < T0 + 210 * MIN  # noqa: E731
    scorer, repo, ml, bcast, m = await make(ml=FakeMl(alarm_when=alarm))
    await scorer.tick()  # first sight at minute 199: quiet
    await repo.upsert_readings(history("A", 30, start=200))  # minutes 200..229
    async with bcast.subscribe() as q:
        await scorer.tick()
        events = [q.get_nowait() for _ in range(q.qsize())]
    assert [(e["action"], e["alert"]["kind"]) for e in events] == [
        ("opened", "anomaly"),
        ("closed", "anomaly"),
    ]
    stored = await repo.list_alerts(False, "A", 10)
    assert len(stored) == 1 and stored[0].id is not None
    assert stored[0].opened_at == T0 + 205 * MIN and stored[0].closed_at == T0 + 210 * MIN
    assert stored[0].peak_score == 9.0 and stored[0].model_version == "1"
    assert val(m.alerts, "anomaly", "opened") == 1 and val(m.alerts, "anomaly", "closed") == 1


async def test_breached_state_raises_a_critical_alert_and_breach_risk_a_warning():
    def alarm(ts, kind):
        if kind == "breach":
            return T0 + 205 * MIN <= ts < T0 + 207 * MIN
        return kind == "breached" and T0 + 207 * MIN <= ts < T0 + 209 * MIN

    scorer, repo, _, _, _ = await make(ml=FakeMl(alarm_when=alarm))
    await scorer.tick()
    await repo.upsert_readings(history("A", 30, start=200))
    await scorer.tick()
    sev = {a.kind: a.severity for a in await repo.list_alerts(False, "A", 10)}
    assert sev == {AlertKind.BREACH_RISK: "warning", AlertKind.BREACHED: "critical"}


async def test_restart_resumes_without_duplicating_an_open_alert():
    alarm = lambda ts, kind: kind == "anomaly" and ts >= T0 + 205 * MIN  # noqa: E731  (alarm never clears)
    scorer, repo, ml, _, _ = await make(ml=FakeMl(alarm_when=alarm))
    await scorer.tick()
    await repo.upsert_readings(history("A", 10, start=200))
    await scorer.tick()
    assert len(await repo.open_alerts()) == 1

    # a new process: fresh engine, same database
    scorer2 = Scorer(repo, ml, AlertEngine(3), BroadcastAlertPublisher(Broadcaster()), Metrics(), SETTINGS)
    await scorer2.start()
    await repo.upsert_readings(history("A", 10, start=210))
    await scorer2.tick()
    assert len(await repo.list_alerts(False, "A", 10)) == 1  # still the same single alert


async def test_zones_are_scored_independently():
    repo = MemoryRepo()
    await repo.upsert_readings(history("A", 60))
    await repo.upsert_readings(history("B", 15))  # B has too little history for breach (20)
    scorer, _, ml, _, m = await make(n_minutes=0, repo=repo)
    await scorer.tick()
    assert {(k, z) for k, z, *_ in ml.calls} == {("anomaly", "A"), ("breach", "A"), ("anomaly", "B")}
    assert val(m.scored, "breach", "skipped_insufficient_data") == 1


async def test_settings_from_env_parses_types(monkeypatch):
    monkeypatch.setenv("SCORE_EVERY_MIN", "5")
    monkeypatch.setenv("MIN_COVERAGE", "0.9")
    monkeypatch.setenv("NATS_URL", "nats://x:4222")
    s = Settings.from_env()
    assert (s.score_every_min, s.min_coverage, s.nats_url) == (5, 0.9, "nats://x:4222")
    assert replace(s, score_every_min=1).score_every_min == 1
    assert reading("A").zone == "A"
