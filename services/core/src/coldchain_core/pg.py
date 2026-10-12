"""Postgres (optionally TimescaleDB) implementation of the Repo contract."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from coldchain_schemas import Alert, AlertKind, Incident, SensorReading
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    zone text NOT NULL,
    ts timestamptz NOT NULL,
    temp_c double precision,
    compressor_on boolean NOT NULL,
    compressor_current_a double precision NOT NULL,
    PRIMARY KEY (zone, ts)
);
-- hypertable when TimescaleDB is present; a plain table otherwise (tests, plain Postgres)
DO $$
BEGIN
    PERFORM create_hypertable('readings', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'timescaledb not available; readings is a plain table';
END $$;

CREATE TABLE IF NOT EXISTS zones (zone text PRIMARY KEY);

CREATE TABLE IF NOT EXISTS alerts (
    id bigserial PRIMARY KEY,
    zone text NOT NULL,
    kind text NOT NULL,
    severity text NOT NULL,
    opened_at timestamptz NOT NULL,
    closed_at timestamptz,
    peak_score double precision,
    summary text,
    model_version text
);
CREATE INDEX IF NOT EXISTS alerts_open_idx ON alerts (zone, kind) WHERE closed_at IS NULL;
CREATE INDEX IF NOT EXISTS alerts_opened_idx ON alerts (opened_at DESC);

CREATE TABLE IF NOT EXISTS incidents (
    id bigserial PRIMARY KEY,
    received_at timestamptz NOT NULL,
    zone text,
    text text NOT NULL,
    label text NOT NULL,
    confidence double precision NOT NULL,
    needs_review boolean NOT NULL,
    model_version text
);
CREATE INDEX IF NOT EXISTS incidents_received_idx ON incidents (received_at DESC);

CREATE TABLE IF NOT EXISTS scoring_state (
    zone text NOT NULL,
    model text NOT NULL,
    last_ts timestamptz NOT NULL,
    PRIMARY KEY (zone, model)
);
"""

_READING_COLS = "zone, ts, temp_c, compressor_on, compressor_current_a"
_INCIDENT_COLS = "id, received_at, zone, text, label, confidence, needs_review, model_version"
_ALERT_COLS = "id, zone, kind, severity, opened_at, closed_at, peak_score, summary, model_version"


def _alert(row: dict) -> Alert:
    return Alert(**{**row, "kind": AlertKind(row["kind"])})


class PgRepo:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 6) -> None:
        self._pool = AsyncConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row, "options": "-c timezone=UTC"},
        )

    async def init(self) -> None:
        await self._pool.open(wait=True, timeout=30)
        async with self._pool.connection() as conn:
            await conn.execute(SCHEMA)

    async def close(self) -> None:
        await self._pool.close()

    async def ping(self) -> bool:
        try:
            async with self._pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def upsert_readings(self, rows: Sequence[SensorReading]) -> int:
        if not rows:
            return 0
        params = [(r.zone, r.ts, r.temp_c, r.compressor_on, r.compressor_current_a) for r in rows]
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                f"INSERT INTO readings ({_READING_COLS}) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (zone, ts) DO NOTHING",
                params,
                returning=False,
            )
            inserted = cur.rowcount
            await cur.executemany(
                "INSERT INTO zones (zone) VALUES (%s) ON CONFLICT DO NOTHING",
                [(z,) for z in {r.zone for r in rows}],
            )
        return max(inserted, 0)

    async def zones(self) -> list[str]:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT zone FROM zones ORDER BY zone")
            return [r["zone"] for r in await cur.fetchall()]

    async def latest(self, zone: str) -> SensorReading | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_READING_COLS} FROM readings WHERE zone = %s ORDER BY ts DESC LIMIT 1", (zone,)
            )
            row = await cur.fetchone()
        return SensorReading(**row) if row else None

    async def window(self, zone: str, end: datetime, minutes: int) -> list[SensorReading]:
        start = end - timedelta(minutes=minutes - 1)
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"(SELECT {_READING_COLS} FROM readings WHERE zone = %(z)s AND ts < %(s)s "
                " ORDER BY ts DESC LIMIT 1) "
                f"UNION ALL (SELECT {_READING_COLS} FROM readings WHERE zone = %(z)s "
                " AND ts >= %(s)s AND ts <= %(e)s) ORDER BY ts",
                {"z": zone, "s": start, "e": end},
            )
            return [SensorReading(**r) for r in await cur.fetchall()]

    async def get_scoring_state(self, zone: str, model: str) -> datetime | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT last_ts FROM scoring_state WHERE zone = %s AND model = %s", (zone, model)
            )
            row = await cur.fetchone()
        return row["last_ts"] if row else None

    async def set_scoring_state(self, zone: str, model: str, ts: datetime) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO scoring_state (zone, model, last_ts) VALUES (%s, %s, %s) "
                "ON CONFLICT (zone, model) DO UPDATE SET last_ts = EXCLUDED.last_ts",
                (zone, model, ts),
            )

    async def save_alert(self, alert: Alert) -> Alert:
        async with self._pool.connection() as conn:
            if alert.id is None:
                cur = await conn.execute(
                    "INSERT INTO alerts (zone, kind, severity, opened_at, closed_at, peak_score, summary, "
                    "model_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (
                        alert.zone,
                        alert.kind.value,
                        alert.severity,
                        alert.opened_at,
                        alert.closed_at,
                        alert.peak_score,
                        alert.summary,
                        alert.model_version,
                    ),
                )
                row = await cur.fetchone()
                return alert.model_copy(update={"id": row["id"]})
            await conn.execute(
                "UPDATE alerts SET closed_at = %s, peak_score = %s, summary = %s WHERE id = %s",
                (alert.closed_at, alert.peak_score, alert.summary, alert.id),
            )
            return alert

    async def open_alerts(self) -> list[Alert]:
        return await self.list_alerts(active_only=True, zone=None, limit=10_000)

    async def list_alerts(self, active_only: bool, zone: str | None, limit: int) -> list[Alert]:
        where, params = [], []
        if active_only:
            where.append("closed_at IS NULL")
        if zone is not None:
            where.append("zone = %s")
            params.append(zone)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_ALERT_COLS} FROM alerts {clause} ORDER BY opened_at DESC LIMIT %s",
                (*params, limit),
            )
            return [_alert(r) for r in await cur.fetchall()]

    async def save_incident(self, incident: Incident) -> Incident:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO incidents (received_at, zone, text, label, confidence, needs_review, "
                "model_version) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    incident.received_at,
                    incident.zone,
                    incident.text,
                    incident.label,
                    incident.confidence,
                    incident.needs_review,
                    incident.model_version,
                ),
            )
            row = await cur.fetchone()
        return incident.model_copy(update={"id": row["id"]})

    async def list_incidents(
        self, label: str | None, needs_review: bool | None, limit: int
    ) -> list[Incident]:
        where, params = [], []
        if label is not None:
            where.append("label = %s")
            params.append(label)
        if needs_review is not None:
            where.append("needs_review = %s")
            params.append(needs_review)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_INCIDENT_COLS} FROM incidents {clause} "
                "ORDER BY received_at DESC, id DESC LIMIT %s",
                (*params, limit),
            )
            return [Incident(**r) for r in await cur.fetchall()]
