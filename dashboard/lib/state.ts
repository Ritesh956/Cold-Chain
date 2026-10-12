import type { Alert, Incident, Reading, StreamEvent } from "./types";

export const MAX_POINTS = 720; // readings kept per zone (12 h at 1/min)
export const MAX_ALERTS = 100;
export const MAX_INCIDENTS = 50;

export type ZoneState = {
  zone: string;
  series: Reading[]; // ascending by ts
  latest: Reading | null;
};

export type State = {
  zones: Record<string, ZoneState>;
  alerts: Alert[]; // newest opened first
  incidents: Incident[]; // newest received first
};

export const empty: State = { zones: {}, alerts: [], incidents: [] };

function addReading(z: ZoneState | undefined, r: Reading): ZoneState {
  const cur = z ?? { zone: r.zone, series: [], latest: null };
  const last = cur.series[cur.series.length - 1];
  // at-least-once delivery and replays: only strictly newer readings extend the series
  if (last && Date.parse(r.ts) <= Date.parse(last.ts)) return cur;
  const series = [...cur.series, r];
  if (series.length > MAX_POINTS) series.splice(0, series.length - MAX_POINTS);
  return { zone: cur.zone, series, latest: r };
}

function sameAlert(a: Alert, b: Alert): boolean {
  if (a.id !== null && b.id !== null) return a.id === b.id;
  return a.zone === b.zone && a.kind === b.kind && a.opened_at === b.opened_at;
}

export function upsertAlert(alerts: Alert[], incoming: Alert): Alert[] {
  const i = alerts.findIndex((a) => sameAlert(a, incoming));
  const next = i >= 0 ? alerts.map((a, j) => (j === i ? incoming : a)) : [incoming, ...alerts];
  next.sort((a, b) => Date.parse(b.opened_at) - Date.parse(a.opened_at));
  return next.slice(0, MAX_ALERTS);
}

export function upsertIncident(incidents: Incident[], incoming: Incident): Incident[] {
  const i = incoming.id === null ? -1 : incidents.findIndex((x) => x.id === incoming.id);
  const next = i >= 0 ? incidents.map((x, j) => (j === i ? incoming : x)) : [incoming, ...incidents];
  next.sort((a, b) => Date.parse(b.received_at) - Date.parse(a.received_at) || (b.id ?? 0) - (a.id ?? 0));
  return next.slice(0, MAX_INCIDENTS);
}

/** Apply a batch of stream events. Pure, so it can be tested without a browser. */
export function applyEvents(state: State, events: StreamEvent[]): State {
  let zones = state.zones;
  let alerts = state.alerts;
  let incidents = state.incidents;
  for (const e of events) {
    if (e.type === "reading") {
      const prev = zones[e.reading.zone];
      const next = addReading(prev, e.reading);
      if (next !== prev) zones = { ...zones, [e.reading.zone]: next };
    } else if (e.type === "alert") {
      alerts = upsertAlert(alerts, e.alert);
    } else if (e.type === "incident") {
      incidents = upsertIncident(incidents, e.incident);
    }
  }
  return zones === state.zones && alerts === state.alerts && incidents === state.incidents
    ? state
    : { zones, alerts, incidents };
}

/** Seed from REST: history per zone and the alert list. Newer live data already in `state` wins. */
export function seed(
  state: State,
  history: Record<string, Reading[]>,
  alerts: Alert[],
  incidents: Incident[] = [],
): State {
  let next = state;
  for (const [zone, rows] of Object.entries(history)) {
    const base: State = { ...next, zones: { ...next.zones, [zone]: { zone, series: [], latest: null } } };
    const live = next.zones[zone]?.series ?? [];
    next = applyEvents(
      base,
      [...rows, ...live].map((reading) => ({ type: "reading" as const, reading })),
    );
  }
  for (const a of alerts) next = { ...next, alerts: upsertAlert(next.alerts, a) };
  for (const i of incidents) next = { ...next, incidents: upsertIncident(next.incidents, i) };
  return next;
}

export type Health = "ok" | "warning" | "critical" | "no-data";

export function zoneHealth(z: ZoneState | undefined, alerts: Alert[]): Health {
  if (!z?.latest) return "no-data";
  const open = alerts.filter((a) => a.zone === z.zone && a.closed_at === null);
  if (open.some((a) => a.severity === "critical")) return "critical";
  return open.length ? "warning" : "ok";
}

/** Thin a series to at most `max` points for drawing, keeping the first and last. */
export function downsample<T>(rows: T[], max: number): T[] {
  if (rows.length <= max) return rows;
  const step = (rows.length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => rows[Math.round(i * step)]);
}
