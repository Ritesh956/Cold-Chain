import { describe, expect, it } from "vitest";
import { applyEvents, downsample, empty, MAX_INCIDENTS, MAX_POINTS, seed, upsertAlert, upsertIncident, zoneHealth } from "../lib/state";
import type { Alert, Incident, Reading, StreamEvent } from "../lib/types";

const T0 = Date.parse("2025-01-01T00:00:00Z");
const reading = (i: number, zone = "A", temp: number | null = -25): Reading => ({
  ts: new Date(T0 + i * 60_000).toISOString(),
  zone,
  temp_c: temp,
  compressor_on: true,
  compressor_current_a: 12,
});
const ev = (r: Reading): StreamEvent => ({ type: "reading", reading: r });
const alert = (over: Partial<Alert> = {}): Alert => ({
  id: 1,
  zone: "A",
  kind: "anomaly",
  severity: "warning",
  opened_at: new Date(T0).toISOString(),
  closed_at: null,
  peak_score: null,
  summary: null,
  model_version: null,
  ...over,
});

describe("applyEvents", () => {
  it("appends readings and tracks the latest", () => {
    const s = applyEvents(empty, [ev(reading(0)), ev(reading(1, "A", -24))]);
    expect(s.zones.A.series).toHaveLength(2);
    expect(s.zones.A.latest?.temp_c).toBe(-24);
  });

  it("drops duplicates and out-of-order readings", () => {
    const s = applyEvents(empty, [ev(reading(5)), ev(reading(5)), ev(reading(3))]);
    expect(s.zones.A.series).toHaveLength(1);
  });

  it("keeps zones separate", () => {
    const s = applyEvents(empty, [ev(reading(0, "A")), ev(reading(0, "B"))]);
    expect(Object.keys(s.zones).sort()).toEqual(["A", "B"]);
  });

  it("caps the series length", () => {
    const events = Array.from({ length: MAX_POINTS + 50 }, (_, i) => ev(reading(i)));
    const s = applyEvents(empty, events);
    expect(s.zones.A.series).toHaveLength(MAX_POINTS);
    expect(s.zones.A.latest?.ts).toBe(reading(MAX_POINTS + 49).ts);
  });

  it("returns the same state object when nothing changed", () => {
    const s = applyEvents(empty, [ev(reading(0))]);
    expect(applyEvents(s, [ev(reading(0))])).toBe(s);
  });

  it("keeps a sensor dropout as a null reading", () => {
    const s = applyEvents(empty, [ev(reading(0, "A", null))]);
    expect(s.zones.A.latest?.temp_c).toBeNull();
  });
});

describe("alerts", () => {
  it("updates an alert in place when it closes", () => {
    let alerts = upsertAlert([], alert());
    alerts = upsertAlert(alerts, alert({ closed_at: new Date(T0 + 600_000).toISOString() }));
    expect(alerts).toHaveLength(1);
    expect(alerts[0].closed_at).not.toBeNull();
  });

  it("orders newest first and caps the list", () => {
    let alerts: Alert[] = [];
    for (let i = 0; i < 120; i++) alerts = upsertAlert(alerts, alert({ id: i, opened_at: new Date(T0 + i * 1000).toISOString() }));
    expect(alerts).toHaveLength(100);
    expect(alerts[0].id).toBe(119);
  });

  it("flows through applyEvents", () => {
    const s = applyEvents(empty, [{ type: "alert", action: "opened", alert: alert() }]);
    expect(s.alerts).toHaveLength(1);
  });
});

describe("seed", () => {
  it("merges history with live readings already received", () => {
    const live = applyEvents(empty, [ev(reading(10)), ev(reading(11))]);
    const s = seed(live, { A: [reading(8), reading(9), reading(10)] }, [alert()]);
    expect(s.zones.A.series.map((r) => r.ts)).toEqual([8, 9, 10, 11].map((i) => reading(i).ts));
    expect(s.alerts).toHaveLength(1);
  });
});

describe("zoneHealth", () => {
  const zone = applyEvents(empty, [ev(reading(0))]).zones.A;
  it("is ok without open alerts", () => expect(zoneHealth(zone, [alert({ closed_at: "2025-01-01T01:00:00Z" })])).toBe("ok"));
  it("is warning with an open warning", () => expect(zoneHealth(zone, [alert()])).toBe("warning"));
  it("is critical with an open critical alert", () =>
    expect(zoneHealth(zone, [alert(), alert({ id: 2, severity: "critical", kind: "breached" })])).toBe("critical"));
  it("is no-data before the first reading", () => expect(zoneHealth(undefined, [])).toBe("no-data"));
  it("ignores other zones' alerts", () => expect(zoneHealth(zone, [alert({ zone: "B" })])).toBe("ok"));
});

describe("downsample", () => {
  it("keeps first and last and respects the cap", () => {
    const rows = Array.from({ length: 1000 }, (_, i) => i);
    const out = downsample(rows, 100);
    expect(out).toHaveLength(100);
    expect(out[0]).toBe(0);
    expect(out[99]).toBe(999);
  });
  it("leaves short series alone", () => expect(downsample([1, 2, 3], 10)).toEqual([1, 2, 3]));
});

const incident = (id: number, over: Partial<Incident> = {}): Incident => ({
  id,
  received_at: new Date(T0 + id * 1000).toISOString(),
  zone: "A",
  text: `report ${id}`,
  label: "damage",
  confidence: 0.9,
  needs_review: false,
  model_version: "1",
  ...over,
});

describe("incidents", () => {
  it("arrive over the stream, newest first, without duplicates", () => {
    const e = (i: Incident): StreamEvent => ({ type: "incident", incident: i });
    const s = applyEvents(empty, [e(incident(1)), e(incident(2)), e(incident(2))]);
    expect(s.incidents.map((i) => i.id)).toEqual([2, 1]);
  });

  it("are capped", () => {
    let list: Incident[] = [];
    for (let i = 0; i < MAX_INCIDENTS + 10; i++) list = upsertIncident(list, incident(i));
    expect(list).toHaveLength(MAX_INCIDENTS);
    expect(list[0].id).toBe(MAX_INCIDENTS + 9);
  });

  it("are seeded from REST without losing a live one", () => {
    const live = applyEvents(empty, [{ type: "incident", incident: incident(5) }]);
    const s = seed(live, {}, [], [incident(3), incident(4), incident(5)]);
    expect(s.incidents.map((i) => i.id)).toEqual([5, 4, 3]);
  });
});
