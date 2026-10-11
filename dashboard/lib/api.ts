import type { Alert, DriftReport, Reading, ZoneSummary } from "./types";

const CORE = "/api/core";
const MONITOR = "/api/monitor";

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return (await r.json()) as T;
}

export const fetchZones = () => getJson<ZoneSummary[]>(`${CORE}/v1/zones`);

export const fetchReadings = async (zone: string, minutes: number): Promise<Reading[]> =>
  (await getJson<{ readings: Reading[] }>(`${CORE}/v1/zones/${encodeURIComponent(zone)}/readings?minutes=${minutes}`))
    .readings;

export const fetchAlerts = () => getJson<Alert[]>(`${CORE}/v1/alerts?status=all&limit=100`);

export const fetchDrift = () => getJson<DriftReport>(`${MONITOR}/v1/drift`);

export const STREAM_URL = `${CORE}/v1/stream`;
