import type { Alert, DriftReport, Incident, Reading, ZoneSummary } from "./types";

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

export async function acknowledgeAlert(id: number): Promise<Alert> {
  const r = await fetch(`${CORE}/v1/alerts/${id}/ack`, { method: "POST" });
  if (!r.ok) throw new Error(r.status === 404 ? "alert not found" : `request failed (${r.status})`);
  return (await r.json()) as Alert;
}

export const fetchIncidents = () => getJson<Incident[]>(`${CORE}/v1/incidents?limit=50`);

export async function submitIncident(text: string, zone: string | null): Promise<Incident> {
  const r = await fetch(`${CORE}/v1/incidents`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text, zone }),
  });
  if (!r.ok) {
    const detail = await r.json().then((j) => j.detail, () => null);
    throw new Error(typeof detail === "string" ? detail : `request failed (${r.status})`);
  }
  return (await r.json()) as Incident;
}

export const fetchDrift = () => getJson<DriftReport>(`${MONITOR}/v1/drift`);

export const STREAM_URL = `${CORE}/v1/stream`;
