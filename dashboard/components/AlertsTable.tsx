"use client";

import { useState } from "react";
import { acknowledgeAlert } from "@/lib/api";
import type { Alert } from "@/lib/types";

const fmt = (iso: string) =>
  new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

export function AlertsTable({ alerts }: { alerts: Alert[] }) {
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function ack(id: number) {
    setBusy(id);
    setError(null);
    try {
      await acknowledgeAlert(id); // the updated alert arrives over the live stream
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="card" aria-label="Alerts">
      <h2>Alerts</h2>
      {error && (
        <p className="banner" role="alert">
          Could not acknowledge: {error}
        </p>
      )}
      {alerts.length === 0 ? (
        <p className="muted">No alerts yet.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Zone</th>
                <th>Kind</th>
                <th>Severity</th>
                <th>Opened</th>
                <th>Status</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((a, i) => (
                <tr key={a.id ?? `${a.zone}-${a.opened_at}-${i}`}>
                  <td>{a.zone}</td>
                  <td>{a.kind.replace("_", " ")}</td>
                  <td>
                    <span className={`pill pill-${a.severity === "critical" ? "critical" : "warning"}`}>
                      {a.severity}
                    </span>
                  </td>
                  <td>{fmt(a.opened_at)}</td>
                  <td>
                    {a.closed_at ? `closed ${fmt(a.closed_at)}` : "active"}
                    {a.acknowledged_at ? (
                      <div className="muted">acknowledged {fmt(a.acknowledged_at)}</div>
                    ) : (
                      a.closed_at === null &&
                      a.id !== null && (
                        <div>
                          <button className="link-button" disabled={busy === a.id} onClick={() => ack(a.id as number)}>
                            {busy === a.id ? "Acknowledging…" : "Acknowledge"}
                          </button>
                        </div>
                      )
                    )}
                  </td>
                  <td>{a.summary ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
