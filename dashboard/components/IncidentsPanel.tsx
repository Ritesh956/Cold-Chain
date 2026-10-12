"use client";

import { useState } from "react";
import { submitIncident } from "@/lib/api";
import type { Incident } from "@/lib/types";

const fmt = (iso: string) =>
  new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

const LABEL_PILL: Record<Incident["label"], string> = {
  safety: "critical",
  temperature: "warning",
  damage: "warning",
  delay: "no-data",
};

export function IncidentsPanel({ incidents, zones }: { incidents: Incident[]; zones: string[] }) {
  const [text, setText] = useState("");
  const [zone, setZone] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      await submitIncident(text.trim(), zone || null); // the new row arrives over the live stream
      setText("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card" aria-label="Incident reports">
      <h2>Incident reports</h2>
      <form onSubmit={onSubmit} className="incident-form">
        <label htmlFor="incident-text" className="muted">
          Describe an incident; the classifier labels it (trained on synthetic text only).
        </label>
        <textarea
          id="incident-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          maxLength={2000}
          rows={2}
          placeholder="e.g. Forklift clipped a pallet at dock 3, wrapping torn and two cases crushed"
        />
        <div className="form-row">
          <label htmlFor="incident-zone" className="muted">
            Zone
          </label>
          <select id="incident-zone" value={zone} onChange={(e) => setZone(e.target.value)}>
            <option value="">none</option>
            {zones.map((z) => (
              <option key={z} value={z}>
                {z}
              </option>
            ))}
          </select>
          <button type="submit" disabled={busy || !text.trim()}>
            {busy ? "Classifying…" : "Classify and save"}
          </button>
        </div>
        {error && (
          <p className="banner" role="alert">
            {error}
          </p>
        )}
      </form>
      {incidents.length === 0 ? (
        <p className="muted">No reports yet.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Received</th>
                <th>Zone</th>
                <th>Label</th>
                <th>Score</th>
                <th>Report</th>
              </tr>
            </thead>
            <tbody>
              {incidents.map((i, n) => (
                <tr key={i.id ?? n}>
                  <td>{fmt(i.received_at)}</td>
                  <td>{i.zone ?? "—"}</td>
                  <td>
                    <span className={`pill pill-${LABEL_PILL[i.label]}`}>{i.label}</span>
                    {i.needs_review && <span className="pill pill-warning review"> needs review</span>}
                  </td>
                  <td title="Top softmax score. A ranking signal, not a calibrated probability.">
                    {i.confidence.toFixed(2)}
                  </td>
                  <td>{i.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
