"use client";

import { AlertsTable } from "@/components/AlertsTable";
import { DriftPanel } from "@/components/DriftPanel";
import { IncidentsPanel } from "@/components/IncidentsPanel";
import { ZoneCard } from "@/components/ZoneCard";
import { useColdChain, type Connection } from "@/lib/useColdChain";

const CONN: Record<Connection, string> = {
  connecting: "Connecting…",
  live: "Live",
  reconnecting: "Reconnecting…",
  offline: "Offline",
};

export default function Page() {
  const { state, connection, error, drift, driftError } = useColdChain();
  const zones = Object.values(state.zones).sort((a, b) => a.zone.localeCompare(b.zone));
  const active = state.alerts.filter((a) => a.closed_at === null).length;

  return (
    <main>
      <header className="top">
        <div>
          <h1>ColdChain control tower</h1>
          <p className="muted">
            Simulated automated cold store. <strong>All data is synthetic.</strong>
          </p>
        </div>
        <div className="status">
          <span className={`dot dot-${connection}`} aria-hidden="true" />
          <span role="status">{CONN[connection]}</span>
          <span className="muted"> · {active} active {active === 1 ? "alert" : "alerts"}</span>
        </div>
      </header>

      {error && (
        <p className="banner" role="alert">
          Could not load data from core ({error}). Is the stack running?
        </p>
      )}

      {zones.length === 0 && !error ? (
        <p className="muted">Waiting for the first readings…</p>
      ) : (
        <div className="zones">
          {zones.map((z) => (
            <ZoneCard key={z.zone} zone={z} alerts={state.alerts} />
          ))}
        </div>
      )}

      <div className="lower">
        <AlertsTable alerts={state.alerts} />
        <DriftPanel report={drift} error={driftError} />
      </div>
      <div className="lower">
        <IncidentsPanel incidents={state.incidents} zones={zones.map((z) => z.zone)} />
      </div>
    </main>
  );
}
