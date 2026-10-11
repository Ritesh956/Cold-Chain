import { zoneHealth, type ZoneState } from "@/lib/state";
import type { Alert } from "@/lib/types";
import { TempChart } from "./TempChart";

const LABEL = { ok: "Normal", warning: "Warning", critical: "Critical", "no-data": "No data" } as const;

export function ZoneCard({ zone, alerts }: { zone: ZoneState; alerts: Alert[] }) {
  const mine = alerts.filter((a) => a.zone === zone.zone);
  const open = mine.filter((a) => a.closed_at === null);
  const health = zoneHealth(zone, alerts);
  const r = zone.latest;
  return (
    <section className={`card zone health-${health}`} aria-label={`Zone ${zone.zone}`}>
      <header>
        <h2>Zone {zone.zone}</h2>
        {/* colour is never the only signal: the state is also written out */}
        <span className={`pill pill-${health}`}>{LABEL[health]}</span>
      </header>
      <div className="readout">
        <div>
          <div className="big">{r?.temp_c == null ? "—" : r.temp_c.toFixed(1)}<small> °C</small></div>
          <div className="muted">{r?.temp_c == null && r ? "sensor dropout" : "temperature"}</div>
        </div>
        <div>
          <div className="mid">{r ? (r.compressor_on ? "On" : "Off") : "—"}</div>
          <div className="muted">compressor{r ? `, ${r.compressor_current_a.toFixed(1)} A` : ""}</div>
        </div>
      </div>
      <TempChart series={zone.series} alerts={mine} />
      {open.length > 0 && (
        <ul className="open-alerts">
          {open.map((a, i) => (
            <li key={a.id ?? i}>
              <strong>{a.kind.replace("_", " ")}</strong>
              {a.summary ? `: ${a.summary}` : ""}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
