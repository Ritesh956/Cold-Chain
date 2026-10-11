import { downsample } from "@/lib/state";
import { LIMIT_C, type Alert, type Reading } from "@/lib/types";

const W = 600;
const H = 150;
const PAD = { l: 38, r: 8, t: 8, b: 18 };

/** Temperature over time with the -18 C limit and open-alert spans. Gaps (sensor dropout) break the line. */
export function TempChart({ series, alerts }: { series: Reading[]; alerts: Alert[] }) {
  const pts = downsample(series, 300);
  if (pts.length < 2) return <div className="chart-empty">Waiting for readings…</div>;

  const t0 = Date.parse(pts[0].ts);
  const t1 = Date.parse(pts[pts.length - 1].ts);
  const temps = pts.map((p) => p.temp_c).filter((v): v is number => v !== null);
  const lo = Math.floor(Math.min(...temps, LIMIT_C - 4) - 1);
  const hi = Math.ceil(Math.max(...temps, LIMIT_C + 2) + 1);
  const x = (t: number) => PAD.l + ((t - t0) / Math.max(t1 - t0, 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + ((hi - v) / (hi - lo)) * (H - PAD.t - PAD.b);

  let d = "";
  let pen = false;
  for (const p of pts) {
    if (p.temp_c === null) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x(Date.parse(p.ts)).toFixed(1)},${y(p.temp_c).toFixed(1)}`;
    pen = true;
  }

  const spans = alerts
    .map((a) => {
      const s = Math.max(Date.parse(a.opened_at), t0);
      const e = Math.min(a.closed_at ? Date.parse(a.closed_at) : t1, t1);
      return e > s ? { a, x: x(s), w: Math.max(x(e) - x(s), 2) } : null;
    })
    .filter((s): s is { a: Alert; x: number; w: number } => s !== null);

  const ticks = [lo, Math.round((lo + hi) / 2), hi];
  const fmt = (t: number) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Temperature over the last hours" className="chart">
      {spans.map((s, i) => (
        <rect
          key={i}
          x={s.x}
          y={PAD.t}
          width={s.w}
          height={H - PAD.t - PAD.b}
          className={s.a.severity === "critical" ? "span-critical" : "span-warning"}
        >
          <title>{`${s.a.kind} (${s.a.severity})`}</title>
        </rect>
      ))}
      {ticks.map((v) => (
        <g key={v}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="grid" />
          <text x={PAD.l - 6} y={y(v) + 4} textAnchor="end" className="axis">
            {v}
          </text>
        </g>
      ))}
      <line x1={PAD.l} x2={W - PAD.r} y1={y(LIMIT_C)} y2={y(LIMIT_C)} className="limit" />
      <text x={W - PAD.r} y={y(LIMIT_C) - 4} textAnchor="end" className="axis limit-label">
        limit {LIMIT_C} °C
      </text>
      <path d={d} className="line" />
      <text x={PAD.l} y={H - 4} className="axis">
        {fmt(t0)}
      </text>
      <text x={W - PAD.r} y={H - 4} textAnchor="end" className="axis">
        {fmt(t1)}
      </text>
    </svg>
  );
}
