import type { DriftReport } from "@/lib/types";

const STATE_PILL: Record<string, string> = {
  ok: "ok",
  warn: "warning",
  drift: "critical",
};
const STATE_TEXT: Record<string, string> = {
  ok: "stable",
  warn: "moderate shift",
  drift: "drifting",
  insufficient_data: "not enough data",
  building_baseline: "building baseline",
};

function Pill({ state }: { state: string }) {
  return <span className={`pill pill-${STATE_PILL[state] ?? "no-data"}`}>{STATE_TEXT[state] ?? state}</span>;
}

export function DriftPanel({ report, error }: { report: DriftReport | null; error: string | null }) {
  return (
    <section className="card" aria-label="Model drift">
      <h2>Model drift</h2>
      {error && !report && <p className="muted">Drift monitor unreachable ({error}).</p>}
      {report && report.checked_at === null && <p className="muted">The monitor has not completed a check yet.</p>}
      {report && report.checked_at !== null && (
        <>
          <p className="muted">
            PSI against the training reference (inputs) and the early-deployment baseline (scores). Under{" "}
            {report.thresholds?.warn ?? 0.1} stable, {report.thresholds?.drift ?? 0.25} and above is drift.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>What</th>
                  <th>Zone</th>
                  <th>PSI</th>
                  <th>State</th>
                </tr>
              </thead>
              <tbody>
                {report.predictions.map((p) => (
                  <tr key={p.model}>
                    <td>{p.model} scores</td>
                    <td>all</td>
                    <td>{p.psi === null ? "—" : p.psi.toFixed(3)}</td>
                    <td>
                      <Pill state={p.state} />
                    </td>
                  </tr>
                ))}
                {report.inputs.map((r) => (
                  <tr key={`${r.zone}-${r.feature}`}>
                    <td>{r.feature}</td>
                    <td>{r.zone}</td>
                    <td>{r.psi === null ? "—" : r.psi.toFixed(3)}</td>
                    <td>
                      <Pill state={r.state} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {report.errors.length > 0 && <p className="muted">Last check had errors: {report.errors.join("; ")}</p>}
        </>
      )}
    </section>
  );
}
