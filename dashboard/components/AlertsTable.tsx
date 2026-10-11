import type { Alert } from "@/lib/types";

const fmt = (iso: string) =>
  new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

export function AlertsTable({ alerts }: { alerts: Alert[] }) {
  return (
    <section className="card" aria-label="Alerts">
      <h2>Alerts</h2>
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
                  <td>{a.closed_at ? `closed ${fmt(a.closed_at)}` : "active"}</td>
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
