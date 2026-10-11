"""How well do the input-drift checks separate "normal days" from injected drift?  (SYNTHETIC data.)

Every zone-day (24 h window, as the monitor uses) of the synthetic history is scored against the
shipped reference, unchanged and then with a drift injected. Writes docs/results/drift_demo.{json,md}.

    uv run python scripts/drift_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from coldchain_monitor import reference
from coldchain_monitor.drift import psi, status

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "results"

SCENARIOS = {
    "none (normal operation)": lambda d: d,
    "temperature sensor +0.5 C": lambda d: d.assign(temp_c=d.temp_c + 0.5),
    "temperature sensor +1.0 C": lambda d: d.assign(temp_c=d.temp_c + 1.0),
    "temperature sensor +2.0 C": lambda d: d.assign(temp_c=d.temp_c + 2.0),
    "compressor current +10 %": lambda d: d.assign(compressor_current_a=d.compressor_current_a * 1.10),
    "compressor current +25 %": lambda d: d.assign(compressor_current_a=d.compressor_current_a * 1.25),
    "compressor stuck on": lambda d: d.assign(compressor_on=True),
}


def main() -> None:
    profiles = reference.load()
    df = pd.read_parquet(ROOT / "data/synthetic/freezer/readings.parquet")
    events = pd.read_parquet(ROOT / "data/synthetic/freezer/events.parquet")
    anomalies = events[events["is_anomaly"]]
    df["day"] = df["ts"].dt.floor("D")
    clean, with_anomaly = [], []
    for (zone, day), g in df.groupby(["zone", "day"]):
        if len(g) < 1440:
            continue
        a = anomalies[(anomalies["zone"] == zone) & (anomalies["start"] < day + pd.Timedelta(days=1))]
        a = a[a["end"] > day]
        (with_anomaly if len(a) else clean).append(g)

    groups = {"none, days with an injected fault": with_anomaly}
    groups |= {name: clean for name in SCENARIOS}
    rows = []
    for name, days in groups.items():
        inject = SCENARIOS.get(name, SCENARIOS["none (normal operation)"])
        counts = {"ok": 0, "warn": 0, "drift": 0}
        by_feature: dict[str, int] = {}
        for g in days:
            g = inject(g.copy())
            flagged = False
            for feature, profile in profiles.items():
                vals = g[feature].dropna().astype(float).to_numpy()
                st = status(psi(profile, vals))
                if st == "drift":
                    by_feature[feature] = by_feature.get(feature, 0) + 1
                    flagged = True
            counts["drift" if flagged else "ok"] += 1
        rows.append(
            {
                "scenario": name,
                "zone_days": len(days),
                "days_flagged_drift": counts["drift"],
                "share_flagged": counts["drift"] / len(days),
                "flagged_by_feature": by_feature,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drift_demo.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Input-drift check on SYNTHETIC data",
        "",
        "Each row scores zone-days (24 h, 1,440 readings) of the synthetic history against the shipped",
        "reference with PSI, flagging 'drift' at PSI >= 0.25 on any feature. Days are split by whether the",
        "simulator injected an anomaly (door left open, compressor drift, power cut) that day. The reference",
        "was built from this same history, so the 'none' rows are the false-alarm rate on data the reference",
        "has seen, not on fresh data. Injected drift is a synthetic edit of the anomaly-free days.",
        "",
        "| Injected drift | Zone-days | Flagged | Share | Flagged by |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        feats = ", ".join(f"{k} {v}" for k, v in r["flagged_by_feature"].items()) or "none"
        cells = [r["scenario"], r["zone_days"], r["days_flagged_drift"], f"{r['share_flagged']:.1%}", feats]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    (OUT / "drift_demo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
