"""Real-data sanity check of the temperature-only anomaly detectors on NAB machine temperature.

Protocol, fixed BEFORE looking at any NAB result (3 scorable failures is far too few to tune on):
  * Fit on NAB's probationary first 15% (no labels used).
  * Operating point A (label-free): threshold = 99.5th percentile of the detector's own scores on
    the probationary period, no debounce.
  * Operating point B (transfer): the threshold + debounce tuned on SYNTHETIC validation data,
    for the unit-free z-score detectors only. Tests whether a synthetic operating point survives
    real data.
  * Score only after the probationary period (NAB convention). Failure 1 lies inside it, so 3 of
    the 4 labelled failures are scored.

Run:  uv run python -I -m coldchain_ml.nab.run     (after scripts/download_nab.py)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coldchain_ml.anomaly.detectors import (  # noqa: E402
    GlobalZScore,
    IsolationForestDetector,
    RobustZScore,
    RollingZScore,
)
from coldchain_ml.anomaly.events_eval import apply_min_run, evaluate_zone, segments  # noqa: E402
from coldchain_ml.common.stats import wilson_interval  # noqa: E402
from coldchain_ml.common.tracking import start_run  # noqa: E402
from coldchain_ml.nab.data import STEP_MIN, NabSeries, load_nab  # noqa: E402
from coldchain_ml.nab.features import UNIVARIATE_FEATURES, build_univariate_features  # noqa: E402

PROBATION_QUANTILE = 0.995
N_SHIFTS = 2000
SYNTHETIC_RESULTS = Path("docs/results/anomaly_synthetic.json")
SYNTHETIC_NAMES = {  # detector -> its name in the synthetic results file
    "z-score": "z-score (baseline)",
    "rolling z-score": "rolling z-score (baseline)",
    "robust z-score (temp)": "robust z-score (temp)",
}
OUT = Path("docs/results")


def _detectors():
    return [
        ("z-score", GlobalZScore()),
        ("rolling z-score", RollingZScore()),
        ("robust z-score (temp)", RobustZScore(("temp",))),
        ("isolation forest", IsolationForestDetector(0, UNIVARIATE_FEATURES, stride=1)),
    ]


def _chance_baseline(
    test_ts: pd.DatetimeIndex, alarm: np.ndarray, events: pd.DataFrame, observed: int, seed: int = 0
) -> dict:
    """Null hypothesis 'alarms are unrelated to the failures': circularly shift the alarm series
    by a random offset (keeps its alarm count, durations and clustering) and recount detections.
    Wide labelled windows (~47 h) let a noisy detector 'detect' failures by luck; this measures how often."""
    rng = np.random.default_rng(seed)
    hits = np.array(
        [
            evaluate_zone(
                test_ts, np.roll(alarm, int(rng.integers(len(alarm)))), events, tol_min=30, merge_gap=5,
                attribute_false=False, step_min=STEP_MIN,
            ).n_detected
            for _ in range(N_SHIFTS)
        ]
    )
    return {
        "chance_mean_detected": float(hits.mean()),
        "p_chance_at_least_observed": float((hits >= observed).mean()),
    }


def _score_protocol(
    s: NabSeries, test_ts: pd.DatetimeIndex, alarm: np.ndarray, events: pd.DataFrame, scored_windows
) -> dict:
    m = evaluate_zone(test_ts, alarm, events, tol_min=30, merge_gap=5, attribute_false=False,
                      step_min=STEP_MIN)
    seg_starts, _ = segments(alarm, 5 // STEP_MIN)
    offsets: list[float | None] = []
    for (a, b), label in scored_windows:
        lo, hi = int(test_ts.searchsorted(a)), int(test_ts.searchsorted(b))
        inside = [x for x in seg_starts if lo <= x < hi]
        offsets.append(None if not inside else (test_ts[min(inside)] - label) / pd.Timedelta(hours=1))
    lo_r, hi_r = wilson_interval(m.n_detected, m.n_events)
    return {
        "events_scored": m.n_events,
        "detected": m.n_detected,
        "recall_ci95": [lo_r, hi_r],
        "first_alarm_offset_h": offsets,  # negative = before the labelled failure timestamp
        "alarm_segments": m.n_segments,
        "false_segments": m.n_false_segments,
        "false_alarms_per_day": m.false_alarms_per_zone_day,
        "alarm_time_fraction": float(alarm.mean()),
        **_chance_baseline(test_ts, alarm, events, m.n_detected),
    }


def to_markdown(res: dict, notes: dict, n_scored_days: float) -> str:
    def off(v: list) -> str:
        return ", ".join("miss" if x is None else f"{x:+.1f}" for x in v)

    lines = [
        "# Real-data check: NAB machine temperature",
        "",
        "Public real sensor data (NAB `machine_temperature_system_failure`, 5-minute samples, "
        f"{notes['rows_clean']:,} rows, units not documented). It is an industrial machine, **not a "
        "freezer**: it has no compressor signals, so only the temperature-only detectors apply. "
        "This is a **sanity check, not a benchmark**: only 3 labelled failures are scorable "
        "(failure 1 falls inside NAB's 15% probationary fit period), so no detector can be shown "
        "better than another.",
        "",
        "Protocol fixed before running: fit on the probationary 15% (no labels); "
        f"**A** = threshold at the {PROBATION_QUANTILE:.1%} quantile of the detector's own "
        "probationary scores, no debounce; **B** = threshold + debounce tuned on *synthetic* "
        "validation data, unit-free z-score detectors only. No NAB label was used to choose "
        f"anything. Scored period: {n_scored_days:.0f} days.",
        "",
        "| Detector | Operating point | Failures detected (of 3, 95% CI) | Chance level: mean detected "
        "/ P(>= observed) | First alarm vs labelled failure (h) | False alarms / day | Time in alarm |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, d in res.items():
        for proto in ("A", "B"):
            r = d.get(proto)
            if r is None:
                continue
            lo, hi = r["recall_ci95"]
            lines.append(
                f"| {name} | {proto} | {r['detected']}/{r['events_scored']} ({lo:.2f}-{hi:.2f}) | "
                f"{r['chance_mean_detected']:.2f} / {r['p_chance_at_least_observed']:.2f} | "
                f"{off(r['first_alarm_offset_h'])} | {r['false_alarms_per_day']:.2f} | "
                f"{r['alarm_time_fraction']:.1%} |"
            )
    lines += [
        "",
        "Chance level: the alarm series is circularly shifted by a random offset "
        f"({N_SHIFTS} times), keeping its alarm count, durations and clustering; the columns give the "
        "mean number of failures that random placement would 'detect' and the probability it does at "
        "least as well as the detector. Labelled windows are ~47 h wide, so noisy detectors score by luck.",
        "",
        "First-alarm offsets are listed for failures 2, 3, 4 in order; negative = the alarm came "
        "before the labelled failure timestamp, \"miss\" = no alarm inside the labelled window.",
        "",
        f"Data notes: {json.dumps(notes)}",
    ]
    return "\n".join(lines) + "\n"


def plot(s: NabSeries, test_start: int, alarms: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ts, v = s.df["timestamp"], s.df["value"]
    ax.plot(ts, v, lw=0.5, color="#1f2a44")
    ax.axvspan(ts.iloc[0], ts.iloc[test_start], color="#9aa5b1", alpha=0.25, label="probationary (fit)")
    for i, (a, b) in enumerate(s.windows):
        ax.axvspan(a, b, color="#d1495b", alpha=0.25, label="labelled window" if i == 0 else None)
    for lab in s.labels:
        ax.axvline(lab, color="#d1495b", ls="--", lw=0.8)
    for k, (name, alarm) in enumerate(alarms.items()):
        idx = np.flatnonzero(alarm) + test_start
        y = np.full(len(idx), v.min() - 3 - 4 * k)
        ax.scatter(ts.iloc[idx], y, s=4, marker="|", label=f"alarm: {name}")
    ax.set_ylabel("sensor value")
    ax.set_title("NAB machine temperature: labelled failures and alarms (operating point A)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=7, ncol=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    s = load_nab()
    p = s.probation_end
    feats = build_univariate_features(s.df["value"], STEP_MIN)
    ts_all = pd.DatetimeIndex(s.df["timestamp"])
    test_ts = ts_all[p:]

    scored = [(w, lab) for w, lab in zip(s.windows, s.labels, strict=True) if w[1] > test_ts[0]]
    events = pd.DataFrame(
        [{"type": "machine_failure", "start": a, "end": b, "is_anomaly": True} for (a, b), _ in scored]
    )
    synth = json.loads(SYNTHETIC_RESULTS.read_text(encoding="utf-8"))

    results: dict[str, dict] = {}
    plot_alarms: dict[str, np.ndarray] = {}
    for name, det in _detectors():
        det.fit(feats.iloc[:p])
        score = det.score(feats)
        fit_scores = score[:p][np.isfinite(score[:p])]
        thr_a = float(np.quantile(fit_scores, PROBATION_QUANTILE))
        alarm_a = score[p:] > thr_a
        res = {"A": _score_protocol(s, test_ts, alarm_a, events, scored) | {"threshold": thr_a}}
        plot_alarms[name] = alarm_a

        if name in SYNTHETIC_NAMES:
            sy = synth[SYNTHETIC_NAMES[name]]
            min_run = math.ceil(sy["min_run"] / STEP_MIN)
            alarm_b = apply_min_run(score[p:] > sy["threshold"], min_run)
            res["B"] = _score_protocol(s, test_ts, alarm_b, events, scored) | {
                "threshold": sy["threshold"], "min_run_samples": min_run}
        results[name] = res
        with start_run("anomaly-nab-real", name):
            mlflow.log_params({"detector": name, "data": "NAB machine_temperature (real)",
                               "threshold_A": thr_a, "probation_quantile": PROBATION_QUANTILE})
            mlflow.log_metrics({"A_detected": res["A"]["detected"],
                                "A_false_alarms_per_day": res["A"]["false_alarms_per_day"]})
        r = res["A"]
        print(f"{name:24s} A: {r['detected']}/{r['events_scored']} detected, "
              f"{r['false_alarms_per_day']:.2f} FA/day, in alarm {r['alarm_time_fraction']:.1%}, "
              f"chance {r['chance_mean_detected']:.2f} (P>= {r['p_chance_at_least_observed']:.2f}), "
              f"offsets {[None if x is None else round(x, 1) for x in r['first_alarm_offset_h']]}")
        if "B" in res:
            r = res["B"]
            print(f"{'':24s} B: {r['detected']}/{r['events_scored']} detected, "
                  f"{r['false_alarms_per_day']:.2f} FA/day, in alarm {r['alarm_time_fraction']:.1%}, "
                  f"chance {r['chance_mean_detected']:.2f} (P>= {r['p_chance_at_least_observed']:.2f}), "
                  f"offsets {[None if x is None else round(x, 1) for x in r['first_alarm_offset_h']]}")

    OUT.mkdir(parents=True, exist_ok=True)
    days = len(test_ts) * STEP_MIN / 1440
    (OUT / "anomaly_nab.json").write_text(json.dumps({"results": results, "notes": s.notes}, indent=2),
                                          encoding="utf-8")
    (OUT / "anomaly_nab.md").write_text(to_markdown(results, s.notes, days), encoding="utf-8")
    plot(s, p, plot_alarms, OUT / "nab_machine_temperature.png")
    print("data notes:", s.notes)
    print(f"wrote {OUT / 'anomaly_nab.md'}")


if __name__ == "__main__":
    main()
