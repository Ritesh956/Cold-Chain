"""Train, tune and evaluate freezer anomaly detectors on SYNTHETIC data.

Splits are separate simulator histories (independent fault schedules and noise), not slices
of one history, so the test set has enough events to be meaningful:
  train seed 100 | validation seed 101 (threshold tuning) | test seeds 102-105 (pooled)

Run:  uv run python -m coldchain_ml.anomaly.run
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from coldchain_simulator import SimConfig, generate, point_labels

from coldchain_ml.anomaly.detectors import Detector, all_detectors
from coldchain_ml.anomaly.events_eval import EventMetrics, evaluate
from coldchain_ml.anomaly.features import build_features
from coldchain_ml.common.stats import wilson_interval
from coldchain_ml.common.tracking import start_run

TRAIN_SEED, VAL_SEED, TEST_SEEDS = 100, 101, (102, 103, 104, 105)
QUANTILES = (0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999)
MIN_RUNS = (1, 5, 15)
CACHE = Path("data/synthetic/anomaly_eval")
OUT = Path("docs/results")


@dataclass
class History:
    seed: int
    readings: pd.DataFrame
    events: pd.DataFrame
    feats: pd.DataFrame
    labels: pd.Series


def load_history(seed: int) -> History:
    d = CACHE / f"seed{seed}"
    r_path, e_path = d / "readings.parquet", d / "events.parquet"
    if r_path.exists() and e_path.exists():
        readings, events = pd.read_parquet(r_path), pd.read_parquet(e_path)
    else:
        readings, events = generate(replace(SimConfig(), seed=seed))
        d.mkdir(parents=True, exist_ok=True)
        readings.to_parquet(r_path, index=False)
        events.to_parquet(e_path, index=False)
    return History(seed, readings, events, build_features(readings), point_labels(readings, events))


def tune(det: Detector, val: History) -> tuple[float, int, float]:
    """Pick (threshold, min_run) maximising event-level F1 on the validation history."""
    score = det.score(val.feats)
    finite = score[np.isfinite(score)]
    best = (-1.0, 0.0, 1)
    # quantiles suit unbounded scores; a linear sweep over the upper range suits probabilities
    grid = {float(np.quantile(finite, q)) for q in QUANTILES}
    grid |= set(np.linspace(float(np.quantile(finite, 0.5)), float(finite.max()), 40).tolist())
    for thr in sorted(grid):
        for min_run in MIN_RUNS:
            m = evaluate(
                val.readings, val.events, score > thr, min_run=min_run, attribute_false=False
            )
            if m.f1 > best[0]:
                best = (m.f1, thr, min_run)
    return best[1], best[2], best[0]


def evaluate_on_test(det: Detector, tests: list[History], thr: float, min_run: int) -> EventMetrics:
    total = EventMetrics()
    for h in tests:
        total.merge(evaluate(h.readings, h.events, det.score(h.feats) > thr, min_run=min_run))
    return total


def summarise(m: EventMetrics) -> dict:
    r_lo, r_hi = wilson_interval(m.n_detected, m.n_events)
    p_lo, p_hi = wilson_interval(m.n_true_segments, m.n_segments)
    return {
        "events": m.n_events,
        "recall": m.recall,
        "recall_ci95": [r_lo, r_hi],
        "precision": m.precision,
        "precision_ci95": [p_lo, p_hi],
        "f1": m.f1,
        "false_alarms_per_zone_day": m.false_alarms_per_zone_day,
        "median_delay_min": m.median_delay(),
        "recall_by_type": {k: m.detected_by_type[k] / v for k, v in m.events_by_type.items()},
        "events_by_type": dict(m.events_by_type),
        "median_delay_by_type_min": {k: m.median_delay(k) for k in m.events_by_type},
        "false_alarms_by_cause": dict(m.false_by_cause),
    }


def to_markdown(results: dict[str, dict]) -> str:
    def f(x: float | None, nd: int = 2) -> str:
        return "—" if x is None else f"{x:.{nd}f}"

    lines = [
        "# Freezer anomaly detection — SYNTHETIC data",
        "",
        "Event-level metrics on 4 pooled test histories (seeds 102-105, 4 zones x 90 days each). "
        "Threshold and debounce tuned on a separate validation history (seed 101); models fit on "
        "seed 100. **All data is simulator output: these numbers measure how well a detector learns "
        "this simulator, not real-world performance.**",
        "",
        "| Detector | Recall (95% CI) | Precision (95% CI) | F1 | False alarms / zone-day "
        "| Median delay (min) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        rc, pc = r["recall_ci95"], r["precision_ci95"]
        lines.append(
            f"| {name} | {r['recall']:.2f} ({rc[0]:.2f}-{rc[1]:.2f}) | "
            f"{r['precision']:.2f} ({pc[0]:.2f}-{pc[1]:.2f}) | {r['f1']:.2f} | "
            f"{r['false_alarms_per_zone_day']:.2f} | {f(r['median_delay_min'], 0)} |"
        )
    kinds = sorted(next(iter(results.values()))["events_by_type"])
    lines += ["", "## Recall and median delay by fault type", "", "| Detector | " + " | ".join(kinds) + " |"]
    lines.append("|---|" + "---|" * len(kinds))
    for name, r in results.items():
        cells = [
            f"{r['recall_by_type'][k]:.2f} / {f(r['median_delay_by_type_min'][k], 0)} min" for k in kinds
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    n = {k: v for k, v in next(iter(results.values()))["events_by_type"].items()}
    lines += ["", f"Test events per type: {n}.", "", "## False alarms by cause", ""]
    lines.append("| Detector | defrost | routine_door | unexplained |")
    lines.append("|---|---|---|---|")
    for name, r in results.items():
        c = r["false_alarms_by_cause"]
        lines.append(
            f"| {name} | {c.get('defrost', 0)} | {c.get('routine_door', 0)} | {c.get('unexplained', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    train, val = load_history(TRAIN_SEED), load_history(VAL_SEED)
    tests = [load_history(s) for s in TEST_SEEDS]
    results: dict[str, dict] = {}

    for det in all_detectors(seed=0):
        with start_run("anomaly-synthetic", det.name):
            det.fit(train.feats, train.labels if det.needs_labels else None)
            thr, min_run, val_f1 = tune(det, val)
            m = evaluate_on_test(det, tests, thr, min_run)
            res = summarise(m)
            res.update({"threshold": thr, "min_run": min_run, "val_f1": val_f1})
            results[det.name] = res

            mlflow.log_params(
                {"detector": det.name, "threshold": thr, "min_run": min_run, "train_seed": TRAIN_SEED,
                 "val_seed": VAL_SEED, "test_seeds": list(TEST_SEEDS), "data": "synthetic"}
            )
            mlflow.log_metrics(
                {"val_f1": val_f1, "test_f1": m.f1, "test_recall": m.recall, "test_precision": m.precision,
                 "false_alarms_per_zone_day": m.false_alarms_per_zone_day}
            )
        print(
            f"{det.name:38s} val F1 {val_f1:.2f} | test recall {m.recall:.2f} "
            f"precision {m.precision:.2f} F1 {m.f1:.2f} FA/zone-day {m.false_alarms_per_zone_day:.2f}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "anomaly_synthetic.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "anomaly_synthetic.md").write_text(to_markdown(results), encoding="utf-8")
    print(f"wrote {OUT / 'anomaly_synthetic.md'}")


if __name__ == "__main__":
    main()
