"""Train, tune and evaluate temperature-breach early warning on SYNTHETIC data.

Same split as the anomaly task: fit on seed 100, tune threshold/debounce on seed 101, report on
seeds 102-105. Run:  uv run python -m coldchain_ml.breach.run
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from coldchain_ml.anomaly.run import QUANTILES, TEST_SEEDS, TRAIN_SEED, VAL_SEED, History, load_history
from coldchain_ml.breach.data import BREACH_C, BREACH_FEATURES, HORIZON, build_breach_frame
from coldchain_ml.breach.eval import BreachMetrics, evaluate
from coldchain_ml.common.stats import wilson_interval
from coldchain_ml.common.tracking import start_run

MIN_RUNS = (1, 3, 5)
EXTRA_TRAIN_SEEDS = (106, 107, 108, 109, 110)  # more independent histories, for the learning curve
CURVE = (1, 3, 6)  # number of training histories; the largest is the final model
OUT = Path("docs/results")


@dataclass
class BreachSet:
    h: History
    frame: pd.DataFrame

    @property
    def at_risk(self) -> np.ndarray:
        return self.frame["at_risk"].to_numpy(dtype=bool)

    @property
    def eligible(self) -> np.ndarray:
        """At-risk rows with a known label and complete features (what PR-AUC is computed on)."""
        f = self.frame
        return self.at_risk & f["y"].notna().to_numpy() & f[BREACH_FEATURES].notna().all(axis=1).to_numpy()


def load(seed: int) -> BreachSet:
    h = load_history(seed)
    return BreachSet(h, build_breach_frame(h.readings, h.feats))


class TempThreshold:
    """Baseline 1 ('persistence'): the closer the temperature is to the limit, the higher the risk."""

    name = "current temperature (baseline)"
    needs_fit = False

    def score(self, f: pd.DataFrame) -> np.ndarray:
        return f["temp"].to_numpy()


class LinearExtrapolation:
    """Baseline 2: project the last 15 minutes' trend HORIZON minutes ahead."""

    name = "linear extrapolation (baseline)"
    needs_fit = False

    def score(self, f: pd.DataFrame) -> np.ndarray:
        return (f["temp"] + f["slope_15"] * HORIZON).to_numpy()


class LGBMBreach:
    needs_fit = True

    def __init__(self, n_histories: int, seed: int = 0) -> None:
        self.name = f"lightgbm ({n_histories} training histor{'y' if n_histories == 1 else 'ies'})"
        # Regularised after a first config (400 trees, 31 leaves) overfit badly: positives are few
        # and autocorrelated (a handful of episodes x many near-identical minutes). Chosen on the
        # validation seed from 3 variants; see docs/model-cards/breach.md.
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=200,
            reg_lambda=5.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, sets: list[BreachSet]) -> None:
        x = pd.concat([s.frame.loc[s.eligible, BREACH_FEATURES] for s in sets])
        y = pd.concat([s.frame.loc[s.eligible, "y"] for s in sets]).astype(int)
        self.model.fit(x, y)

    def score(self, f: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(f[BREACH_FEATURES])[:, 1]


def tune(det, val: BreachSet) -> tuple[float, int, float]:
    score = det.score(val.frame)
    pool = score[val.eligible]
    grid = {float(np.quantile(pool, q)) for q in QUANTILES}
    grid |= set(np.linspace(float(np.quantile(pool, 0.5)), float(pool.max()), 40).tolist())
    best = (-1.0, 0.0, 1)
    for thr in sorted(grid):
        for min_run in MIN_RUNS:
            m = evaluate(val.h.readings, val.h.events, score, val.at_risk, thr, min_run)
            f1 = m.f1(5.0)
            if f1 > best[0]:
                best = (f1, thr, min_run)
    return best[1], best[2], best[0]


def summarise(m: BreachMetrics, pr_pooled: float, pr_by_seed: dict[int, float], prevalence: float) -> dict:
    n = m.n_episodes
    w5, w10 = m.count_warned(5.0), m.count_warned(10.0)
    by_cause = {
        c: {
            "episodes": len(v),
            "warned_5min": m.warned_frac(5.0, c),
            "warned_10min": m.warned_frac(10.0, c),
            "median_lead_min": m.median_lead(c),
        }
        for c, v in sorted(m.leads.items())
    }
    return {
        "pr_auc": pr_pooled,
        "pr_auc_by_seed": pr_by_seed,
        "prevalence": prevalence,
        "episodes": n,
        "warned_any": m.warned_frac(1.0),
        "warned_5min": w5 / n if n else 0.0,
        "warned_5min_ci95": list(wilson_interval(w5, n)),
        "warned_10min": w10 / n if n else 0.0,
        "warned_10min_ci95": list(wilson_interval(w10, n)),
        "precision": m.precision,
        "precision_ci95": list(wilson_interval(m.n_true_segments, m.n_segments)),
        "false_alarms_per_zone_day": m.false_alarms_per_zone_day,
        "median_lead_min": m.median_lead(),
        "by_cause": by_cause,
    }


def to_markdown(results: dict[str, dict]) -> str:
    first = next(iter(results.values()))
    causes = sorted(first["by_cause"])
    fmt = lambda x: "—" if x is None else f"{x:.0f}"  # noqa: E731
    lines = [
        "# Temperature-breach early warning — SYNTHETIC data",
        "",
        f"Question: will the temperature exceed {BREACH_C:.0f} C within the next {HORIZON} min? "
        "Scored on at-risk minutes only (currently within limits, no breach in the last 10 min). "
        "Fit on seed 100, threshold + debounce tuned on seed 101, reported on seeds 102-105 pooled "
        f"({first['episodes']} breach episodes). **Simulator output: measures fit to this simulator, "
        "not real-world performance.**",
        "",
        f"At-risk-minute prevalence of a breach within {HORIZON} min: {first['prevalence']:.4f}. "
        "A random scorer's PR-AUC equals this prevalence.",
        "",
        "| Model | PR-AUC (per seed min-max) | Warned >=5 min (95% CI) | Warned >=10 min | "
        "Precision (95% CI) | False alarms / zone-day | Median lead (min) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        seeds = list(r["pr_auc_by_seed"].values())
        lines.append(
            f"| {name} | {r['pr_auc']:.3f} ({min(seeds):.3f}-{max(seeds):.3f}) | "
            f"{r['warned_5min']:.2f} ({r['warned_5min_ci95'][0]:.2f}-{r['warned_5min_ci95'][1]:.2f}) | "
            f"{r['warned_10min']:.2f} | {r['precision']:.2f} "
            f"({r['precision_ci95'][0]:.2f}-{r['precision_ci95'][1]:.2f}) | "
            f"{r['false_alarms_per_zone_day']:.2f} | {fmt(r['median_lead_min'])} |"
        )
    lines += ["", "## Breaches warned >=5 min ahead, by cause (median lead of warned ones)", ""]
    headers = [f"{c} (n={first['by_cause'][c]['episodes']})" for c in causes]
    lines.append("| Model | " + " | ".join(headers) + " |")
    lines.append("|---|" + "---|" * len(causes))
    for name, r in results.items():
        cells = [
            f"{r['by_cause'][c]['warned_5min']:.2f} / {fmt(r['by_cause'][c]['median_lead_min'])} min"
            if c in r["by_cause"]
            else "—"
            for c in causes
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    train_sets = [load(s) for s in (TRAIN_SEED, *EXTRA_TRAIN_SEEDS)]
    val = load(VAL_SEED)
    tests = [load(s) for s in TEST_SEEDS]
    elig_y = np.concatenate([t.frame.loc[t.eligible, "y"].to_numpy() for t in tests])
    prevalence = float(elig_y.mean())
    print(f"test at-risk minutes: {len(elig_y):,}, positives: {int(elig_y.sum()):,} ({prevalence:.4f})")

    results: dict[str, dict] = {}
    importance: dict[str, float] = {}
    detectors = [TempThreshold(), LinearExtrapolation(), *(LGBMBreach(n, seed=0) for n in CURVE)]
    for det in detectors:
        with start_run("breach-synthetic", det.name):
            if det.needs_fit:
                det.fit(train_sets[: int(det.name.split('(')[1].split()[0])])
            thr, min_run, val_f1 = tune(det, val)

            pooled = BreachMetrics()
            ys, ss, by_seed = [], [], {}
            for t in tests:
                score = det.score(t.frame)
                pooled.merge(evaluate(t.h.readings, t.h.events, score, t.at_risk, thr, min_run))
                e = t.eligible
                y, s = t.frame.loc[e, "y"].to_numpy(), score[e]
                ys.append(y)
                ss.append(s)
                by_seed[t.h.seed] = float(average_precision_score(y, s))
            pr = float(average_precision_score(np.concatenate(ys), np.concatenate(ss)))

            res = summarise(pooled, pr, by_seed, prevalence)
            res.update({"threshold": thr, "min_run": min_run, "val_f1_warned5": val_f1})
            results[det.name] = res

            mlflow.log_params({"model": det.name, "threshold": thr, "min_run": min_run,
                               "horizon_min": HORIZON, "breach_c": BREACH_C, "data": "synthetic"})
            mlflow.log_metrics({"test_pr_auc": pr, "test_warned_5min": res["warned_5min"],
                                "test_precision": res["precision"],
                                "false_alarms_per_zone_day": res["false_alarms_per_zone_day"]})
            if isinstance(det, LGBMBreach) and det.name == detectors[-1].name:
                gain = det.model.booster_.feature_importance("gain")
                importance = dict(sorted(zip(BREACH_FEATURES, (gain / gain.sum()).tolist(), strict=True),
                                         key=lambda kv: -kv[1]))
                mlflow.lightgbm.log_model(det.model, "model")
        print(f"{det.name:40s} PR-AUC {pr:.3f} | warned>=5min {res['warned_5min']:.2f} "
              f">=10min {res['warned_10min']:.2f} | precision {res['precision']:.2f} "
              f"FA/zone-day {res['false_alarms_per_zone_day']:.2f} | median lead {res['median_lead_min']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "breach_synthetic.json").write_text(
        json.dumps({"results": results, "lightgbm_gain_importance": importance}, indent=2), encoding="utf-8"
    )
    (OUT / "breach_synthetic.md").write_text(to_markdown(results), encoding="utf-8")
    print("top features:", {k: round(v, 3) for k, v in list(importance.items())[:6]})
    print(f"wrote {OUT / 'breach_synthetic.md'}")


if __name__ == "__main__":
    main()
