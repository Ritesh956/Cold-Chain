"""Breach models compared at MATCHED false-alarm budgets (SYNTHETIC data).

`run.py` tunes each model for its own best F1, so the models sit at different false-alarm rates and
their rows are not directly comparable. Here every model gets the same budget: for each budget, the
threshold and debounce are chosen on the validation history (seed 101) to warn as many breaches
>= 5 min ahead as possible while staying at or under the budget, then applied unchanged to the
test histories (seeds 102-105). The test false-alarm rate is reported as measured; it can differ from
the budget because the threshold was not tuned on it.

Run:  uv run python -m coldchain_ml.breach.matched
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from coldchain_ml.breach.eval import BreachMetrics, evaluate
from coldchain_ml.breach.run import (
    CURVE,
    EXTRA_TRAIN_SEEDS,
    MIN_RUNS,
    OUT,
    QUANTILES,
    TEST_SEEDS,
    TRAIN_SEED,
    VAL_SEED,
    LGBMBreach,
    LinearExtrapolation,
    TempThreshold,
    load,
)
from coldchain_ml.common.stats import wilson_interval

BUDGETS = (0.01, 0.03, 0.05, 0.10, 0.22)  # false alarms per zone-day; 0.22 is the temperature baseline's own


@dataclass
class Candidate:
    thr: float
    min_run: int
    fa: float
    warned5: float


def grid_for(score: np.ndarray, eligible: np.ndarray) -> list[float]:
    pool = score[eligible]
    grid = {float(np.quantile(pool, q)) for q in QUANTILES}
    grid |= set(np.linspace(float(np.quantile(pool, 0.5)), float(pool.max()), 40).tolist())
    return sorted(grid)


def candidates(det, val) -> list[Candidate]:
    score = det.score(val.frame)
    out = []
    for thr in grid_for(score, val.eligible):
        for min_run in MIN_RUNS:
            m = evaluate(val.h.readings, val.h.events, score, val.at_risk, thr, min_run)
            out.append(Candidate(thr, min_run, m.false_alarms_per_zone_day, m.warned_frac(5.0)))
    return out


def pick(cands: list[Candidate], budget: float) -> Candidate | None:
    ok = [c for c in cands if c.fa <= budget]
    # most breaches warned within budget; among ties, fewest false alarms
    return max(ok, key=lambda c: (c.warned5, -c.fa)) if ok else None


def main() -> None:
    train_sets = [load(s) for s in (TRAIN_SEED, *EXTRA_TRAIN_SEEDS)]
    val = load(VAL_SEED)
    tests = [load(s) for s in TEST_SEEDS]

    lgbm = LGBMBreach(CURVE[-1], seed=0)
    lgbm.fit(train_sets[: CURVE[-1]])
    detectors = [TempThreshold(), LinearExtrapolation(), lgbm]

    results: dict[str, dict] = {}
    for det in detectors:
        cands = candidates(det, val)
        rows = {}
        for b in BUDGETS:
            c = pick(cands, b)
            if c is None:
                rows[str(b)] = None  # the model cannot get this quiet on validation
                continue
            pooled = BreachMetrics()
            for t in tests:
                score = det.score(t.frame)
                pooled.merge(evaluate(t.h.readings, t.h.events, score, t.at_risk, c.thr, c.min_run))
            n = pooled.n_episodes
            w5 = pooled.count_warned(5.0)
            rows[str(b)] = {
                "threshold": c.thr,
                "min_run": c.min_run,
                "val_false_alarms_per_zone_day": c.fa,
                "test_false_alarms_per_zone_day": pooled.false_alarms_per_zone_day,
                "test_warned_5min": w5 / n,
                "test_warned_5min_ci95": list(wilson_interval(w5, n)),
                "test_episodes": n,
                "test_warned_5min_by_cause": {k: pooled.warned_frac(5.0, k) for k in sorted(pooled.leads)},
                "test_episodes_by_cause": {k: len(v) for k, v in sorted(pooled.leads.items())},
            }
        results[det.name] = rows
        print(det.name)
        for b, r in rows.items():
            if r:
                by_cause = {k: round(v, 2) for k, v in r["test_warned_5min_by_cause"].items()}
                print(
                    f"  budget {b:>5}: test FA {r['test_false_alarms_per_zone_day']:.3f}  "
                    f"warned>=5min {r['test_warned_5min']:.2f}  by cause {by_cause}"
                )
            else:
                print(f"  budget {b:>5}: not reachable")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "breach_matched.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "breach_matched.md").write_text(to_markdown(results), encoding="utf-8")
    print(f"wrote {OUT / 'breach_matched.md'}")


def to_markdown(results: dict[str, dict]) -> str:
    names = list(results)
    lines = [
        "# Breach early warning at matched false-alarm budgets — SYNTHETIC data",
        "",
        "Each model's threshold and debounce are chosen on a validation history to warn the most breaches "
        ">= 5 min ahead while staying at or under the budget, then applied unchanged to the pooled test "
        "histories (188 breach episodes). Cells are test results: share warned >= 5 min ahead (95% Wilson "
        "interval), with the measured test false alarms per zone-day in brackets. The budget is met on "
        "validation; the test rate can differ. **Simulator data: this measures fit to the simulator.**",
        "",
        "| Budget (false alarms / zone-day) | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for b in BUDGETS:
        cells = []
        for n in names:
            r = results[n][str(b)]
            if r is None:
                cells.append("not reachable")
            else:
                lo, hi = r["test_warned_5min_ci95"]
                fa = r["test_false_alarms_per_zone_day"]
                cells.append(f"{r['test_warned_5min']:.2f} ({lo:.2f}-{hi:.2f}) [{fa:.2f}]")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    sample = results[names[-1]]["0.05"] or {}
    causes = sorted(sample.get("test_warned_5min_by_cause", {}))
    lines += ["", "## By cause at the 0.05 budget (share warned >= 5 min ahead)", ""]
    lines += ["| Model | " + " | ".join(causes) + " |", "|---|" + "---|" * len(causes)]
    for n in names:
        r = results[n]["0.05"]
        cells = ["n/a" if r is None else f"{r['test_warned_5min_by_cause'].get(c, 0):.2f}" for c in causes]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
