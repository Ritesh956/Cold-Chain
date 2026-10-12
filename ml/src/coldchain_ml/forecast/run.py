"""Rolling-origin evaluation of 28-day warehouse-outflow forecasts on M5 FOODS (real retail data,
reframed). Protocol fixed before any test origin was scored:

  * Hyperparameters: 4 candidate configurations compared on ONE validation origin (day 1800); the best
    by WAPE is used unchanged for every test origin. No test origin is used to choose anything.
  * Test: 4 consecutive 28-day blocks ending at the last day of the data. For each origin the model is
    refit on history up to that origin only (targets already observed), then forecasts 28 days.
  * Items scored at an origin: those already on sale at the origin (same set for every method).

Run:  uv run python -m coldchain_ml.forecast.run     (after scripts/download_m5.py)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRegressor

from coldchain_ml.common.tracking import start_run
from coldchain_ml.forecast.baselines import BASELINES, HORIZON
from coldchain_ml.forecast.data import Demand, load_demand
from coldchain_ml.forecast.features import FEATURES, Context, origin_frame, training_frame
from coldchain_ml.forecast.metrics import bias, mase, naive_scale, rmse, wape

VAL_ORIGIN = 1800
N_BLOCKS = 4
OUT = Path("docs/results")
CANDIDATES: dict[str, dict[str, Any]] = {
    "tweedie 1.1, 63 leaves": dict(
        objective="tweedie", tweedie_variance_power=1.1, num_leaves=63, n_estimators=400
    ),
    "poisson, 63 leaves": dict(objective="poisson", num_leaves=63, n_estimators=400),
    "L1, 63 leaves": dict(objective="regression_l1", num_leaves=63, n_estimators=400),
    "tweedie 1.1, 31 leaves": dict(
        objective="tweedie", tweedie_variance_power=1.1, num_leaves=31, n_estimators=400
    ),
}
COMMON: dict[str, Any] = dict(
    learning_rate=0.05, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=0, n_jobs=16, verbose=-1, force_row_wise=True,
)  # fmt: skip
CATEGORICAL = ["dept", "event_type"]


def fit(ctx: Context, last_origin: int, params: dict) -> LGBMRegressor:
    tr = training_frame(ctx, last_origin)
    model = LGBMRegressor(**COMMON, **params)
    model.fit(tr[FEATURES], tr["y"], categorical_feature=CATEGORICAL)
    return model


def forecast_origin(ctx: Context, model: LGBMRegressor, o: int) -> dict:
    """Forecasts of every method for the items on sale at origin o, as (n_items, HORIZON) arrays."""
    d = ctx.d
    fr = origin_frame(ctx, o, True)
    items = np.sort(fr["item"].unique().astype(int))
    pred = np.clip(model.predict(fr[FEATURES]), 0, None).reshape(len(items), HORIZON)
    actual = fr["y"].to_numpy().reshape(len(items), HORIZON)
    methods = {"LightGBM (global)": pred}
    for name, fn in BASELINES.items():
        methods[name] = fn(d.units, o)[items]
    return {"items": items, "actual": actual, "methods": methods, "mean90": ctx.window_mean(o, 90)[items]}


def evaluate_val(ctx: Context) -> dict:
    out = {}
    for name, params in CANDIDATES.items():
        t0 = time.time()
        r = forecast_origin(ctx, fit(ctx, VAL_ORIGIN, params), VAL_ORIGIN)
        w = wape(r["actual"], r["methods"]["LightGBM (global)"])
        out[name] = {"wape": w, "seconds": round(time.time() - t0)}
        print(f"  val {name:26s} WAPE {w:.4f}  ({out[name]['seconds']} s)", flush=True)
    return out


def _bucket_rows(res: list[dict], name: str) -> dict:
    """WAPE by horizon bucket and by volume tercile (terciles of each origin's 90-day mean)."""
    by_h = {}
    for lo, hi in ((1, 7), (8, 14), (15, 21), (22, 28)):
        a = np.concatenate([r["actual"][:, lo - 1 : hi].ravel() for r in res])
        p = np.concatenate([r["methods"][name][:, lo - 1 : hi].ravel() for r in res])
        by_h[f"{lo}-{hi}"] = wape(a, p)
    by_v: dict[str, float] = {}
    for label, q in (
        ("slow third", (0, 1 / 3)),
        ("middle third", (1 / 3, 2 / 3)),
        ("fast third", (2 / 3, 1.0)),
    ):
        acts: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for r in res:
            lo_v, hi_v = np.quantile(r["mean90"], q[0]), np.quantile(r["mean90"], q[1])
            sel = (r["mean90"] >= lo_v) & ((r["mean90"] <= hi_v) if q[1] == 1.0 else (r["mean90"] < hi_v))
            acts.append(r["actual"][sel].ravel())
            preds.append(r["methods"][name][sel].ravel())
        by_v[label] = wape(np.concatenate(acts), np.concatenate(preds))
    return {"by_horizon": by_h, "by_volume": by_v}


def summarise(d: Demand, res: list[dict], origins: list[int]) -> dict:
    methods = list(res[0]["methods"])
    table: dict[str, dict] = {}
    for name in methods:
        a = np.concatenate([r["actual"].ravel() for r in res])
        p = np.concatenate([r["methods"][name].ravel() for r in res])
        scales = [naive_scale(d.units, o, _first_on(d)) for o in origins]
        masel = [
            mase(r["actual"], r["methods"][name], s[r["items"]]) for r, s in zip(res, scales, strict=True)
        ]
        daily_a = np.concatenate([r["actual"].sum(axis=0) for r in res])
        daily_p = np.concatenate([r["methods"][name].sum(axis=0) for r in res])
        table[name] = {
            "wape": wape(a, p),
            "wape_per_origin": [wape(r["actual"], r["methods"][name]) for r in res],
            "mase": float(np.mean(masel)),
            "rmse": rmse(a, p),
            "bias": bias(a, p),
            "wape_total_daily_outflow": wape(daily_a, daily_p),
            **_bucket_rows(res, name),
        }
    return table


def _first_on(d: Demand) -> np.ndarray:
    on = ~np.isnan(d.price)
    return np.where(on.any(axis=1), on.argmax(axis=1), d.n_days)


def paired_bootstrap(res: list[dict], a_name: str, b_name: str, n: int = 2000, seed: int = 0) -> dict:
    """95% CI of WAPE(a) - WAPE(b), resampling items with replacement (an item's error pooled over
    origins). Negative = a better. Items are the unit of resampling because their errors are correlated."""
    n_items = max(int(r["items"].max()) for r in res) + 1
    ea, eb, ya = (np.zeros(n_items) for _ in range(3))
    for r in res:
        np.add.at(ea, r["items"], np.abs(r["actual"] - r["methods"][a_name]).sum(axis=1))
        np.add.at(eb, r["items"], np.abs(r["actual"] - r["methods"][b_name]).sum(axis=1))
        np.add.at(ya, r["items"], r["actual"].sum(axis=1))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_items, size=(n, n_items))
    diff = (ea[idx].sum(1) - eb[idx].sum(1)) / ya[idx].sum(1)
    return {
        "diff": float((ea.sum() - eb.sum()) / ya.sum()),
        "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
        "share_items_a_better": float((ea < eb).mean()),
    }


def to_markdown(summary: dict, val: dict, chosen: str, boot: dict, meta: dict, importance: list) -> str:
    best_base = min((k for k in summary if k != "LightGBM (global)"), key=lambda k: summary[k]["wape"])
    lines = [
        "# Volume forecast on M5 FOODS (real retail data, reframed as warehouse outflow)",
        "",
        "Target: units shipped per item per day, 28 days ahead, summed over the 10 M5 stores (a distribution "
        f"centre's outflow). {meta['n_items']} FOODS items, {meta['n_days']} days. **Real sales data, but the "
        "'warehouse' framing is mine**: sales are not stock demand, and nothing here converts units to pallets.",
        "",
        f"Protocol: 4 rolling 28-day test blocks ending at the last day of the data (origins at day "
        f"{', '.join(map(str, meta['origins']))}); each model refit on history up to its origin. "
        f"LightGBM configuration chosen on one earlier validation origin (day {VAL_ORIGIN}); no test origin "
        "was used to choose anything.",
        "",
        "## Test results (pooled over the 4 origins)",
        "",
        "| Method | WAPE | MASE | RMSE | Bias | WAPE of daily total outflow |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m['wape']:.3f} | {m['mase']:.3f} | {m['rmse']:.2f} | {m['bias']:+.1%} | "
            f"{m['wape_total_daily_outflow']:.3f} |"
        )
    d = boot[best_base]
    lines += [
        "",
        f"LightGBM minus best baseline ({best_base}) in WAPE: {d['diff']:+.4f}, 95% CI "
        f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] (items resampled, 2,000 draws); LightGBM has the lower "
        f"error on {d['share_items_a_better']:.0%} of items.",
        "",
        "WAPE per test origin: "
        + "; ".join(
            f"{k}: " + ", ".join(f"{x:.3f}" for x in v["wape_per_origin"]) for k, v in summary.items()
        ),
        "",
        "MASE is the mean over items of (item MAE / item in-sample seasonal-naive MAE, last 365 days from "
        "launch); items with no in-sample variation are skipped. WAPE = sum|error| / sum(actual).",
        "",
        "## WAPE by horizon and by item volume",
        "",
        "| Method | days 1-7 | 8-14 | 15-21 | 22-28 | slow third | middle third | fast third |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        h, v = m["by_horizon"], m["by_volume"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{h[k]:.3f}" for k in h)
            + " | "
            + " | ".join(f"{v[k]:.3f}" for k in v)
            + " |"
        )
    lines += [
        "",
        "Volume thirds are terciles of each origin's 90-day mean daily outflow.",
        "",
        "## Choosing the LightGBM configuration (validation origin only)",
        "",
        "| Configuration | validation WAPE |",
        "|---|---|",
    ]
    for name, v in val.items():
        lines.append(f"| {name}{' **(chosen)**' if name == chosen else ''} | {v['wape']:.4f} |")
    lines += [
        "",
        "## Most important features (LightGBM gain, final origin)",
        "",
        ", ".join(f"`{n}` ({g:.1%})" for n, g in importance),
        "",
        f"Notes: {json.dumps(meta['notes'])}",
    ]
    return "\n".join(lines) + "\n"


def plot(d: Demand, res: list[dict], origins: list[int], best_base: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = np.concatenate([np.arange(o + 1, o + 1 + HORIZON) for o in origins])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(
        d.dates[days],
        np.concatenate([r["actual"].sum(0) for r in res]),
        color="#1f2a44",
        lw=1.4,
        label="actual",
    )
    for name, c in (("LightGBM (global)", "#d1495b"), (best_base, "#6c9a8b")):
        ax.plot(
            d.dates[days], np.concatenate([r["methods"][name].sum(0) for r in res]), color=c, lw=1, label=name
        )
    ax.set_ylabel("units / day, all FOODS items")
    ax.set_title(
        "Total daily outflow in the four test blocks: actual vs forecast (items on sale at each origin)"
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    d = load_demand()
    ctx = Context(d)
    last = d.n_days - 1
    origins = [last - HORIZON * k for k in range(N_BLOCKS, 0, -1)]
    print(f"loaded {d.n_items} items x {d.n_days} days; test origins {origins}", flush=True)

    with start_run("forecast-m5", "rolling-origin") as run:
        val = evaluate_val(ctx)
        chosen = min(val, key=lambda k: val[k]["wape"])
        print(f"chosen on validation: {chosen}", flush=True)

        res: list[dict] = []
        last_model: LGBMRegressor | None = None
        for o in origins:
            last_model = fit(ctx, o, CANDIDATES[chosen])
            res.append(forecast_origin(ctx, last_model, o))
            print(
                f"  origin {o}: LightGBM WAPE {wape(res[-1]['actual'], res[-1]['methods']['LightGBM (global)']):.4f}",
                flush=True,
            )

        summary = summarise(d, res, origins)
        best_base = min((k for k in summary if k != "LightGBM (global)"), key=lambda k: summary[k]["wape"])
        boot = {k: paired_bootstrap(res, "LightGBM (global)", k) for k in summary if k != "LightGBM (global)"}
        assert last_model is not None
        gain = last_model.booster_.feature_importance("gain")
        order = np.argsort(gain)[::-1][:8]
        importance = [(FEATURES[i], float(gain[i] / gain.sum())) for i in order]
        meta = {
            "n_items": d.n_items, "n_days": d.n_days, "origins": origins,
            "notes": {
                "items_scored_per_origin": [int(len(r["items"])) for r in res],
                "unit": "units/day summed over 10 stores",
                "first_day": str(d.dates[0].date()), "last_day": str(d.dates[-1].date()),
            },
        }  # fmt: skip

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "forecast_m5.json").write_text(
            json.dumps({"summary": summary, "validation": val, "chosen": chosen, "bootstrap": boot,
                        "importance": importance, "meta": meta}, indent=2),
            encoding="utf-8",
        )  # fmt: skip
        (OUT / "forecast_m5.md").write_text(
            to_markdown(summary, val, chosen, boot, meta, importance), encoding="utf-8"
        )
        plot(d, res, origins, best_base, OUT / "forecast_m5.png")
        import mlflow

        mlflow.log_params({"chosen": chosen, "origins": str(origins)})
        mlflow.log_metrics(
            {f"wape_{re.sub(r'[^A-Za-z0-9_.-]+', '_', k).strip('_')}": v["wape"] for k, v in summary.items()}
        )
        print(f"done in {time.time() - t0:.0f} s; run {run.info.run_id}", flush=True)


if __name__ == "__main__":
    main()
