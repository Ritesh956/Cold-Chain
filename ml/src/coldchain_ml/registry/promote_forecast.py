"""Train the history-only outflow forecaster on ALL of M5 FOODS and register it as `coldchain-forecast`.

Gate: the same model refit at the last test origin must reproduce the WAPE recorded in
docs/results/forecast_m5.json for the history-and-calendar-only variant, so a registered model is the one the
model card describes. The registered model itself is then refit through the last day of the data.

Needs data/real/m5 (scripts/download_m5.py) and a finished `coldchain_ml.forecast.run`.

Run:  uv run python -m coldchain_ml.registry.promote_forecast
      (MLFLOW_TRACKING_URI selects the registry; default is the local ./mlruns file store)
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.lightgbm

from coldchain_ml.common.tracking import start_run
from coldchain_ml.forecast.baselines import HORIZON
from coldchain_ml.forecast.data import load_demand
from coldchain_ml.forecast.features import MIN_ORIGIN, SERVED_FEATURES, Context
from coldchain_ml.forecast.metrics import wape
from coldchain_ml.forecast.run import CANDIDATES, N_BLOCKS, SERVED, fit, forecast_origin
from coldchain_ml.registry.promote import _gate, _register

NAME = "coldchain-forecast"
RESULTS = Path("docs/results/forecast_m5.json")
MAX_HISTORY_DAYS = 400  # request size bound; only the last MIN_ORIGIN days enter the features


def main() -> None:
    reported = json.loads(RESULTS.read_text(encoding="utf-8"))
    params = CANDIDATES[reported["chosen"]]
    d = load_demand()
    ctx = Context(d)
    last = d.n_days - 1

    last_test_origin = last - HORIZON
    check = forecast_origin(
        ctx, {SERVED: fit(ctx, last_test_origin, params, SERVED_FEATURES)}, last_test_origin
    )
    print("forecast model:")
    _gate(
        "WAPE at the last test origin",
        wape(check["actual"], check["methods"][SERVED]),
        reported["summary"][SERVED]["wape_per_origin"][N_BLOCKS - 1],
        1e-6,
    )

    model = fit(ctx, last, params, SERVED_FEATURES)  # all data: targets are observed up to the last day
    summary = reported["summary"]
    config = {
        "type": "lightgbm_outflow_forecast",
        "features": SERVED_FEATURES,
        "horizon_days": HORIZON,
        "min_history_days": MIN_ORIGIN,
        "max_history_days": MAX_HISTORY_DAYS,
        "data": "REAL M5 FOODS retail sales (units per item per day, summed over 10 stores), "
        "reframed as outflow",
        "trained_through": str(d.dates[last].date()),
        "params": {"config": reported["chosen"], **params},
        "reported_test": {
            "wape": summary[SERVED]["wape"],
            "mase": summary[SERVED]["mase"],
            "bias": summary[SERVED]["bias"],
            "best_baseline_wape": min(v["wape"] for k, v in summary.items() if k in reported["bootstrap"]),
            "protocol": "4 rolling 28-day origins, refit per origin; see docs/results/forecast_m5.md",
        },
    }
    with start_run("registry", "promote forecast") as run:
        mlflow.lightgbm.log_model(model, "model")
        mlflow.log_dict(config, "model/serving_config.json")
        mlflow.log_metrics({"test_wape": summary[SERVED]["wape"], "test_mase": summary[SERVED]["mase"]})
        version = _register(run.info.run_id, "model", NAME)
    print(f"  registered {NAME} v{version} @champion")


if __name__ == "__main__":
    main()
