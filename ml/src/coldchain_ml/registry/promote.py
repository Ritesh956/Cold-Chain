"""Refit the production models, check they reproduce the REPORTED metrics, then register them in
MLflow with the alias `champion`.

The gate matters: a registered model must be the one the model card describes. If a refit does
not reproduce the numbers in docs/results/*.json, nothing is registered.

Run:  uv run python -m coldchain_ml.registry.promote
      (MLFLOW_TRACKING_URI selects the registry; default is the local ./mlruns file store)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score

from coldchain_ml.anomaly.detectors import RobustZScore
from coldchain_ml.anomaly.run import TEST_SEEDS, TRAIN_SEED, VAL_SEED, evaluate_on_test, load_history
from coldchain_ml.breach.data import BREACH_C, BREACH_FEATURES, HORIZON
from coldchain_ml.breach.eval import BreachMetrics
from coldchain_ml.breach.eval import evaluate as evaluate_breach
from coldchain_ml.breach.run import CURVE, EXTRA_TRAIN_SEEDS, LGBMBreach, load
from coldchain_ml.common.tracking import start_run
from coldchain_ml.contract import ANOMALY_MIN_HISTORY, BREACH_MIN_HISTORY

RESULTS = Path("docs/results")
ANOMALY_MODEL, BREACH_MODEL, ALIAS = "coldchain-anomaly", "coldchain-breach", "champion"
ANOMALY_KEY = "robust z-score (temp + compressor current + compressor-off duration)"


class GateFailed(RuntimeError):
    pass


def _gate(name: str, got: float, want: float, tol: float) -> None:
    if abs(got - want) > tol:
        raise GateFailed(f"{name}: refit gives {got:.4f}, reported {want:.4f} (tolerance {tol})")
    print(f"  gate ok  {name}: refit {got:.4f} vs reported {want:.4f}")


def _register(run_id: str, artifact_path: str, name: str) -> str:
    mv = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name)
    MlflowClient().set_registered_model_alias(name, ALIAS, mv.version)
    return str(mv.version)


def promote_anomaly() -> str:
    reported = json.loads((RESULTS / "anomaly_synthetic.json").read_text(encoding="utf-8"))[ANOMALY_KEY]
    train = load_history(TRAIN_SEED)
    det = RobustZScore(("temp", "cur_on_360", "off_run"))
    det.fit(train.feats)

    tests = [load_history(s) for s in TEST_SEEDS]
    m = evaluate_on_test(det, tests, reported["threshold"], reported["min_run"])
    print("anomaly detector:")
    _gate("test recall", m.recall, reported["recall"], 1e-6)
    _gate("test precision", m.precision, reported["precision"], 1e-6)

    config = {
        "type": "robust_z_score",
        "signals": list(det.signals),
        "stats": {k: list(v) for k, v in det.stats.items()},  # [median, 1.4826*MAD]
        "threshold": reported["threshold"],
        "min_run": reported["min_run"],
        "min_history_min": ANOMALY_MIN_HISTORY,
        "data": "SYNTHETIC",
        "fit_seed": TRAIN_SEED,
        "reported_test": {k: reported[k] for k in ("recall", "precision", "f1", "median_delay_by_type_min")},
    }
    with start_run("registry", "promote anomaly") as run, tempfile.TemporaryDirectory() as d:
        (Path(d) / "detector.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(Path(d) / "detector.json"), "detector")
        mlflow.log_metrics({"test_recall": m.recall, "test_precision": m.precision, "test_f1": m.f1})
        version = _register(run.info.run_id, "detector", ANOMALY_MODEL)
    print(f"  registered {ANOMALY_MODEL} v{version} @{ALIAS}")
    return version


def promote_breach() -> str:
    n_hist = CURVE[-1]
    key = f"lightgbm ({n_hist} training histories)"
    reported = json.loads((RESULTS / "breach_synthetic.json").read_text(encoding="utf-8"))["results"][key]

    train_sets = [load(s) for s in (TRAIN_SEED, *EXTRA_TRAIN_SEEDS)][:n_hist]
    det = LGBMBreach(n_hist, seed=0)
    det.fit(train_sets)

    pooled, ys, ss = BreachMetrics(), [], []
    for s in TEST_SEEDS:
        t = load(s)
        score = det.score(t.frame)
        pooled.merge(
            evaluate_breach(
                t.h.readings, t.h.events, score, t.at_risk, reported["threshold"], reported["min_run"]
            )
        )
        ys.append(t.frame.loc[t.eligible, "y"].to_numpy())
        ss.append(score[t.eligible])
    pr = float(average_precision_score(np.concatenate(ys), np.concatenate(ss)))
    print("breach model:")
    _gate("test PR-AUC", pr, reported["pr_auc"], 0.01)
    _gate("warned >=5 min", pooled.warned_frac(5.0), reported["warned_5min"], 0.02)

    config = {
        "type": "lightgbm_breach",
        "features": BREACH_FEATURES,
        "threshold": reported["threshold"],
        "min_run": reported["min_run"],
        "horizon_min": HORIZON,
        "breach_c": BREACH_C,
        "min_history_min": BREACH_MIN_HISTORY,
        "data": "SYNTHETIC",
        "train_seeds": [TRAIN_SEED, *EXTRA_TRAIN_SEEDS][:n_hist],
        "val_seed": VAL_SEED,
        "reported_test": {
            k: reported[k]
            for k in ("pr_auc", "warned_5min", "warned_10min", "precision", "false_alarms_per_zone_day")
        },
    }
    with start_run("registry", "promote breach") as run:
        mlflow.lightgbm.log_model(det.model, "model")
        mlflow.log_dict(config, "model/serving_config.json")
        mlflow.log_metrics({"test_pr_auc": pr, "test_warned_5min": pooled.warned_frac(5.0)})
        version = _register(run.info.run_id, "model", BREACH_MODEL)
    print(f"  registered {BREACH_MODEL} v{version} @{ALIAS}")
    return version


def main() -> None:
    promote_anomaly()
    promote_breach()


if __name__ == "__main__":
    main()
