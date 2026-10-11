"""Package the int8 ONNX incident classifier, verify it, and register it as `coldchain-incident`.

Gate: the SERVING code path (onnxruntime + the `tokenizers` library, no PyTorch) must reproduce the
macro-F1 recorded when the model was benchmarked with the training-side tokenizer. That catches any
difference between the two tokenizers before the model reaches the registry.

Also chooses the `needs_review` threshold on the VALIDATION set and measures, on held-out data,
whether low confidence really flags errors.

Run:  uv run python -m coldchain_ml.registry.promote_incident
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

from coldchain_ml.common.tracking import start_run
from coldchain_ml.nlp.bert import MAX_LEN
from coldchain_ml.nlp.corpus import CLASSES
from coldchain_ml.nlp.metrics import report
from coldchain_ml.nlp.run import MODELS, RESULTS, datasets
from coldchain_ml.nlp.serving import IncidentModel
from coldchain_ml.registry.promote import ALIAS, GateFailed, _gate

NAME = "coldchain-incident"
REVIEW_FRACTION = 0.20  # on validation, flag the 20% least confident reports
OUT = Path("docs/results/incident_serving.json")


def build_package(src: Path, dst: Path, config: dict) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copy(src / "model_int8.onnx", dst / "model_int8.onnx")
    shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")
    (dst / "serving_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return dst


def selective(model: IncidentModel, texts: list[str], labels: list[str]) -> dict:
    res = model.classify(texts)
    pred = [r["label"] for r in res]
    conf = np.array([r["confidence"] for r in res])
    wrong = np.array([p != y for p, y in zip(pred, labels, strict=True)])
    flagged = np.array([r["needs_review"] for r in res])
    out = {
        "macro_f1": report(labels, pred)["macro_f1"],
        "accuracy": float(1 - wrong.mean()),
        "flagged_fraction": float(flagged.mean()),
        "accuracy_if_auto_accepted": float(1 - wrong[~flagged].mean()) if (~flagged).any() else None,
        "accuracy_if_flagged": float(1 - wrong[flagged].mean()) if flagged.any() else None,
        "share_of_errors_flagged": float(flagged[wrong].mean()) if wrong.any() else None,
    }
    if wrong.any() and not wrong.all():
        out["auroc_low_confidence_predicts_error"] = float(roc_auc_score(wrong, 1 - conf))
    return out


def main() -> None:
    src = MODELS / "serving"
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    bench = results["onnx_benchmark"]["accuracy"]["onnx int8"]
    parts, ind_t, ind_l = datasets()
    sets = {
        "val": (parts["val"]["text"].tolist(), parts["val"]["label"].tolist()),
        "test": (parts["test"]["text"].tolist(), parts["test"]["label"].tolist()),
        "independent": (ind_t, ind_l),
    }

    config = {
        "type": "distilbert_onnx_int8",
        "labels": list(CLASSES),
        "onnx_file": "model_int8.onnx",
        "max_len": MAX_LEN,
        "pad_id": 0,
        "pad_token": "[PAD]",
        "threads": 1,
        "max_chars": 2000,
        "review_below": 0.0,  # provisional; chosen on validation below
        "data": "SYNTHETIC",
        "source": f"distilbert-base-uncased fine-tuned ({results['onnx_benchmark']['source_model']}), "
        "dynamic int8 quantisation, scored one report at a time",
    }
    pkg = build_package(src, MODELS / "package", config)

    # choose the review threshold on VALIDATION only
    provisional = IncidentModel.from_dir(pkg, "provisional")
    val_conf = np.array([r["confidence"] for r in provisional.classify(sets["val"][0])])
    config["review_below"] = float(np.quantile(val_conf, REVIEW_FRACTION))

    # the serving path with the final config, on all three sets
    (pkg / "serving_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = IncidentModel.from_dir(pkg, "candidate")
    measured = {k: selective(model, *v) for k, v in sets.items()}

    print("incident model:")
    _gate("test macro-F1 (serving tokenizer)", measured["test"]["macro_f1"], bench["test"]["macro_f1"], 1e-6)
    _gate(
        "independent macro-F1 (serving tokenizer)",
        measured["independent"]["macro_f1"],
        bench["independent"]["macro_f1"],
        1e-6,
    )

    config["reported_test"] = {
        "test_macro_f1": measured["test"]["macro_f1"],
        "independent_macro_f1": measured["independent"]["macro_f1"],
        "baseline_tfidf_test_macro_f1": results["tfidf_logreg"]["test"]["macro_f1"],
    }
    (pkg / "serving_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    OUT.write_text(json.dumps({"config": config, "measured": measured}, indent=2), encoding="utf-8")

    for k, m in measured.items():
        print(
            f"  {k:11s} F1 {m['macro_f1']:.3f} | flagged {m['flagged_fraction']:.2f} | acc auto {m['accuracy_if_auto_accepted']:.3f} "
            f"vs flagged {m['accuracy_if_flagged']:.3f} | errors caught {m['share_of_errors_flagged']:.2f} "
            f"| AUROC {m.get('auroc_low_confidence_predicts_error', float('nan')):.3f}"
        )

    with start_run("registry", "promote incident") as run:
        mlflow.log_artifacts(str(pkg), "model")
        mlflow.log_metrics(
            {
                "test_macro_f1": measured["test"]["macro_f1"],
                "independent_macro_f1": measured["independent"]["macro_f1"],
            }
        )
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", NAME)
        MlflowClient().set_registered_model_alias(NAME, ALIAS, str(mv.version))
    print(f"  registered {NAME} v{mv.version} @{ALIAS}")


if __name__ == "__main__":
    try:
        main()
    except GateFailed as e:
        raise SystemExit(f"NOT REGISTERED: {e}") from e
