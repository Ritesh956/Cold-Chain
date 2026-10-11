"""Incident-classification experiments on SYNTHETIC text.

    uv run python -m coldchain_ml.nlp.run baseline
    uv run python -m coldchain_ml.nlp.run bert --seed 0          # CPU: ~10 min; Colab GPU: ~1 min
    uv run python -m coldchain_ml.nlp.run export --seed 0        # ONNX + int8 + benchmarks

Selection (C, best epoch) uses the validation families only. Test (unseen families) and the
independent set are evaluated once the model is chosen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from coldchain_ml.common.tracking import start_run
from coldchain_ml.nlp.corpus import build_corpus
from coldchain_ml.nlp.independent import INDEPENDENT
from coldchain_ml.nlp.metrics import bootstrap_macro_f1, report

OUT = Path("docs/results")
RESULTS = OUT / "incident_nlp.json"
MODELS = Path("models/nlp")


def datasets():
    df = build_corpus()
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    ind_texts, ind_labels = [t for _, t in INDEPENDENT], [label for label, _ in INDEPENDENT]
    return parts, ind_texts, ind_labels


def evaluate_all(predict_fn, parts, ind_texts, ind_labels) -> dict:
    sets = {
        "val": (parts["val"]["text"].tolist(), parts["val"]["label"].tolist()),
        "test": (parts["test"]["text"].tolist(), parts["test"]["label"].tolist()),
        "independent": (ind_texts, ind_labels),
    }
    out = {}
    for name, (texts, labels) in sets.items():
        preds = predict_fn(texts)
        r = report(labels, preds)
        r["macro_f1_ci95"] = list(bootstrap_macro_f1(labels, preds))
        out[name] = r
    return out


def _load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def _save(key: str, value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = _load()
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=2), encoding="utf-8")


def stage_baseline_cost() -> None:
    """Latency, throughput and size of the TF-IDF baseline, on the same machine as the ONNX benchmark."""
    import pickle
    import time

    import numpy as np

    from coldchain_ml.nlp.baseline import fit_baseline

    parts, _, _ = datasets()
    tr, va, te = parts["train"], parts["val"], parts["test"]
    pipe, _, _ = fit_baseline(tr.text.tolist(), tr.label.tolist(), va.text.tolist(), va.label.tolist())
    pool = te.text.tolist()
    rng = np.random.default_rng(0)
    sample = [pool[i] for i in rng.integers(0, len(pool), 300)]
    for t in sample[:20]:
        pipe.predict([t])
    singles = []
    for t in sample:
        t0 = time.perf_counter()
        pipe.predict([t])
        singles.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    pipe.predict(pool)
    bulk = len(pool) / (time.perf_counter() - t0)
    _save(
        "baseline_cost",
        {
            "single_p50_ms": float(np.percentile(singles, 50)),
            "single_p95_ms": float(np.percentile(singles, 95)),
            "batch_reports_per_s": float(bulk),
            "pickled_mb": len(pickle.dumps(pipe)) / 2**20,
        },
    )


def stage_baseline() -> None:
    from coldchain_ml.nlp.baseline import fit_baseline

    parts, ind_t, ind_l = datasets()
    tr, va = parts["train"], parts["val"]
    pipe, c, grid = fit_baseline(tr.text.tolist(), tr.label.tolist(), va.text.tolist(), va.label.tolist())
    res = evaluate_all(lambda t: list(pipe.predict(t)), parts, ind_t, ind_l)
    res.update({"chosen_C": c, "val_macro_f1_by_C": grid})
    with start_run("incident-nlp", "tfidf + logistic regression"):
        mlflow.log_params({"model": "tfidf+logreg", "C": c, "data": "synthetic"})
        mlflow.log_metrics({f"{k}_macro_f1": res[k]["macro_f1"] for k in ("val", "test", "independent")})
    _save("tfidf_logreg", res)
    for k in ("val", "test", "independent"):
        print(f"tfidf  {k:11s} macro-F1 {res[k]['macro_f1']:.3f} {res[k]['macro_f1_ci95']}")


def stage_bert(seed: int, epochs: int) -> None:
    from coldchain_ml.nlp import bert

    parts, ind_t, ind_l = datasets()
    tr, va = parts["train"], parts["val"]
    cfg = bert.TrainConfig(seed=seed, epochs=epochs)
    with start_run("incident-nlp", f"distilbert seed {seed}"):
        model, tok, history = bert.train(
            tr.text.tolist(),
            tr.label.tolist(),
            va.text.tolist(),
            va.label.tolist(),
            cfg,
            log=lambda m: print(m, flush=True),
        )
        res = evaluate_all(lambda t: bert.predict(model, tok, t), parts, ind_t, ind_l)
        res.update({"history": history, "config": cfg.__dict__})
        mlflow.log_params({"model": "distilbert-base-uncased", **cfg.__dict__, "data": "synthetic"})
        mlflow.log_metrics({f"{k}_macro_f1": res[k]["macro_f1"] for k in ("val", "test", "independent")})
    bert.save(model, tok, MODELS / f"distilbert-seed{seed}")
    _save(f"distilbert_seed{seed}", res)
    for k in ("val", "test", "independent"):
        print(f"bert   {k:11s} macro-F1 {res[k]['macro_f1']:.3f} {res[k]['macro_f1_ci95']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline", "baseline-cost", "bert", "export"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--latency-only", action="store_true", help="export: only re-run the latency benchmark")
    ap.add_argument("--accuracy-only", action="store_true", help="export: skip latency, keep earlier timings")
    a = ap.parse_args()
    if a.stage == "baseline":
        stage_baseline()
    elif a.stage == "baseline-cost":
        stage_baseline_cost()
    elif a.stage == "bert":
        stage_bert(a.seed, a.epochs)
    else:
        from coldchain_ml.nlp.export import stage_export

        stage_export(a.seed, a.accuracy_only, a.latency_only)


if __name__ == "__main__":
    main()
