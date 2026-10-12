"""Does the incident classifier recognise REAL injury text as `safety`?  (one-sided real-text check)

Input: the OSHA Severe Injury Reports file (public domain; every row is a hospitalisation, amputation or
loss of an eye). Every narrative is a workplace injury, so a good classifier should call (nearly) all of them
`safety`. That measures recall on one class, on industrial rather than cold-store text. It says nothing about
precision, nor about the other three classes, because no real text for them is used.

    # data/real/osha/severeinjury.csv comes from the full SIR data set at osha.gov/severe-injury-reports
    uv run python scripts/osha_check.py
Writes docs/results/incident_osha.{json,md}. Uses the model currently registered as @champion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from coldchain_ml.common.stats import wilson_interval
from coldchain_ml.nlp.baseline import fit_baseline
from coldchain_ml.nlp.corpus import build_corpus
from coldchain_ml_api.config import Settings
from coldchain_ml_api.store import Models

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "real" / "osha" / "severeinjury.csv"
OUT = ROOT / "docs" / "results"
N_SAMPLE = 3000
COLD_STORAGE_NAICS = "493120"  # refrigerated warehousing and storage (stored as text in the file)
SEED = 0


def classify(model, texts: list[str]) -> pd.DataFrame:
    rows = model.classify([t[: model.max_chars] for t in texts])
    return pd.DataFrame(rows)[["label", "confidence", "needs_review"]]


def summarise(df: pd.DataFrame) -> dict:
    n = len(df)
    safety = int((df["label"] == "safety").sum())
    lo, hi = wilson_interval(safety, n)
    reviewed = int(df["needs_review"].sum())
    # of the reports the classifier would NOT send to a human, how many are right?
    confident = df[~df["needs_review"]]
    conf_safety = int((confident["label"] == "safety").sum())
    return {
        "n": n,
        "safety_recall": safety / n,
        "safety_recall_ci95": [lo, hi],
        "label_counts": df["label"].value_counts().to_dict(),
        "needs_review_share": reviewed / n,
        "confident_n": len(confident),
        "confident_safety_share": conf_safety / len(confident) if len(confident) else None,
        "median_confidence_by_label": df.groupby("label")["confidence"].median().round(3).to_dict(),
    }


def main() -> None:
    raw = pd.read_csv(CSV, encoding="latin-1", low_memory=False)
    raw = raw.dropna(subset=["Final Narrative"])
    raw["Final Narrative"] = raw["Final Narrative"].astype(str).str.strip()
    raw = raw[raw["Final Narrative"].str.len() >= 20]  # a few rows are fragments, not reports

    models = Models.load(Settings.from_env())
    model = models.incident
    if model is None:
        raise SystemExit("no incident model registered: run coldchain_ml.registry.promote_incident")

    sample = raw.sample(min(N_SAMPLE, len(raw)), random_state=SEED)
    cold = raw[raw["Primary NAICS"].astype(str) == COLD_STORAGE_NAICS]
    results = {"model_version": model.version, "source_rows": len(raw), "seed": SEED}

    s = classify(model, sample["Final Narrative"].tolist())
    results["random_sample"] = summarise(s)
    c = classify(model, cold["Final Narrative"].tolist())
    results["refrigerated_warehousing_naics_493120"] = summarise(c)

    # the TF-IDF + logistic regression baseline, fitted exactly as in the model card, on the same texts
    corpus = build_corpus()
    tr, va = corpus[corpus["split"] == "train"], corpus[corpus["split"] == "val"]
    pipe, c_chosen, _ = fit_baseline(tr.text.tolist(), tr.label.tolist(), va.text.tolist(), va.label.tolist())

    def baseline_frame(texts: list[str]) -> pd.DataFrame:
        proba = pipe.predict_proba(texts)
        labels = pipe.classes_[proba.argmax(axis=1)]
        return pd.DataFrame({"label": labels, "confidence": proba.max(axis=1), "needs_review": False})

    results["baseline_random_sample"] = summarise(baseline_frame(sample["Final Narrative"].tolist()))
    results["baseline_cold_storage"] = summarise(baseline_frame(cold["Final Narrative"].tolist()))
    results["baseline_c"] = c_chosen

    # what does it call the reports it gets "wrong"? (a handful, shortened, for the write-up)
    sample = sample.reset_index(drop=True)
    miss = sample[s["label"] != "safety"].copy()
    miss["pred"] = s.loc[miss.index, "label"]
    miss["conf"] = s.loc[miss.index, "confidence"].round(2)
    shown = miss.sample(min(12, len(miss)), random_state=SEED)
    results["examples_not_safety"] = [
        {"pred": pred, "confidence": float(conf), "text": text[:220]}
        for text, pred, conf in zip(shown["Final Narrative"], shown["pred"], shown["conf"], strict=True)
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "incident_osha.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "incident_osha.md").write_text(to_markdown(results), encoding="utf-8")
    print(to_markdown(results))


def to_markdown(r: dict) -> str:
    def row(name: str, d: dict) -> str:
        lo, hi = d["safety_recall_ci95"]
        counts = ", ".join(f"{k} {v}" for k, v in sorted(d["label_counts"].items(), key=lambda kv: -kv[1]))
        recall = f"**{d['safety_recall']:.3f}** ({lo:.3f}-{hi:.3f})"
        return (
            f"| {name} | {d['n']:,} | {recall} | {d['needs_review_share']:.3f} "
            f"| {d['confident_safety_share']:.3f} | {counts} |"
        )

    lines = [
        "# Incident classifier on REAL injury text (OSHA Severe Injury Reports)",
        "",
        "Every OSHA report is a workplace injury (hospitalisation, amputation or loss of an eye), so the right "
        "label is `safety` for all of them. This is a **one-sided** check: recall on one class, on industrial "
        "text. It cannot measure precision or the other three classes (no real text for them), and cold-store "
        f"reports are a tiny part of it. Model: coldchain-incident v{r['model_version']} (trained on synthetic "
        f"text only). {r['source_rows']:,} reports with a usable narrative.",
        "",
        "| Subset | Reports | Labelled `safety` (95% CI) | Flagged needs-review "
        "| Safety share among confident | Labels |",
        "|---|---|---|---|---|---|",
        row("Random sample", r["random_sample"]),
        row("Refrigerated warehousing (NAICS 493120)", r["refrigerated_warehousing_naics_493120"]),
        row("Baseline (TF-IDF + logistic regression), random sample", r["baseline_random_sample"]),
        row("Baseline (TF-IDF + logistic regression), refrigerated warehousing", r["baseline_cold_storage"]),
        "",
        "The baseline has no review threshold, so its needs-review column is 0 by construction and its "
        "'confident' share equals its overall share.",
        "",
        "Median top-score by predicted label (a ranking signal, not a probability): "
        + ", ".join(f"{k} {v}" for k, v in r["random_sample"]["median_confidence_by_label"].items()),
        "",
        "## Reports it did not call safety (random sample of 12)",
        "",
    ]
    for e in r["examples_not_safety"]:
        lines.append(f"- predicted **{e['pred']}** ({e['confidence']}): {e['text']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
