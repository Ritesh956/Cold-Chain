"""Paired comparison of the baseline and the fine-tuned model, plus per-family error analysis.

uv run python -m coldchain_ml.nlp.compare --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from coldchain_ml.nlp import bert
from coldchain_ml.nlp.baseline import fit_baseline
from coldchain_ml.nlp.corpus import CLASSES
from coldchain_ml.nlp.run import MODELS, _save, datasets


def paired(y: list[str], a: list[str], b: list[str], n: int = 2000, seed: int = 0) -> dict:
    """A = baseline, B = model. Exact McNemar test and a paired bootstrap of macro-F1(B) - macro-F1(A)."""
    y_, a_, b_ = np.array(y), np.array(a), np.array(b)
    a_ok, b_ok = a_ == y_, b_ == y_
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y_), len(y_))
        fa = f1_score(y_[i], a_[i], labels=list(CLASSES), average="macro", zero_division=0)
        fb = f1_score(y_[i], b_[i], labels=list(CLASSES), average="macro", zero_division=0)
        diffs.append(fb - fa)
    return {
        "baseline_only_correct": only_a,
        "model_only_correct": only_b,
        "mcnemar_exact_p": float(p),
        "macro_f1_diff_model_minus_baseline": float(np.mean(diffs)),
        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed

    parts, ind_t, ind_l = datasets()
    tr, va, te = parts["train"], parts["val"], parts["test"]
    pipe, _, _ = fit_baseline(tr.text.tolist(), tr.label.tolist(), va.text.tolist(), va.label.tolist())

    src = MODELS / f"distilbert-seed{seed}"
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, use_safetensors=True).eval()

    out: dict = {}
    for name, texts, labels in (("test", te.text.tolist(), te.label.tolist()), ("independent", ind_t, ind_l)):
        a, b = list(pipe.predict(texts)), bert.predict(model, tok, texts)
        out[name] = paired(labels, a, b)
        r = out[name]
        print(
            f"{name:12s} baseline-only-right {r['baseline_only_correct']:3d} | model-only-right {r['model_only_correct']:3d} "
            f"| McNemar p {r['mcnemar_exact_p']:.4f} | macro-F1 diff {r['macro_f1_diff_model_minus_baseline']:+.3f} "
            f"(95% {r['diff_ci95'][0]:+.3f} to {r['diff_ci95'][1]:+.3f})"
        )
        if name == "test":
            fam = te.assign(a=a, b=b)
            rows = {}
            for f, g in fam.groupby("family"):
                rows[f] = {
                    "baseline_acc": float((g.a == g.label).mean()),
                    "model_acc": float((g.b == g.label).mean()),
                    "n": len(g),
                }
            out["test_by_family"] = rows
    print("\nper held-out test family (accuracy)       baseline   distilbert")
    for f, r in sorted(out["test_by_family"].items()):
        print(f"  {f:28s} {r['baseline_acc']:8.2f} {r['model_acc']:10.2f}")
    _save(f"paired_comparison_seed{seed}", out)


if __name__ == "__main__":
    main()
