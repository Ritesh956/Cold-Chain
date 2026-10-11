"""Classification metrics with honest uncertainty."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from coldchain_ml.nlp.corpus import CLASSES


def report(y_true: list[str], y_pred: list[str]) -> dict:
    labels = list(CLASSES)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(labels)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def bootstrap_macro_f1(
    y_true: list[str], y_pred: list[str], n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval for macro-F1 (resampling reports, not classes)."""
    rng = np.random.default_rng(seed)
    yt, yp = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        scores.append(f1_score(yt[idx], yp[idx], labels=list(CLASSES), average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
