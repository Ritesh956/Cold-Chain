"""TF-IDF + logistic regression: the baseline every transformer number is compared against."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from coldchain_ml.nlp.corpus import CLASSES

C_GRID = (0.3, 1, 3, 10, 30)


def make_pipeline(c: float) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)),
        ]
    )
    return Pipeline([("tfidf", features), ("clf", LogisticRegression(C=c, max_iter=3000))])


def fit_baseline(
    train_texts: list[str], train_labels: list[str], val_texts: list[str], val_labels: list[str]
):
    """Choose C on the validation families only. Returns (fitted pipeline, chosen C, {C: val macro-F1})."""
    best, scores = None, {}
    for c in C_GRID:
        pipe = make_pipeline(c).fit(train_texts, train_labels)
        f1 = f1_score(val_labels, pipe.predict(val_texts), labels=list(CLASSES), average="macro")
        scores[c] = float(f1)
        if best is None or f1 > scores[best[1]]:
            best = (pipe, c)
    return best[0], best[1], scores
