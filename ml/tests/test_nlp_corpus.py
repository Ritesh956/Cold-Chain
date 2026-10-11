from collections import Counter

import pytest
from coldchain_ml.nlp.corpus import CLASSES, FAMILIES, build_corpus, family_split
from coldchain_ml.nlp.independent import INDEPENDENT
from coldchain_ml.nlp.metrics import bootstrap_macro_f1, report


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


def test_generation_is_deterministic(corpus):
    again = build_corpus()
    assert corpus.equals(again)
    assert not corpus.equals(build_corpus(seed=1))


def test_every_family_belongs_to_exactly_one_split_and_splits_are_family_disjoint(corpus):
    split = family_split()
    assert set(split) == {f.name for f in FAMILIES}
    fams = {s: set(corpus[corpus["split"] == s]["family"]) for s in ("train", "val", "test")}
    assert fams["train"].isdisjoint(fams["val"]) and fams["train"].isdisjoint(fams["test"])
    assert fams["val"].isdisjoint(fams["test"])


def test_split_sizes_are_balanced_per_class_and_family_counts_are_5_1_2():
    split = family_split()
    for label in CLASSES:
        counts = Counter(split[f.name] for f in FAMILIES if f.label == label)
        assert counts == {"train": 5, "val": 1, "test": 2}


def test_no_text_appears_twice_anywhere(corpus):
    assert corpus["text"].is_unique


def test_classes_are_balanced_within_each_split(corpus):
    for _, g in corpus.groupby("split"):
        assert g["label"].value_counts().nunique() == 1


def test_labels_match_family_class(corpus):
    cls = {f.name: f.label for f in FAMILIES}
    assert (corpus["family"].map(cls) == corpus["label"]).all()


def test_independent_set_is_balanced_and_does_not_overlap_the_corpus(corpus):
    assert Counter(label for label, _ in INDEPENDENT) == {c: 30 for c in CLASSES}
    texts = [t for _, t in INDEPENDENT]
    assert len(set(texts)) == len(texts)
    assert not set(t.lower() for t in texts) & set(corpus["text"].str.lower())


def test_reports_contain_realistic_noise(corpus):
    # typos/shorthand/case changes mean the same family produces many surface forms
    assert corpus["text"].str.islower().mean() > 0.1
    assert corpus["text"].str.len().between(15, 250).mean() > 0.97


def test_report_and_bootstrap_interval():
    y = ["damage", "delay", "safety", "temperature"] * 25
    r = report(y, y)
    assert r["accuracy"] == r["macro_f1"] == 1.0 and r["confusion"][0][0] == 25
    wrong = ["damage"] * 100
    assert report(y, wrong)["macro_f1"] < 0.25
    lo, hi = bootstrap_macro_f1(y, y, n=100)
    assert lo == hi == 1.0
    lo, hi = bootstrap_macro_f1(y, y[:50] + wrong[:50], n=200)
    assert lo < hi
