"""Training used the Hugging Face tokenizer; serving uses the `tokenizers` library on the same
tokenizer.json. They must agree. Runs only where the opt-in PyTorch group is installed and the
serving artefacts exist (they are not in the repository)."""

from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
SERVING = Path("models/nlp/serving")
pytestmark = pytest.mark.skipif(
    not (SERVING / "tokenizer.json").exists(), reason="serving artefacts not built"
)


def test_serving_tokenizer_matches_training_tokenizer():
    from coldchain_ml.nlp.corpus import build_corpus
    from coldchain_ml.nlp.independent import INDEPENDENT
    from tokenizers import Tokenizer

    hf = transformers.AutoTokenizer.from_pretrained(SERVING)
    tk = Tokenizer.from_file(str(SERVING / "tokenizer.json"))
    tk.enable_truncation(64)
    texts = build_corpus()["text"].sample(300, random_state=0).tolist() + [t for _, t in INDEPENDENT]
    mismatches = [t for t in texts if hf(t, truncation=True, max_length=64)["input_ids"] != tk.encode(t).ids]
    assert not mismatches, mismatches[:3]
