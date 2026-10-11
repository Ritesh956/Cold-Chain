"""Incident-report classifier served with ONNX Runtime (no PyTorch in the service image)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class IncidentModel:
    name = "coldchain-incident"

    def __init__(self, session: Any, tokenizer: Any, config: dict[str, Any], version: str) -> None:
        self.session, self.tokenizer, self.config, self.version = session, tokenizer, config, version
        self.labels: list[str] = config["labels"]
        self.review_below: float = config["review_below"]
        self.max_chars: int = config.get("max_chars", 2000)

    @classmethod
    def from_dir(cls, path: Path, version: str) -> IncidentModel:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        config = json.loads((path / "serving_config.json").read_text(encoding="utf-8"))
        tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        tok.enable_truncation(max_length=config["max_len"])
        tok.enable_padding(pad_id=config["pad_id"], pad_token=config["pad_token"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = config.get("threads", 1)
        sess = ort.InferenceSession(str(path / config["onnx_file"]), so, providers=["CPUExecutionProvider"])
        return cls(sess, tok, config, version)

    def classify(self, texts: list[str]) -> list[dict[str, Any]]:
        """Score each report ON ITS OWN. Dynamic int8 quantisation derives activation scales from the
        whole batch, so batching changes predictions (measured: ~2.7% of reports flip between a batch
        of 1 and a batch of 64). One-by-one inference makes the answer independent of the request."""
        return [self._classify_one(t) for t in texts]

    def _classify_one(self, text: str) -> dict[str, Any]:
        enc = self.tokenizer.encode_batch([text])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        logits = self.session.run(["logits"], feed)[0][0]
        z = np.exp(logits - logits.max())
        p = z / z.sum()
        top = int(p.argmax())
        return {
            "label": self.labels[top],
            "confidence": float(p[top]),
            "scores": {lab: float(p[i]) for i, lab in enumerate(self.labels)},
            "needs_review": bool(p[top] < self.review_below),
        }
