"""Fine-tune DistilBERT for incident classification (CPU-friendly; the same script runs on a Colab GPU)."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from coldchain_ml.nlp.corpus import CLASSES

BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 64
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 4
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def encode(tok, texts: list[str]) -> dict[str, torch.Tensor]:
    return tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


@torch.no_grad()
def predict(model, tok, texts: list[str], batch_size: int = 64) -> list[str]:
    model.eval()
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        enc = encode(tok, texts[i : i + batch_size])
        logits = model(**enc).logits
        out += [ID2LABEL[int(j)] for j in logits.argmax(-1)]
    return out


def train(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    cfg: TrainConfig,
    log=print,
):
    """Returns (model with the best-validation weights, tokenizer, history). Test data is not seen here."""
    seed_everything(cfg.seed)
    tok = load_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(CLASSES), id2label=ID2LABEL, label2id=LABEL2ID, use_safetensors=True
    )
    y = torch.tensor([LABEL2ID[t] for t in train_labels])
    n = len(train_texts)
    steps_per_epoch = (n + cfg.batch_size - 1) // cfg.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(
        opt, int(cfg.warmup_frac * steps_per_epoch * cfg.epochs), steps_per_epoch * cfg.epochs
    )
    gen = torch.Generator().manual_seed(cfg.seed)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(n, generator=gen).tolist()
        t0, total = time.perf_counter(), 0.0
        for s in range(steps_per_epoch):
            idx = order[s * cfg.batch_size : (s + 1) * cfg.batch_size]
            enc = encode(tok, [train_texts[i] for i in idx])
            loss = model(**enc, labels=y[idx]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += float(loss)
        preds = predict(model, tok, val_texts)
        f1 = float(f1_score(val_labels, preds, labels=list(CLASSES), average="macro"))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / steps_per_epoch,
                "val_macro_f1": f1,
                "seconds": time.perf_counter() - t0,
            }
        )
        log(
            f"epoch {epoch}: train loss {total / steps_per_epoch:.4f} | val macro-F1 {f1:.4f} | "
            f"{time.perf_counter() - t0:.0f}s"
        )
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, tok, history


def save(model, tok, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
