"""Export the fine-tuned DistilBERT to ONNX, quantise to int8, and benchmark all variants."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from coldchain_ml.nlp.bert import ID2LABEL, MAX_LEN
from coldchain_ml.nlp.corpus import CLASSES
from coldchain_ml.nlp.metrics import report
from coldchain_ml.nlp.run import MODELS, _save, datasets


class _Logits(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(model, tok, path: Path) -> None:
    model.eval()
    enc = tok(["pallet crushed at dock 3", "freezer temperature alarm"], padding=True, return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"])
    kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
    )
    try:
        torch.onnx.export(_Logits(model), args, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` flag
        torch.onnx.export(_Logits(model), args, str(path), **kwargs)


def ort_session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_predict(sess: ort.InferenceSession, tok, texts: list[str], batch_size: int = 64) -> list[str]:
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i : i + batch_size], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np"
        )
        logits = sess.run(
            ["logits"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        out += [ID2LABEL[int(j)] for j in logits.argmax(-1)]
    return out


def _timeit(fn, n: int, warmup: int = 20) -> list[float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q))


ROUNDS, SINGLES_PER_ROUND, BATCH_REPS = 5, 40, 3


def bench_latency(model, tok, fp32: Path, int8: Path, pool: list[str], cpus: int) -> dict:
    """Interleaved latency benchmark. The three variants take turns over several rounds (rotating who
    goes first), so background load on a shared laptop hits each of them equally. Reports pooled
    percentiles, the spread of per-round medians, and the median per-round speed-up ratio, which is far
    less sensitive to drift than absolute milliseconds."""
    rng = np.random.default_rng(0)
    sample = [pool[i] for i in rng.integers(0, len(pool), 400)]
    np_in = [tok([t], truncation=True, max_length=MAX_LEN, return_tensors="np") for t in sample]
    pt_in = [tok([t], truncation=True, max_length=MAX_LEN, return_tensors="pt") for t in sample]
    batch_np = tok(pool[:32], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
    names = ["pytorch fp32", "onnx fp32", "onnx int8"]
    out: dict = {n: {} for n in names}
    out["speedups_median_of_rounds"] = {}

    for threads in (1, cpus):
        label = f"{threads}_thread{'s' if threads > 1 else ''}"
        torch.set_num_threads(threads)
        sess = {"onnx fp32": ort_session(fp32, threads), "onnx int8": ort_session(int8, threads)}

        def run_one(name: str, k: int, sess=sess) -> None:
            if name == "pytorch fp32":
                with torch.no_grad():
                    model(**pt_in[k])
            else:
                sess[name].run(
                    ["logits"],
                    {
                        "input_ids": np_in[k]["input_ids"].astype(np.int64),
                        "attention_mask": np_in[k]["attention_mask"].astype(np.int64),
                    },
                )

        feed = {k: batch_np[k].astype(np.int64) for k in ("input_ids", "attention_mask")}

        def run_batch(name: str, sess=sess, feed=feed) -> None:
            if name == "pytorch fp32":
                with torch.no_grad():
                    model(**{k: torch.from_numpy(v) for k, v in feed.items()})
            else:
                sess[name].run(["logits"], feed)

        samples = {n: [] for n in names}
        round_p50 = {n: [] for n in names}
        round_tput = {n: [] for n in names}
        cursor = 0
        for r in range(ROUNDS):
            for name in names[r % 3 :] + names[: r % 3]:
                ts: list[float] = []
                for j in range(SINGLES_PER_ROUND + 3):  # the first 3 calls are warm-up
                    k = (cursor + j) % len(np_in)
                    t0 = time.perf_counter()
                    run_one(name, k)
                    if j >= 3:
                        ts.append((time.perf_counter() - t0) * 1000)
                cursor += SINGLES_PER_ROUND
                samples[name] += ts
                round_p50[name].append(statistics.median(ts))
                run_batch(name)
                bt = []
                for _ in range(BATCH_REPS):
                    t0 = time.perf_counter()
                    run_batch(name)
                    bt.append(time.perf_counter() - t0)
                round_tput[name].append(32 / statistics.mean(bt))

        def speedup(slow: str, fast: str, round_p50=round_p50) -> float:
            return float(statistics.median(round_p50[slow][i] / round_p50[fast][i] for i in range(ROUNDS)))

        for n in names:
            out[n][label] = {
                "single_p50_ms": _pct(samples[n], 50),
                "single_p95_ms": _pct(samples[n], 95),
                "round_p50_min_ms": min(round_p50[n]),
                "round_p50_max_ms": max(round_p50[n]),
                "batch32_reports_per_s": float(statistics.median(round_tput[n])),
            }
            r_ = out[n][label]
            print(
                f"{n:13s} {threads:2d} thread(s): single p50 {r_['single_p50_ms']:6.1f} ms "
                f"(rounds {r_['round_p50_min_ms']:.0f}-{r_['round_p50_max_ms']:.0f}) p95 {r_['single_p95_ms']:6.1f} ms "
                f"| batch32 {r_['batch32_reports_per_s']:6.1f} reports/s",
                flush=True,
            )
        out["speedups_median_of_rounds"][label] = {
            "onnx_fp32_vs_pytorch": speedup("pytorch fp32", "onnx fp32"),
            "int8_vs_pytorch": speedup("pytorch fp32", "onnx int8"),
            "int8_vs_onnx_fp32": speedup("onnx fp32", "onnx int8"),
        }
        print("  speed-ups", label, {k: round(v, 2) for k, v in out["speedups_median_of_rounds"][label].items()}, flush=True)
    return out


def stage_export(seed: int, accuracy_only: bool = False, latency_only: bool = False) -> None:
    src = MODELS / f"distilbert-seed{seed}"
    out_dir = MODELS / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out_dir / "model_fp32.onnx", out_dir / "model_int8.onnx"

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, use_safetensors=True).eval()
    if latency_only:  # reuse the ONNX files exported earlier
        from coldchain_ml.nlp.run import _load

        parts, _, _ = datasets()
        cpus = os.cpu_count() or 1
        latency = bench_latency(model, tok, fp32, int8, parts["test"]["text"].tolist(), cpus)
        prev = _load().get("onnx_benchmark", {})
        _save(
            "onnx_benchmark",
            {
                **prev,
                "machine": f"{platform.processor()} | {cpus} logical CPUs | torch {torch.__version__} | onnxruntime {ort.__version__}",
                "latency": latency,
                "latency_method": f"{ROUNDS} interleaved rounds x {SINGLES_PER_ROUND} single-report calls per variant, model only (pre-tokenised)",
            },
        )
        return
    export_onnx(model, tok, fp32)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    tok.save_pretrained(out_dir)
    (out_dir / "labels.json").write_text(json.dumps(list(CLASSES)), encoding="utf-8")
    sizes = {
        "torch_safetensors_mb": os.path.getsize(src / "model.safetensors") / 2**20,
        "onnx_fp32_mb": os.path.getsize(fp32) / 2**20,
        "onnx_int8_mb": os.path.getsize(int8) / 2**20,
    }

    parts, ind_t, ind_l = datasets()
    sets = {
        "test": (parts["test"]["text"].tolist(), parts["test"]["label"].tolist()),
        "independent": (ind_t, ind_l),
    }

    def torch_predict(texts: list[str]) -> list[str]:
        from coldchain_ml.nlp import bert

        return bert.predict(model, tok, texts)

    cpus = os.cpu_count() or 1
    variants = {
        "pytorch fp32": torch_predict,
        "onnx fp32": lambda t: ort_predict(ort_session(fp32, cpus), tok, t),
        # serving scores one report at a time: dynamic int8 results depend on batch composition
        "onnx int8": lambda t: ort_predict(ort_session(int8, cpus), tok, t, batch_size=1),
        "onnx int8 (batches of 64)": lambda t: ort_predict(ort_session(int8, cpus), tok, t, batch_size=64),
    }

    # accuracy of every variant, and agreement with the PyTorch fp32 predictions
    accuracy: dict[str, dict] = {}
    base_preds = {k: torch_predict(x) for k, (x, _) in sets.items()}
    for name, fn in variants.items():
        accuracy[name] = {}
        for k, (x, y) in sets.items():
            p = base_preds[k] if name == "pytorch fp32" else fn(x)
            r = report(y, p)
            accuracy[name][k] = {
                "macro_f1": r["macro_f1"],
                "accuracy": r["accuracy"],
                "agreement_with_pytorch": float(
                    np.mean([a == b for a, b in zip(p, base_preds[k], strict=True)])
                ),
            }
            print(
                f"{name:13s} {k:11s} macro-F1 {r['macro_f1']:.4f} agreement {accuracy[name][k]['agreement_with_pytorch']:.4f}",
                flush=True,
            )

    if accuracy_only:  # keep the latency/size figures from the earlier quiet-machine run
        from coldchain_ml.nlp.run import _load

        prev = _load().get("onnx_benchmark", {})
        _save("onnx_benchmark", {**prev, "accuracy": accuracy})
        return

    latency = bench_latency(model, tok, fp32, int8, parts["test"]["text"].tolist(), cpus)

    _save(
        "onnx_benchmark",
        {
            "machine": f"{platform.processor()} | {cpus} logical CPUs | torch {torch.__version__} | onnxruntime {ort.__version__}",
            "sizes_mb": sizes,
            "accuracy": accuracy,
            "latency": latency,
            "latency_method": f"{ROUNDS} interleaved rounds x {SINGLES_PER_ROUND} single-report calls per variant, model only (pre-tokenised)",
            "source_model": str(src),
        },
    )
