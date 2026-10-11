# Incident classification (damage / temperature / delay / safety) — SYNTHETIC text

All text is generated or authored by the AI assistant that built the project; none is from a real warehouse. Scores measure fit to this generator and to a hand-written shift set, **not** real-world performance. Model card: [incident-nlp.md](../model-cards/incident-nlp.md).

Splits: 32 template families held out **whole** (5 train / 1 validation / 2 test per class), so validation and test contain sentence structures and phrasings never seen in training. 'Independent' is 120 reports (30 per class) written in a different style; it was used once, after all choices were made on validation.

## Macro-F1 (95% bootstrap interval over reports)

| Model | Validation (4 families) | Test (8 unseen families, n=960) | Independent (n=120) |
|---|---|---|---|
| TF-IDF + logistic regression | 0.577 (0.55-0.60) | 0.755 (0.72-0.78) | 0.799 (0.72-0.86) |
| DistilBERT, seed 0 | 0.585 (0.56-0.61) | 0.748 (0.72-0.77) | 0.723 (0.64-0.79) |
| DistilBERT, seed 1 ← served (best validation) | 0.619 (0.60-0.64) | 0.731 (0.70-0.76) | 0.795 (0.72-0.87) |
| DistilBERT, seed 2 | 0.614 (0.59-0.64) | 0.720 (0.69-0.75) | 0.858 (0.79-0.91) |
| **DistilBERT, mean ± sd over 3 seeds** | 0.606 ± 0.019 (range 0.585-0.619) | 0.733 ± 0.014 (range 0.720-0.748) | 0.792 ± 0.067 (range 0.723-0.858) |

The seed-to-seed spread on the independent set (sd about 0.07) is roughly ten times the gap between the seed mean and the baseline (0.792 vs 0.799), so a single training run cannot rank the two. Training loss falls to about 0.001 within 3 epochs: the model memorises the training templates.

## Paired comparison with the baseline (model minus baseline)

| Seed | Test: macro-F1 diff (95% CI), McNemar p | Independent: macro-F1 diff (95% CI), McNemar p |
|---|---|---|
| 0 | -0.007 (-0.034 to +0.019), p=0.93 | -0.076 (-0.172 to +0.023), p=0.27 |
| 1 | -0.023 (-0.051 to +0.005), p=0.24 | -0.002 (-0.090 to +0.087), p=1.00 |
| 2 | -0.034 (-0.063 to -0.002), p=0.14 | +0.061 (-0.022 to +0.147), p=0.33 |

## Accuracy per held-out test family (seed 1)

| Family | TF-IDF | DistilBERT |
|---|---|---|
| damage/crane | 0.85 | 1.00 |
| damage/telegraphic | 0.95 | 1.00 |
| delay/dock_queue | 0.42 | 0.27 |
| delay/telegraphic | 0.88 | 0.63 |
| safety/ppe | 0.93 | 0.92 |
| safety/slip | 0.93 | 0.99 |
| temperature/logger | 0.48 | 0.47 |
| temperature/telegraphic | 0.65 | 0.67 |

## ONNX export and int8 quantisation (seed 1) — Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 20 logical CPUs | torch 2.14.1+cpu | onnxruntime 1.31.0

| Variant | Test macro-F1 | Independent macro-F1 | Test: same label as PyTorch fp32 |
|---|---|---|---|
| pytorch fp32 | 0.731 | 0.795 | 100.0% |
| onnx fp32 | 0.731 | 0.795 | 100.0% |
| onnx int8 | 0.741 | 0.795 | 91.8% |
| onnx int8 (batches of 64) | 0.744 | 0.803 | 92.0% |

File size: PyTorch/ONNX fp32 256 MB, ONNX int8 64 MB (4.0x smaller).

**Dynamic int8 results depend on the batch.** The same report gets a different label depending on which other reports share its batch (about 2.7% flip between a batch of 1 and a batch of 64, logit differences up to 2.1); fp32 has none. The API therefore scores each report on its own, and the 'onnx int8' row above is measured that way. The 'batches of 64' row shows the batched numbers.

### Latency, single report (5 interleaved rounds x 40 single-report calls per variant, model only (pre-tokenised))

| Variant | 1 thread p50 / p95 (ms) | all threads p50 / p95 (ms) | batch-32 reports/s (1 thread / all) |
|---|---|---|---|
| pytorch fp32 | 82 / 113 | 22 / 36 | 12 / 72 |
| onnx fp32 | 60 / 85 | 18 / 65 | 12 / 84 |
| onnx int8 | 21 / 30 | 11 / 69 | 34 / 155 |

Median per-round speed-up ratios (interleaved rounds, robust to background load):

- 1_thread: ONNX fp32 is 1.4x PyTorch; int8 is 4.0x PyTorch and 2.9x ONNX fp32.
- 20_threads: ONNX fp32 is 1.3x PyTorch; int8 is 2.0x PyTorch and 1.5x ONNX fp32.

### What the baseline costs (same machine, 1 thread)

TF-IDF + logistic regression: 3.3 ms p50 per report (p95 4.1), 5992 reports/s in bulk, 0.6 MB.

## `needs_review` flag (served model)

Threshold 0.960, chosen on validation as the 20th percentile of confidence. Does low confidence actually flag errors?

| Set | Flagged | Accuracy if auto-accepted | Accuracy if flagged | Errors caught | AUROC (low confidence predicts error) |
|---|---|---|---|---|---|
| val | 20% | 0.677 | 0.667 | 21% | 0.45 |
| test | 24% | 0.851 | 0.430 | 55% | 0.84 |
| independent | 27% | 0.898 | 0.531 | 62% | 0.78 |
