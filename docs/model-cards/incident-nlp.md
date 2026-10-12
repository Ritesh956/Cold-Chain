# Model card: incident-report classifier

**Status:** trained and evaluated on **SYNTHETIC** text only. No real incident report has been used.
Every number below measures fit to my generator and to a hand-written shift set, not real-world
performance. Full tables (generated from the saved results): [results/incident_nlp.md](../results/incident_nlp.md).
Code: `ml/src/coldchain_ml/nlp/`. Served by the [ml-api](../ml-api.md) at `/v1/incidents/classify`.

## Task
Assign a free-text warehouse incident report to its **main issue**: `damage`, `temperature`, `delay`
or `safety`. Reports often mention a second issue ("pallet crushed. The room also felt warmer."); the
label is the main one.

## Data (all synthetic)
- **Generator** (`corpus.py`): 32 template *families*, 8 per class. Each family has its own sentence
  structures and its own pool of event phrases. Surface variation: locations, ids, products, actors,
  times; typos, shorthand ("plt", "FLT"), dropped punctuation, odd casing, filler; 25% of reports
  mention a secondary issue from another class; 40% carry class-independent context.
- **Split by family, not by report:** per class 5 families train, 1 validation, 2 test. The validation
  and test sets therefore contain sentence structures and phrasings never seen in training.
  Sizes: 3,200 train, 480 validation, 960 test (corpus hash `5c101ecf`, deterministic).
- **Independent set** (`independent.py`): 120 reports, 30 per class, in a narrative style that does
  not reuse the generator's templates. **Provenance: written by the AI assistant that built this
  project. Not collected from a warehouse, not written by the project owner. Still synthetic.** Its
  value is distribution shift. Used once, after the model was chosen on validation.
- A real-text check on OSHA injury narratives (safety class only) is done; see
  [Real-text check](#real-text-check-osha-safety-class-only).

## Models compared
| Model | Notes |
|---|---|
| TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression | Baseline. C chosen on validation. |
| DistilBERT (`distilbert-base-uncased`) fine-tuned | 4 epochs, lr 5e-5, batch 32, max length 64, best epoch by validation macro-F1. **3 seeds.** CPU, 17 to 30 minutes per run depending on machine load. |
| DistilBERT, ONNX fp32 and int8 (dynamic quantisation) | The served model is the validation-best seed (seed 1), int8. |

## Results (macro-F1; 95% bootstrap interval over reports)
| | Validation | Test (unseen families) | Independent |
|---|---|---|---|
| TF-IDF + logistic regression | 0.577 | **0.755** (0.72-0.78) | **0.799** (0.72-0.86) |
| DistilBERT, mean of 3 seeds | 0.606 | 0.733 ± 0.014 (0.720-0.748) | 0.792 ± 0.067 (0.723-0.858) |
| DistilBERT, served (seed 1, fp32) | 0.619 | 0.731 (0.70-0.76) | 0.795 (0.72-0.87) |
| DistilBERT, served (seed 1, int8, one report at a time) | 0.625 | 0.741 | 0.795 |

## What this shows
1. **No evidence that fine-tuned DistilBERT beats TF-IDF here.** On the unseen-template test set all
   three seeds are below the baseline (by 0.7, 2.3 and 3.4 macro-F1 points); only seed 2's bootstrap
   interval excludes zero and no McNemar test reaches p < 0.05. On the independent set the sign flips
   by seed (-0.08, -0.00, +0.06) and the seed spread (sd 0.07) is about ten times the gap between the
   mean and the baseline. One training run cannot rank the two; I trained three for that reason.
2. **The transformer memorises.** Training loss falls to about 0.001 by epoch 3 while validation stays
   flat. 3,200 reports from 20 training families are easy to memorise.
3. **Both models fail on the same families** (accuracy on `delay/dock_queue`: TF-IDF 0.42, DistilBERT
   0.27; on `temperature/logger`: 0.48 and 0.47). Part of this is my generator: the injected secondary clause ("the room also felt warmer")
   pulls the model to the wrong class, and some logger reports never mention temperature at all, so a
   person could not label them with confidence. The accuracy ceiling on this data is below 100%.
4. **Int8 quantisation: 4.0x faster, 4x smaller, not free.** Single report on one thread: PyTorch
   82 ms, ONNX fp32 60 ms, **int8 21 ms** (median per-round speed-up 4.0x vs PyTorch, 2.9x vs ONNX
   fp32; interleaved rounds, so robust to background load). With all 20 threads the gain is 2.0x and
   1.5x and the timings are noisy. File size 256 MB to 64 MB. int8 changes about 8% of labels versus
   fp32 (91.8% agreement on test); the macro-F1 difference (+0.010) is noise, not an improvement,
   because the changes cancel out.
5. **Dynamic int8 predictions depend on the batch.** Found when the registry gate refused the model:
   the same report got a different label depending on which others shared its batch (about 2.7%
   flipped between a batch of 1 and a batch of 64, logit differences up to 2.1). fp32 has no such
   dependence. The API therefore scores each report on its own (about 20 ms each), a regression test
   guards it, and all int8 numbers above are measured that way.
6. **The `needs_review` flag helps on average but is not a guarantee.** The threshold (0.96) is the
   20th percentile of validation confidence. On the test set it flags 24% of reports: auto-accepted
   ones are 85% accurate, flagged ones 43%, and it catches 55% of errors (AUROC 0.84). On the
   independent set 27% are flagged (90% vs 53% accurate, 62% of errors, AUROC 0.78). On the
   **validation** families it does nothing (AUROC 0.45): there the model is confidently wrong.
   Confidence is a softmax score, not a calibrated probability.

## Why the served model is DistilBERT int8 anyway
The baseline matches it on this data and is cheaper: 3.3 ms per report, about 6,000 reports/s in bulk,
0.55 MB, against 21 ms, 34 reports/s (one thread) and 64 MB. I chose to serve the transformer because
the project is meant to demonstrate fine-tuning, ONNX export, quantisation and serving it, and because
real incident text is less templated than my generator, which is where pretrained language knowledge
should matter. That was a bet when I wrote it. **The OSHA check below is the first evidence on it, and it favours
the transformer strongly:** on 3,000 real injury narratives the TF-IDF baseline calls only 34% `safety` against
the transformer's 89%. Real text is where the pretrained model earns its cost. (One class only; see the caveats
below. Rerun `nlp.compare` on real reports of all four classes before treating this as settled.)

## What went wrong along the way (kept in the record)
- **Generator flaw.** The first baseline misread a held-out family (`temperature/product`) as damage
  because pallet ids and products appeared mostly in damage families, so they became class
  shortcuts. Found using validation, not test; fixed by adding class-independent context to every
  class. Baseline scores changed from 0.69 / 0.78 (test / independent) to 0.755 / 0.80 after the change.
- **Seed noise nearly produced a wrong headline.** Seed 0 alone showed DistilBERT 0.08 below the baseline
  on the independent set; seeds 1 and 2 showed -0.00 and +0.06.
- **Benchmarks on a shared laptop are noisy.** A first latency run gave a PyTorch 1-thread median
  of 142 ms against 82 ms minutes earlier. The benchmark now interleaves variants over rounds and
  reports speed-up ratios.
- **Model selection.** I first exported seed 0 before seeing the others. My protocol selects by
  validation only, so I re-exported seed 1.

## Real-text check (OSHA, safety class only)
Full table: [results/incident_osha.md](../results/incident_osha.md); script `scripts/osha_check.py`. Every OSHA
severe-injury narrative is a workplace injury, so the right label is `safety` for all of them. The model never
saw real text. **One-sided:** this measures recall on one class on industrial text; it cannot measure precision
or the other three classes.

| Subset | Reports | Labelled `safety` (95% CI) | Flagged needs-review | Safety share among confident |
|---|---|---|---|---|
| Random sample | 3,000 | **0.887** (0.875-0.898) | 0.268 | 0.971 |
| Refrigerated warehousing (NAICS 493120) | 340 | **0.829** (0.786-0.866) | 0.359 | 0.894 |
| *TF-IDF baseline*, random sample | 3,000 | 0.336 (0.319-0.353) | n/a | n/a |
| *TF-IDF baseline*, refrigerated warehousing | 340 | 0.400 (0.349-0.453) | n/a | n/a |

- **The synthetic-text tie with the baseline does not hold on real text.** The same TF-IDF + logistic regression
  that matched the transformer on synthetic reports labels only 34% of real injury reports `safety` (it spreads
  them over temperature 31%, damage 25%, delay 10%): it learned my generator's topic words, not the concept.
  The transformer is far more robust, though not good enough.
- **The transformer's own real-text result is still weak.** About 11% of real injury narratives are
  labelled something else (delay 5%, temperature 3%, damage 3%), and 17% in the cold-store subset.
- **The score does not reliably catch those errors.** Several misses carry scores of 0.9 to 0.98; the review flag
  (26.8% of reports) helps, but the confident reports are still only 97% safety (89% in cold storage). A `safety`
  report mislabelled with high confidence would not be escalated by anything in this system.
- **What goes wrong looks like vocabulary leakage from the generator:** injury reports that mention pallets,
  forklifts, deliveries or dock doors drift to `damage` or `delay`; an electrical shock "in the box room" goes to
  `temperature`; "heat stress" goes to `delay`. The classes in the synthetic corpus are separated by topic words
  that real injury text also contains, which the template families did not teach the model to ignore.
- This is a check, not a fix: nothing was retrained on OSHA text, and the numbers above were not used to tune
  anything. Using them to train would need a held-out slice and would make this check optimistic.

## Limitations
- **Synthetic text.** Fit to my generator, not to real reports; the independent set is also mine.
- One main issue per report; no multi-label output, no "other/no incident" class.
- English only; no multilingual or heavily abbreviated warehouse jargon beyond the generator's.
- Validation is only 4 families (one per class), which makes model selection noisy.
- Reports longer than 64 tokens are truncated.
- Connected to `core` (`POST /v1/incidents` classifies and stores a report) and shown in the dashboard, but
  nothing escalates a `safety` label, and the real-text check above shows safety reports can be mislabelled.

## Reproduce
```bash
uv sync --all-packages --group nlp                       # PyTorch etc.; opt-in, about 1 GB
uv run python -m coldchain_ml.nlp.run baseline
uv run python -m coldchain_ml.nlp.run bert --seed 0      # also --seed 1, --seed 2; 17-30 min each on CPU
uv run python -m coldchain_ml.nlp.compare --seed 1       # paired comparison, per-family errors
uv run python -m coldchain_ml.nlp.run export --seed 1    # ONNX, int8, benchmarks
uv run python -m coldchain_ml.nlp.run baseline-cost
uv run python -m coldchain_ml.registry.promote_incident  # gated registration in MLflow
uv run python -m coldchain_ml.nlp.report                 # regenerates results/incident_nlp.md
```
Model weights are not committed (`models/` is git-ignored); they are rebuilt by the steps above.
