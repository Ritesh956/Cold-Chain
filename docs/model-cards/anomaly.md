# Model card: freezer anomaly detection

**Status:** evaluated on SYNTHETIC data only. Real-data check (NAB) is still to do.
Full table: [results/anomaly_synthetic.md](../results/anomaly_synthetic.md). Code: `ml/src/coldchain_ml/anomaly/`.

## Task
Flag, per freezer zone and per minute, the three injected faults: **door left open**
(12-35 min), **power cut** (60-180 min), **compressor drift** (efficiency decays over 3-6 days).
It must *not* alarm on benign defrost cycles or routine forklift door openings.

## Data
Simulator output (`services/simulator`), 4 zones x 90 days at 1-minute resolution.
Train = seed 100, validation = seed 101 (threshold + debounce tuning), test = seeds 102-105
pooled (176 events: 80 door, 64 power cut, 32 drift). Separate seeds give independent fault
schedules, so the test set has enough events for error bars. Ground-truth events are never
features. Inputs: temperature, compressor on/off, compressor current.

## Approaches compared
| Detector | Needs labels | Idea |
|---|---|---|
| z-score (baseline) | no | \|temp - mean\| / std from training |
| rolling z-score (baseline) | no | vs the trailing 6 h |
| robust z-score (temp) | no | median/MAD instead of mean/std (training data contains anomalies) |
| **robust z-score (temp + compressor current)** | no | adds mean current while the compressor runs, over 6 h |
| isolation forest | no | 14 causal rolling features |
| lightgbm supervised | **yes** | same features; an upper bound, not a deployable choice |

Metrics are event-level (recall over events, precision over alarm segments, false alarms per
zone-day, detection delay) with 95% Wilson intervals. Per-minute F1 is deliberately not used.

## Results (test, synthetic)
All six land at event-F1 about 0.92-0.93 except the rolling z-score (0.62; it adapts to slow
change and never sees power cuts). What differs is **how early** drift is caught:

| Detector | Drift: recall / median delay | Power cut: recall |
|---|---|---|
| robust z-score (temp only) | 1.00 / 4,416 min (~3.1 days) | 0.80 |
| robust z-score (temp + compressor current) | 1.00 / **368 min (~6 h)** | 0.78 |
| lightgbm supervised | 1.00 / 326 min | **0.98** |

## What this shows, and what it does not
- The drift gain comes from the **signal** (compressor current), not the model: the
  temperature-only ablation is the same detector without it.
- Isolation forest did not beat a tuned robust z-score here. On this simulator the faults are
  large, so a simple detector is enough. A more complex model is not justified by these numbers.
- Only the supervised model catches power cuts well (0.98 vs ~0.80), because it learns that
  "compressor off for a long time" differs from a 20-minute defrost. A label-free fix (alarm on
  compressor-off duration) is the obvious next step.

## Limitations
- **Synthetic.** The simulator is mine. Detectors learn its signatures; the compressor-current
  rise under drift is something I built in, so the 12x earlier-detection figure is evidence for
  the *method*, not a field measurement.
- Faults are non-overlapping and separated by 12 h; real incidents compound.
- Thresholds are tuned for F1 on one validation history; a site would tune for its own cost of
  a miss vs a false alarm.
- Each zone is scored independently; no cross-zone context.
