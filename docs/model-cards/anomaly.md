# Model card: freezer anomaly detection

**Status:** evaluated on SYNTHETIC data, plus a small real-data sanity check on NAB (below).
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
| robust z-score (temp + compressor current) | no | adds mean current while the compressor runs, over 6 h |
| **robust z-score (temp + compressor current + compressor-off duration)** | no | adds how long the compressor has been continuously off, scaled so a run longer than any normal cycle in training reads as an outlier. **The served model.** |
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
| **robust z-score (temp + compressor current + compressor-off duration)** | 1.00 / 380 min | **0.97** / 62 min |
| lightgbm supervised | 1.00 / 326 min | **0.98** |

## What this shows, and what it does not
- The drift gain comes from the **signal** (compressor current), not the model: the
  temperature-only ablation is the same detector without it.
- Isolation forest did not beat a tuned robust z-score here. On this simulator the faults are
  large, so a simple detector is enough. A more complex model is not justified by these numbers.
- Power cuts were the gap: every label-free detector caught about 80% of them, because a short cut ends
  before the temperature leaves its band, while the supervised model caught 0.98 by learning that
  "compressor off for a long time" differs from a 20-minute defrost.
- **The compressor-off rule closes it without labels.** Adding the continuous compressor-off duration lifts
  power-cut recall from 0.78 to 0.97 (median delay 62 min) and event-F1 from 0.92 to 0.95, with the same
  0.01 false alarms per zone-day. Cost: door-left-open recall slipped from 1.00 to 0.97 (the threshold tuned
  on validation moved from 5.4 to 5.7), and the rule's alarm cannot arrive before the run exceeds the limit
  (about an hour), so it is late for the shortest cuts by construction.
- **How the rule works:** the limit is 1.25x the 99.5th percentile of the off-run length over all training
  minutes (about 55 min here; normal thermostat and 20-minute defrost cycles end before that), expressed as
  a z-equivalent so it shares the detector's threshold and debounce. It tolerates training data in which
  power cuts are under 0.5% of minutes.
- **Honest provenance:** I added this rule after seeing the power-cut recall on the test histories, so the
  idea is not test-blind. The constants (99.5th percentile, 1.25 margin, z-equivalent 5) were set from the
  training history's off-run distribution before the run, and the threshold was tuned on validation only;
  nothing was tuned on the test seeds. Every other row reproduces its earlier numbers exactly.

## Real-data sanity check (NAB machine temperature)
Full table and figure: [results/anomaly_nab.md](../results/anomaly_nab.md),
[results/nab_machine_temperature.png](../results/nab_machine_temperature.png).

Real 5-minute sensor data from an industrial machine (NAB, 22,683 rows after dropping 12
duplicate timestamps; MIT licence). **Not a freezer and no compressor signal**, so only the
temperature-only detectors apply, and only **3** labelled failures are scorable (the first sits
in NAB's probationary fit period). It is a sanity check, not a benchmark: no detector can be
ranked on 3 events. The protocol was fixed before running and used no NAB labels: fit on the
probationary 15%; operating point **A** = 99.5th percentile of the detector's own probationary
scores; **B** = threshold and debounce tuned on *synthetic* validation data (unit-free z-score
detectors only).

| Detector (point) | Failures found | Chance level (mean / P>=obs) | False alarms / day |
|---|---|---|---|
| robust z-score, synthetic-tuned (B) | 2/3 | 0.20 / 0.04 | 0.00 |
| z-score (A) | 2/3 | 0.53 / 0.07 | 0.09 |
| rolling z-score (A) | 3/3 | 2.26 / 0.41 | 0.64 |
| isolation forest (A) | 3/3 | 1.85 / 0.21 | 0.81 |

- **Chance matters.** Labelled windows are ~47 h wide, so a detector with many false alarms lands in
  them by luck. Shifting each alarm series randomly shows the rolling z-score and isolation forest
  "3/3" results are indistinguishable from chance (P = 0.41 and 0.21).
- **What holds up:** the z-score detectors find the sharp failures (the Dec 16 crash and the Feb 8
  decline) with almost no false alarms (0.00-0.09/day over 67 days) and miss the subtle Jan 28 one,
  which resembles the machine's ordinary dips. The synthetic-tuned robust z-score alarmed about 1.2 h
  before both failures it caught, but 7 configurations x 3 events means a P of 0.04 is suggestive,
  not proof.
- **Transfer:** a z-threshold tuned on the simulator carried over to real data without a flood of
  false alarms. With 3 events that is encouraging, not established.
- Isolation forest gave no benefit over the simple detectors on real data either.
- Under operating point A the z-score and robust z-score are near-identical by construction (the
  quantile rule cancels the scale), so their A rows are not independent evidence.

## Limitations
- **Synthetic.** The simulator is mine. Detectors learn its signatures; the compressor-current
  rise under drift is something I built in, so the 12x earlier-detection figure is evidence for
  the *method*, not a field measurement.
- Faults are non-overlapping and separated by 12 h; real incidents compound.
- Thresholds are tuned for F1 on one validation history; a site would tune for its own cost of
  a miss vs a false alarm.
- Each zone is scored independently; no cross-zone context.
- The compressor-off rule is calibrated to this simulator's cycles (20-minute defrosts, thermostat cycles up
  to about 45 minutes). A site with longer defrosts would get a longer limit from its own training data, but the
  rule has not been checked on any real compressor log.
- Real-data evidence is 3 events from a non-freezer machine; the compressor-current finding (the main
  result) could **not** be checked on real data because no public dataset has it.
