# Model card: temperature-breach early warning

**Status:** evaluated on SYNTHETIC data only. Full table: [results/breach_synthetic.md](../results/breach_synthetic.md).
Code: `ml/src/coldchain_ml/breach/`.

## Task
At every minute, predict whether the observed zone temperature will exceed **-18 C** (the
frozen-food limit) within the next **30 minutes**. Only "at-risk" minutes are scored: currently within
limits and no breach in the last 10 minutes (predicting a breach already under way is not a
prediction). A useful warning leaves time to act, so we report **how many breaches were warned
>= 5 and >= 10 minutes ahead** and the median lead time, not just a ranking metric.

## Data
Same simulator and split as the anomaly task: tune on seed 101, report on seeds 102-105 pooled
(188 breach episodes, 2.05 M at-risk minutes, 4,937 positive minutes = 0.24%). Training uses 1, 3
or 6 independent simulator histories (seeds 100, 106-110) to show a learning curve. Features are
causal rolling statistics plus gap to the limit, 5/15-minute slopes and compressor on/off run
length. Ground-truth events are never features; the label looks forward and is used only as a label.

## Results (test, synthetic)
| Model | PR-AUC | Warned >= 5 min | Precision | False alarms / zone-day |
|---|---|---|---|---|
| current temperature (baseline) | 0.311 | 0.62 | 0.35 | 0.22 |
| linear extrapolation (baseline) | 0.167 | 0.33 | 0.83 | 0.01 |
| **LightGBM, 6 training histories** | **0.571** | **0.72** | **0.81** | **0.03** |
| LightGBM, 1 training history | 0.447 | 0.65 | 0.67 | 0.05 |

A random scorer's PR-AUC is the prevalence, 0.0024. Learning curve: PR-AUC 0.447 -> 0.533 -> 0.571
for 1 -> 3 -> 6 histories; still rising, so more data would likely help.

By cause (share of breaches warned >= 5 min ahead):

| Model | drift (n=78) | door left open (n=71) | power cut (n=39) |
|---|---|---|---|
| current temperature | 0.96 | **0.06** | 0.97 |
| linear extrapolation | **0.00** | 0.87 | **0.00** |
| LightGBM (6 histories) | **0.42** | 0.93 | 0.95 |

## Findings
- **No single heuristic covers all causes.** A temperature threshold catches slow causes but almost
  never a door left open (the temperature is nowhere near the limit when the warning should fire);
  extrapolating the trend is the reverse. The model learns both, with about 7x fewer false
  alarms per zone-day than the temperature baseline (0.03 vs 0.22) and a higher warned share (0.72 vs 0.62).
- **The model's weak spot is drift.** It warns only 42% of drift breaches, where a plain
  temperature threshold warns 96% (at the cost of 0.22 false alarms per zone-day). The operating point was
  tuned for overall F1 on a mixture of causes, which trades drift recall for precision. A different
  threshold or a drift-specific model could change this; I have not tested that.
- Compressor signals matter less here than for anomaly detection: top features by gain are
  temperature, 30-minute slope and gap to the limit.

## What went wrong first (kept in the record)
The first configuration (400 trees, 31 leaves) **overfit**: PR-AUC *fell* with more data
(0.30 -> 0.17 on test), because positives are few and autocorrelated. I first guessed the
cause was drift memorisation; a split of PR-AUC inside/outside drift periods disproved that. I then
compared 3 regularisation variants **on the validation seed only** and kept the one with the best
validation PR-AUC at full size (15 leaves, min 200 samples per leaf, L2 = 5; val PR-AUC 0.20 -> 0.63).
Caveat: the test seeds had already been seen with the first configuration when I diagnosed it, so
the test numbers are not perfectly untouched, though the final choice used validation only.

## Limitations
- **Synthetic.** The simulator is mine, and so is the -18 C / 30 min framing; results measure fit
  to it, not a real site.
- Door-left-open breaches are partly unpredictable: roughly 12 minutes pass between the door opening
  and the breach, and nothing observable precedes the opening, so a 10-minute lead is near the ceiling.
- Operating points differ per model (each tuned for F1 of warned-5-min vs precision), so rows are
  not compared at matched false-alarm budgets. A matched-budget comparison is the fairer next test.
- Per-seed PR-AUC ranges (0.53-0.62 for the final model) show real spread across histories; there is
  no confidence interval on PR-AUC itself.
