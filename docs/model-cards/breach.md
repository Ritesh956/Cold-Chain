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

## Matched false-alarm budgets
The table above compares models at the operating point each was tuned to (best F1), so their false-alarm
rates differ. [results/breach_matched.md](../results/breach_matched.md) fixes the budget instead: for each
budget the threshold and debounce are chosen on the validation history to warn as many breaches >= 5 min ahead
as possible within it, then applied unchanged to the test histories. Share warned >= 5 min ahead (test, 188
breaches; measured test false alarms per zone-day in brackets):

| Budget | current temperature | linear extrapolation | LightGBM (6 histories) |
|---|---|---|---|
| 0.03 | 0.26 [0.02] | 0.35 [0.01] | **0.80** [0.03] |
| 0.05 | 0.41 [0.04] | 0.39 [0.04] | **0.80** [0.03] |
| 0.10 | 0.55 [0.08] | 0.44 [0.10] | **0.95** [0.11] |
| 0.22 | 0.62 [0.18] | 0.49 [0.22] | **0.96** [0.15] |

- **LightGBM wins at every budget**, by a wide margin and with non-overlapping 95% intervals (see the full
  table). The comparison at the temperature baseline's own 0.22 budget is the fairest one: 0.96 against 0.62.
- **The "weak at drift" finding above was partly an operating-point artefact.** At the F1-tuned point the
  model warns 42% of drift breaches; with a 0.10 budget it warns 92% (0.56 at 0.05). The model can warn
  drift early; the F1-tuned threshold simply spends its budget elsewhere. Drift is still the hardest cause
  at tight budgets.
- **The published headline point (0.72 at 0.03) is not the best the model can do at that budget** (0.80),
  because F1 trades recall against precision rather than maximising recall inside a budget.
- **Budgets are met on validation, not guaranteed on test.** At the tightest budget (0.01) the model's test
  rate was 0.022, so the 0.71 there comes with twice the false alarms it was allowed; read that row with care.
- Same caveats as everything here: one simulator, a single validation history for the thresholds, no
  confidence interval on the false-alarm rate.

## Findings
- **No single heuristic covers all causes.** A temperature threshold catches slow causes but almost
  never a door left open (the temperature is nowhere near the limit when the warning should fire);
  extrapolating the trend is the reverse. The model learns both, with about 7x fewer false
  alarms per zone-day than the temperature baseline (0.03 vs 0.22) and a higher warned share (0.72 vs 0.62).
- **At the F1-tuned operating point the model's weak spot is drift.** It warns only 42% of drift breaches,
  where a plain temperature threshold warns 96% (at the cost of 0.22 false alarms per zone-day). The matched-budget
  comparison above shows this is mostly the operating point: at a 0.10 budget the model warns 92% of drift
  breaches.
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
- The main table's operating points differ per model (each tuned for F1 of warned-5-min vs precision); the
  matched-budget section above is the like-for-like comparison. Its budgets are chosen on a single validation
  history, so the test false-alarm rate can differ from the budget.
- Per-seed PR-AUC ranges (0.53-0.62 for the final model) show real spread across histories; there is
  no confidence interval on PR-AUC itself.
