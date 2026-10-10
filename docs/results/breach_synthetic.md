# Temperature-breach early warning — SYNTHETIC data

Question: will the temperature exceed -18 C within the next 30 min? Scored on at-risk minutes only (currently within limits, no breach in the last 10 min). Fit on seed 100, threshold + debounce tuned on seed 101, reported on seeds 102-105 pooled (188 breach episodes). **Simulator output: measures fit to this simulator, not real-world performance.**

At-risk-minute prevalence of a breach within 30 min: 0.0024. A random scorer's PR-AUC equals this prevalence.

| Model | PR-AUC (per seed min-max) | Warned >=5 min (95% CI) | Warned >=10 min | Precision (95% CI) | False alarms / zone-day | Median lead (min) |
|---|---|---|---|---|---|---|
| current temperature (baseline) | 0.311 (0.242-0.361) | 0.62 (0.55-0.69) | 0.48 | 0.35 (0.31-0.39) | 0.22 | 16 |
| linear extrapolation (baseline) | 0.167 (0.144-0.206) | 0.33 (0.27-0.40) | 0.00 | 0.83 (0.73-0.89) | 0.01 | 7 |
| lightgbm (1 training history) | 0.447 (0.435-0.463) | 0.65 (0.58-0.71) | 0.50 | 0.67 (0.60-0.73) | 0.05 | 13 |
| lightgbm (3 training histories) | 0.533 (0.506-0.562) | 0.67 (0.60-0.73) | 0.56 | 0.80 (0.73-0.85) | 0.02 | 13 |
| lightgbm (6 training histories) | 0.571 (0.529-0.619) | 0.72 (0.66-0.78) | 0.60 | 0.81 (0.75-0.85) | 0.03 | 12 |

## Breaches warned >=5 min ahead, by cause (median lead of warned ones)

| Model | compressor_drift (n=78) | door_left_open (n=71) | power_cut (n=39) |
|---|---|---|---|
| current temperature (baseline) | 0.96 / 32 min | 0.06 / 3 min | 0.97 / 16 min |
| linear extrapolation (baseline) | 0.00 / — min | 0.87 / 7 min | 0.00 / — min |
| lightgbm (1 training history) | 0.40 / 30 min | 0.77 / 10 min | 0.92 / 26 min |
| lightgbm (3 training histories) | 0.41 / 22 min | 0.87 / 11 min | 0.82 / 26 min |
| lightgbm (6 training histories) | 0.42 / 26 min | 0.93 / 11 min | 0.95 / 23 min |
