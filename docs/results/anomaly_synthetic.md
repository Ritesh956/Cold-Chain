# Freezer anomaly detection — SYNTHETIC data

Event-level metrics on 4 pooled test histories (seeds 102-105, 4 zones x 90 days each). Threshold and debounce tuned on a separate validation history (seed 101); models fit on seed 100. **All data is simulator output: these numbers measure how well a detector learns this simulator, not real-world performance.**

| Detector | Recall (95% CI) | Precision (95% CI) | F1 | False alarms / zone-day | Median delay (min) |
|---|---|---|---|---|---|
| z-score (baseline) | 0.93 (0.88-0.96) | 0.93 (0.90-0.96) | 0.93 | 0.01 | 42 |
| rolling z-score (baseline) | 0.47 (0.39-0.54) | 0.91 (0.83-0.95) | 0.62 | 0.01 | 9 |
| robust z-score (temp) | 0.92 (0.87-0.95) | 0.94 (0.90-0.96) | 0.93 | 0.01 | 38 |
| robust z-score (temp + compressor current) | 0.92 (0.87-0.95) | 0.92 (0.87-0.95) | 0.92 | 0.01 | 32 |
| robust z-score (temp + compressor current + compressor-off duration) | 0.98 (0.94-0.99) | 0.93 (0.88-0.96) | 0.95 | 0.01 | 34 |
| isolation forest | 0.93 (0.88-0.96) | 0.93 (0.90-0.95) | 0.93 | 0.02 | 37 |
| lightgbm supervised (upper bound) | 0.93 (0.88-0.96) | 0.91 (0.87-0.94) | 0.92 | 0.02 | 29 |

## Recall and median delay by fault type

| Detector | compressor_drift | door_left_open | power_cut |
|---|---|---|---|
| z-score (baseline) | 1.00 / 4257 min | 1.00 / 10 min | 0.80 / 76 min |
| rolling z-score (baseline) | 0.53 / 5281 min | 0.81 / 8 min | 0.00 / — min |
| robust z-score (temp) | 1.00 / 4416 min | 0.99 / 8 min | 0.80 / 71 min |
| robust z-score (temp + compressor current) | 1.00 / 368 min | 1.00 / 10 min | 0.78 / 76 min |
| robust z-score (temp + compressor current + compressor-off duration) | 1.00 / 380 min | 0.97 / 10 min | 0.97 / 62 min |
| isolation forest | 1.00 / 3688 min | 1.00 / 6 min | 0.80 / 78 min |
| lightgbm supervised (upper bound) | 1.00 / 326 min | 0.85 / 8 min | 0.98 / 45 min |

Test events per type: {'power_cut': 64, 'door_left_open': 80, 'compressor_drift': 32}.

## False alarms by cause

| Detector | defrost | routine_door | unexplained |
|---|---|---|---|
| z-score (baseline) | 16 | 5 | 0 |
| rolling z-score (baseline) | 9 | 0 | 0 |
| robust z-score (temp) | 13 | 1 | 0 |
| robust z-score (temp + compressor current) | 13 | 1 | 0 |
| robust z-score (temp + compressor current + compressor-off duration) | 12 | 1 | 0 |
| isolation forest | 19 | 4 | 8 |
| lightgbm supervised (upper bound) | 10 | 9 | 7 |
