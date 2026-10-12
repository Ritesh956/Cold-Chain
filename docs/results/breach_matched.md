# Breach early warning at matched false-alarm budgets — SYNTHETIC data

Each model's threshold and debounce are chosen on a validation history to warn the most breaches >= 5 min ahead while staying at or under the budget, then applied unchanged to the pooled test histories (188 breach episodes). Cells are test results: share warned >= 5 min ahead (95% Wilson interval), with the measured test false alarms per zone-day in brackets. The budget is met on validation; the test rate can differ. **Simulator data: this measures fit to the simulator.**

| Budget (false alarms / zone-day) | current temperature (baseline) | linear extrapolation (baseline) | lightgbm (6 training histories) |
|---|---|---|---|
| 0.01 | 0.18 (0.13-0.24) [0.00] | 0.33 (0.27-0.40) [0.01] | 0.71 (0.64-0.77) [0.02] |
| 0.03 | 0.26 (0.20-0.32) [0.02] | 0.35 (0.29-0.42) [0.01] | 0.80 (0.73-0.85) [0.03] |
| 0.05 | 0.41 (0.35-0.49) [0.04] | 0.39 (0.32-0.46) [0.04] | 0.80 (0.73-0.85) [0.03] |
| 0.1 | 0.55 (0.48-0.62) [0.08] | 0.44 (0.37-0.51) [0.10] | 0.95 (0.91-0.97) [0.11] |
| 0.22 | 0.62 (0.55-0.68) [0.18] | 0.49 (0.42-0.57) [0.22] | 0.96 (0.92-0.98) [0.15] |

## By cause at the 0.05 budget (share warned >= 5 min ahead)

| Model | compressor_drift | door_left_open | power_cut |
|---|---|---|---|
| current temperature (baseline) | 0.69 | 0.00 | 0.62 |
| linear extrapolation (baseline) | 0.06 | 0.93 | 0.05 |
| lightgbm (6 training histories) | 0.56 | 0.96 | 0.97 |
