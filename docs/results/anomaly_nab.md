# Real-data check: NAB machine temperature

Public real sensor data (NAB `machine_temperature_system_failure`, 5-minute samples, 22,683 rows, units not documented). It is an industrial machine, **not a freezer**: it has no compressor signals, so only the temperature-only detectors apply. This is a **sanity check, not a benchmark**: only 3 labelled failures are scorable (failure 1 falls inside NAB's 15% probationary fit period), so no detector can be shown better than another.

Protocol fixed before running: fit on the probationary 15% (no labels); **A** = threshold at the 99.5% quantile of the detector's own probationary scores, no debounce; **B** = threshold + debounce tuned on *synthetic* validation data, unit-free z-score detectors only. No NAB label was used to choose anything. Scored period: 67 days.

| Detector | Operating point | Failures detected (of 3, 95% CI) | Chance level: mean detected / P(>= observed) | First alarm vs labelled failure (h) | False alarms / day | Time in alarm |
|---|---|---|---|---|---|---|
| z-score | A | 2/3 (0.21-0.94) | 0.53 / 0.07 | -9.6, miss, -18.2 | 0.09 | 3.2% |
| z-score | B | 2/3 (0.21-0.94) | 0.26 / 0.05 | -1.8, miss, -9.5 | 0.00 | 2.0% |
| rolling z-score | A | 3/3 (0.44-1.00) | 2.26 / 0.41 | -21.9, -10.2, +21.3 | 0.64 | 0.8% |
| rolling z-score | B | 2/3 (0.21-0.94) | 0.91 / 0.22 | -1.9, miss, +21.3 | 0.15 | 0.2% |
| robust z-score (temp) | A | 2/3 (0.21-0.94) | 0.53 / 0.07 | -9.6, miss, -18.2 | 0.09 | 3.2% |
| robust z-score (temp) | B | 2/3 (0.21-0.94) | 0.20 / 0.04 | -1.2, miss, -1.2 | 0.00 | 0.2% |
| isolation forest | A | 3/3 (0.44-1.00) | 1.85 / 0.21 | -21.3, -20.2, -17.7 | 0.81 | 3.9% |

Chance level: the alarm series is circularly shifted by a random offset (2000 times), keeping its alarm count, durations and clustering; the columns give the mean number of failures that random placement would 'detect' and the probability it does at least as well as the detector. Labelled windows are ~47 h wide, so noisy detectors score by luck.

First-alarm offsets are listed for failures 2, 3, 4 in order; negative = the alarm came before the labelled failure timestamp, "miss" = no alarm inside the labelled window.

Data notes: {"rows_raw": 22695, "rows_clean": 22683, "out_of_order_steps": 1, "duplicate_timestamps_dropped": 12, "steps_not_5min_after_cleaning": 0}
