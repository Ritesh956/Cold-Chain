"""Serving contract: how much recent history each model needs to reproduce its training features.

All rolling windows are <= 1440 minutes and run-lengths are capped at 1440 (breach/data.py), so
1500 minutes of 1-minute readings reproduces training-time features exactly (see
ml/tests/test_serving_parity.py). The anomaly detector only uses windows up to 360 minutes.
"""

STEP_MIN = 1
ANOMALY_MIN_HISTORY = 400
BREACH_MIN_HISTORY = 1500
MAX_HISTORY = 5000
