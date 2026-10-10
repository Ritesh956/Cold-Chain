# Data sheet

Every dataset, whether it is real or synthetic, and its licence. Updated as data is added.

| Dataset | Used for | Real / synthetic | Licence / terms | Status |
|---|---|---|---|---|
| Freezer sensor history (`coldchain_simulator`) | Anomaly detection, breach prediction | **SYNTHETIC** | Ours (MIT) | Week 1 ✅ |
| M5 Forecasting (Kaggle), FOODS subset | Volume forecast, slotting | Real retail sales, **reframed** as warehouse outflow | Kaggle competition terms; not redistributed | Week 2 |
| Inbound pallet volumes | Forecast | **SYNTHETIC** (replenishment policy over M5 demand) | Ours | Week 2 |
| NAB `machine_temperature_system_failure` (commit `ea702d7`) | Real-data sanity check for anomaly detection | Real, industrial machine (not a freezer), units undocumented | MIT (`LICENSE.txt` at that commit); not redistributed, fetched by `scripts/download_nab.py` | Done ✅ |
| Roboflow Universe pallet/forklift/person sets | Detector training | Real | Per-dataset (recorded here when chosen) | Week 3 |
| Own annotated images (~200, CVAT) | Detector training/test | Real, self-labelled | Source video licences recorded here | Week 3 |
| Mock shipping labels | Barcode/OCR eval | Printed and photographed by me | Ours | Week 5 |
| Incident reports (LLM-generated + hand-written) | Incident NLP | **SYNTHETIC** / hand-written | Ours | Week 5 |
| OSHA severe injury reports | Real text for the safety class | Real | US public domain | Week 5 |

## Synthetic sensor data: how it is made and what it can and cannot show

`services/simulator` runs a first-order thermal model of a freezer zone (envelope leak,
thermostat with hysteresis, door heat exchange, defrost heater) at 1-minute resolution.
Ground-truth events are injected and kept in a separate table the detectors never see:

- **Anomalies (should be flagged):** door left open 12-35 min, power cut 60-180 min,
  compressor drift (efficiency ramps down over 3-6 days, then repaired).
- **Benign look-alikes (should NOT be flagged):** defrost every 8 h (staggered per zone) and
  routine forklift door openings (about 3 per hour in working hours).
- **Nuisance:** 0.1% sensor dropouts, sensor noise, compressor-current noise.

Limits: the model is mine, not a real plant's. Detectors that score well here have learned
my simulator, not a real freezer. Results on this data are reported next to results on the
real NAB series, and never presented as real-world performance.

Default dataset (seed 42, 90 days, 4 zones): 44 anomaly events, 1,080 defrosts,
17,214 routine door events.
