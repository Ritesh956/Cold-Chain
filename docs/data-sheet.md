# Data sheet

Every dataset, whether it is real or synthetic, and its licence. Updated as data is added.

| Dataset | Used for | Real / synthetic | Licence / terms | Status |
|---|---|---|---|---|
| Freezer sensor history (`coldchain_simulator`) | Anomaly detection, breach prediction | **SYNTHETIC** | Ours (MIT) | Done ✅ |
| M5 Forecasting - Accuracy (Kaggle), FOODS category: `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv` (1,437 items, 10 stores, 1,941 days to 2016-05-22) | Volume forecast, slotting | Real retail sales, **reframed** as warehouse outflow (units per item per day, summed over stores). Sales are not stock demand; no unit-to-pallet conversion | Kaggle competition terms; not redistributed, fetched with your own account by `scripts/download_m5.py` into `data/real/m5/` (git-ignored) | Done: [forecast](results/forecast_m5.md), [slotting](results/slotting_m5.md) |
| Inbound pallet volumes | Forecast | **SYNTHETIC** (replenishment policy over M5 demand) | Ours | **Not built.** Only outflow is forecast; there is no inbound or pallet series |
| NAB `machine_temperature_system_failure` (commit `ea702d7`) | Real-data sanity check for anomaly detection | Real, industrial machine (not a freezer), units undocumented | MIT (`LICENSE.txt` at that commit); not redistributed, fetched by `scripts/download_nab.py` | Done ✅ |
| Roboflow Universe pallet/forklift/person sets | Detector training | Real | Per-dataset (recorded here when chosen) | Week 3 |
| Own annotated images (~200, CVAT) | Detector training/test | Real, self-labelled | Source video licences recorded here | Week 3 |
| Mock shipping labels | Barcode/OCR eval | Printed and photographed by me | Ours | Week 5 |
| Incident-report corpus (`nlp/corpus.py`): 4,640 reports from 32 template families | Incident NLP training, validation, test | **SYNTHETIC** (programmatic generator, no LLM) | Ours (MIT) | Done ✅ |
| Independent incident set (`nlp/independent.py`): 120 reports | Incident NLP distribution-shift test | **SYNTHETIC**, written by the AI assistant that built the project (not by the owner, not from a warehouse) | Ours (MIT) | Done ✅ |
| OSHA severe injury reports (Jan 2015 to Nov 2025, 105,993 usable narratives) | Real-text check of the incident model's `safety` class (never trained on) | Real | US public domain (OSHA); not redistributed | Done, one-sided: [results/incident_osha.md](results/incident_osha.md). Downloaded by hand from osha.gov/severe-injury-reports (`January2015toNovember2025.zip`, 16.2 MB, SHA-256 `A3F7F434...846BB0`); osha.gov refuses scripted clients (HTTP 403), so there is no download script. Kept in `data/real/osha/`, which is git-ignored. |

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

## Synthetic incident text: how it is made and what it can and cannot show

`ml/src/coldchain_ml/nlp/corpus.py` builds reports from 32 *families* (8 per class: damage, temperature,
delay, safety). A family is a set of sentence structures plus its own pool of event phrases, filled with
random locations, ids, products, actors and times. Real-world mess is added: typos, shorthand ("plt",
"FLT"), dropped punctuation, odd casing, filler sentences, class-independent context (40% of reports),
and a secondary issue from another class (25%). The label is always the main issue.

- **Split by family:** per class 5 families train, 1 validation, 2 test, so held-out data has sentence
  structures never seen in training. 3,200 train / 480 validation / 960 test. The generator is
  deterministic (corpus hash `5c101ecf`) and covered by leakage tests.
- **Independent set:** 120 reports (30 per class) in a different, narrative style. Written by the AI
  assistant that built the project, so it is synthetic too; it provides distribution shift, not
  realism. Used once, after model selection.
- **A design flaw was found and fixed:** pallet ids and products appeared mostly in damage families and
  became shortcuts for the class. Adding class-independent context removed that. The baseline's scores
  moved from 0.69 / 0.78 to 0.755 / 0.80 (test / independent); the change was motivated by validation,
  not by test results.
- **Known ambiguity:** some generated reports cannot be labelled with confidence even by a person (for
  example a logger "flagged out of range" that never mentions temperature), so the achievable accuracy
  is below 100%.

Limits: a model that scores well here has learned my generator. No real incident report was used.
