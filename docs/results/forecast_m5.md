# Volume forecast on M5 FOODS (real retail data, reframed as warehouse outflow)

Target: units shipped per item per day, 28 days ahead, summed over the 10 M5 stores (a distribution centre's outflow). 1437 FOODS items, 1941 days. **Real sales data, but the 'warehouse' framing is mine**: sales are not stock demand, and nothing here converts units to pallets.

Protocol: 4 rolling 28-day test blocks ending at the last day of the data (origins at day 1828, 1856, 1884, 1912); each model refit on history up to its origin. LightGBM configuration chosen on one earlier validation origin (day 1800); no test origin was used to choose anything.

## Test results (pooled over the 4 origins)

| Method | WAPE | MASE | RMSE | Bias | WAPE of daily total outflow |
|---|---|---|---|---|---|
| LightGBM (global) | 0.285 | 0.948 | 11.90 | -3.0% | 0.053 |
| LightGBM (history + calendar only) | 0.297 | 0.974 | 13.73 | -5.2% | 0.068 |
| seasonal naive (last week) | 0.362 | 1.208 | 14.47 | -8.0% | 0.103 |
| moving average (28 d) | 0.332 | 1.064 | 14.35 | -3.2% | 0.130 |
| seasonal mean (4 weeks) | 0.316 | 1.059 | 13.04 | -3.2% | 0.069 |

LightGBM (global) minus best baseline (seasonal mean (4 weeks)) in WAPE: -0.0315, 95% CI [-0.0370, -0.0264] (items resampled, 2,000 draws); LightGBM has the lower error on 88% of items.

The history-and-calendar-only variant (the one the ml-api serves) minus the best baseline: -0.0189, 95% CI [-0.0269, -0.0080]. What the price, event, SNAP, item-age and department features add (global minus history-only): -0.0126, 95% CI [-0.0209, -0.0069]. Both variants use the configuration chosen for the full model.

WAPE per test origin: LightGBM (global): 0.305, 0.295, 0.274, 0.266; LightGBM (history + calendar only): 0.326, 0.309, 0.280, 0.276; seasonal naive (last week): 0.376, 0.362, 0.372, 0.338; moving average (28 d): 0.345, 0.340, 0.325, 0.319; seasonal mean (4 weeks): 0.335, 0.326, 0.301, 0.304

MASE is the mean over items of (item MAE / item in-sample seasonal-naive MAE, last 365 days from launch); items with no in-sample variation are skipped. WAPE = sum|error| / sum(actual).

## WAPE by horizon and by item volume

| Method | days 1-7 | 8-14 | 15-21 | 22-28 | slow third | middle third | fast third |
|---|---|---|---|---|---|---|---|
| LightGBM (global) | 0.264 | 0.285 | 0.289 | 0.300 | 0.519 | 0.351 | 0.235 |
| LightGBM (history + calendar only) | 0.269 | 0.294 | 0.307 | 0.319 | 0.547 | 0.356 | 0.248 |
| seasonal naive (last week) | 0.338 | 0.365 | 0.369 | 0.376 | 0.651 | 0.457 | 0.297 |
| moving average (28 d) | 0.326 | 0.320 | 0.326 | 0.356 | 0.586 | 0.385 | 0.283 |
| seasonal mean (4 weeks) | 0.297 | 0.312 | 0.319 | 0.336 | 0.600 | 0.390 | 0.258 |

Volume thirds are terciles of each origin's 90-day mean daily outflow.

## Choosing the LightGBM configuration (validation origin only)

| Configuration | validation WAPE |
|---|---|
| tweedie 1.1, 63 leaves **(chosen)** | 0.3080 |
| poisson, 63 leaves | 0.3145 |
| L1, 63 leaves | 0.3287 |
| tweedie 1.1, 31 leaves | 0.3105 |

## Most important features (LightGBM gain, final origin)

`mean7` (48.5%), `mean14` (23.6%), `mean28` (8.7%), `swd_mean` (3.5%), `mean56` (2.8%), `mean90` (2.7%), `price_vs_origin` (2.1%), `price` (1.2%)

Notes: {"items_scored_per_origin": [1437, 1437, 1437, 1437], "unit": "units/day summed over 10 stores", "first_day": "2011-01-29", "last_day": "2016-05-22"}
