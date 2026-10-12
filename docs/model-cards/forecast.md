# Model card: 28-day warehouse-outflow forecast

**Status:** evaluated on **real M5 retail sales, reframed as warehouse outflow**. Full tables:
[results/forecast_m5.md](../results/forecast_m5.md). Code: `ml/src/coldchain_ml/forecast/`. Not served by the
`ml-api`, not in the MLflow registry, and not connected to `core` or the dashboard.

## Task
For every FOODS item, forecast the units shipped on each of the next 28 days. The series is the M5 sales of
the item summed over the 10 stores, which is what a distribution centre serving them would ship. Two things in
that framing are mine: **sales are not stock demand** (a zero can be a stock-out), and **there is no pallet
conversion**, so these are units, not pallets. Inbound volumes are not forecast.

## Data and protocol
1,437 FOODS items, 1,941 days (2011-01-29 to 2016-05-22). Four rolling test blocks of 28 days ending at the last
day of the data; each model is refit on history up to its origin only. A single earlier origin (day 1800) was
used to pick the LightGBM configuration among four candidates, and no test origin was used to choose anything.
Items are scored if they are on sale at the origin (all 1,437 at every test origin).

LightGBM is one **global** model over all items with a **direct** multi-horizon design: one row per
(item, origin, horizon). Features are causal summaries of the past at the origin (means over 7 to 90 days, 28-day
spread and zero share, the same weekday in each of the last 4 weeks) plus what is known about the target day in
advance: weekday, month, day of month, events, SNAP share and shelf price. Training rows: origins every 10 days
over the 730 days before the test origin, targets observed by the origin. A test checks that changing any sale
after the origin leaves every feature unchanged.

## Results (test, pooled over 4 origins)
| Method | WAPE | MASE | RMSE | Bias | WAPE of daily total outflow |
|---|---|---|---|---|---|
| seasonal naive (last week) | 0.362 | 1.208 | 14.47 | -8.0% | 0.103 |
| moving average (28 d) | 0.332 | 1.064 | 14.35 | -3.2% | 0.130 |
| seasonal mean (4 weeks) | 0.316 | 1.059 | 13.04 | -3.2% | 0.069 |
| **LightGBM (global)** | **0.285** | **0.948** | **11.90** | -3.0% | **0.053** |

- LightGBM beats the best baseline by 0.0315 WAPE (95% CI 0.026 to 0.037, items resampled) and has the lower
  error on 88% of items. It wins at every one of the four origins (0.266 to 0.305 against 0.301 to 0.335).
- The gain is largest where volume is lowest: WAPE 0.519 against 0.600 on the slowest third of items, 0.235
  against 0.258 on the fastest third. Error grows slowly with horizon (0.264 for days 1 to 7, 0.300 for days 22 to 28).
- Both the model and the baselines under-forecast (about 3% less than actual, 8% for seasonal naive).
- By gain, the model relies mostly on recent level (the 7-day and 14-day means carry 72% of it); price features
  carry at least 3%.

## Findings
- **The model earns its place, modestly.** A 10% relative WAPE reduction over a four-week seasonal mean is real
  and consistent, but it is not a different order of magnitude. The strongest classical baseline is hard to beat
  by much on weekly-seasonal retail series.
- **Total outflow is far easier than item outflow** (WAPE 0.05 against 0.29), as expected from pooling many noisy
  items; a capacity plan for the whole dock needs the first, a slotting or replenishment decision needs the second.
- **MASE below 1 is not "better than naive" in the usual sense here.** Its scale is the one-step seasonal-naive error
  (lag 7) in the last year, while the forecast horizon is up to 28 days, so the denominator is easier than what
  is being predicted.

## Limitations
- **The 95% interval covers item sampling only.** The four test blocks are consecutive and all in spring 2016, so
  nothing here says how the model does around Christmas, Thanksgiving or Easter, where the calendar features matter
  most and the training window contains only two of each.
- **Prices of the forecast weeks are assumed known**, as M5 publishes them. A real planner may not know next
  month's promotions. Price features carry little weight, but no run without them was made.
- Point forecasts only. Safety stock and capacity decisions need prediction intervals, which this does not produce.
- One LightGBM seed and four candidate configurations that differ by 0.003 to 0.02 on a single validation origin,
  so the choice among them is weakly supported.
- Aggregating over stores makes the series smoother and the problem easier than store-level M5; the numbers must not
  be compared with the M5 leaderboard.
- Training and evaluation take about 7 minutes on a 20-core CPU. No latency or serving numbers exist.
