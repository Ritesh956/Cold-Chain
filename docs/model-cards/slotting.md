# Model card: item slotting tiers (ABC-XYZ vs clustering)

**Status:** evaluated on **real M5 retail sales, reframed as warehouse outflow**. Full table:
[results/slotting_m5.md](../results/slotting_m5.md). Code: `ml/src/coldchain_ml/slotting/`. Not served by the
`ml-api` and not connected to `core` or the dashboard.

## Task
Group items into velocity tiers for pick-face placement (fast movers near the dock, slow or erratic ones
further away). The output is a tier per item. It does **not** place anything: no layout, travel distance,
slot capacity or replenishment is modelled, so "slotting" here means the grouping step only.

## Data
M5 FOODS, daily units per item summed over the 10 stores (the outflow of a distribution centre that serves them).
1,355 of 1,437 items were on sale for two full 52-week windows (2014-05-26 to 2015-05-24, and 2015-05-25 to
2016-05-22). Four features per item and window, standardised: log mean daily units, coefficient of variation (CV)
of weekly units, weekday amplitude, share of zero days. Sales are not stock demand; a zero can be a stock-out.

## Methods
- **ABC-XYZ rules (baseline).** Textbook thresholds fixed before any result: A/B/C = the items making up the first
  80% / next 15% / last 5% of volume; X/Y/Z = weekly CV below 0.5 / 0.5 to 1.0 / above 1.0.
- **k-means**, k chosen by silhouette among 3 to 8 (k = 5).
- **HDBSCAN**, `min_cluster_size` 25 fixed in advance.

## Results
| Method | Groups | Silhouette (same space) | Year-over-year ARI | ARI vs ABC-XYZ |
|---|---|---|---|---|
| ABC-XYZ rules | 9 | 0.129 | 0.411 | 1 |
| k-means (k = 5) | 5 | 0.298 | 0.309 | 0.263 |
| HDBSCAN (fixed setting) | 0 + all noise | n/a | n/a | n/a |
| HDBSCAN (`min_samples` 5, tried afterwards) | 2 + 26% noise | 0.415 | 0.082 | 0.149 |

Year-over-year ARI = cluster each year separately and compare the labels of the same items (adjusted Rand index:
1 = identical partition, 0 = chance). k-means refit on random 80% subsets reproduces the full fit with ARI 0.96
on average (worst 0.88).

k-means tiers, fastest first (year B): 246 items with a median of 38 units/day carry 55% of volume (all class A);
527 steady mid-volume items (9.5/day, CV 0.24); 231 erratic items (8.0/day, CV 0.74, 29% zero days); 42 very
erratic items (CV 1.37, 59% zero days, 1.5% of volume); 309 low-volume items with a strong weekday pattern
(6.5/day, amplitude 0.60).

## Findings
- **The rules are not beaten.** k-means scores a much higher silhouette (0.30 vs 0.13), but that is partly by
  construction: k-means optimises compactness in this very space and the nine rule cells do not. On the stability
  that matters for a layout people have to live with, the rules win: 0.41 against 0.31 year over year. Two years of
  one retailer's data is a small basis for choosing between them.
- **Clustering adds one thing the rules lack:** it separates *how* an item is erratic. The 42 intermittent items
  (many zero days) and the 309 weekday-driven items are not distinguishable by ABC-XYZ, which sees only volume and
  weekly CV. Whether that is worth a more complex scheme was not tested.
- **ABC alone is steadier than the nine cells** (year-over-year ARI 0.53 vs 0.41), and three of the nine cells are
  nearly empty (AZ 16, BZ 6, CZ 22 items), so XYZ adds instability for little gain at these thresholds.
- **HDBSCAN found no density structure.** With the setting chosen in advance every item is noise: the features
  form a continuum rather than separate clumps. A second setting was tried after seeing that and is reported as a
  follow-up, not as the protocol result; it splits off two groups with 26% noise and low stability (0.08).
- Silhouette favours few, well-separated groups, so it cannot say whether five tiers are more useful than four
  or six (k = 3 to 8 span 0.26 to 0.30).

## Limitations
- One retailer, one category, aggregated across stores; a warehouse with different customers would see different
  patterns.
- No operational outcome (pick time, travel, replenishment) is measured, so "better tiers" is undefined beyond
  stability and agreement with the rules.
- The k and thresholds were not tuned against any downstream cost.
- Items launched or delisted inside the two windows (82 of 1,437) are excluded.
