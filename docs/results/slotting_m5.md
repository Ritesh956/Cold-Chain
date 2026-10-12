# Slotting on M5 FOODS: ABC-XYZ rules vs clustering

1355 of 1437 FOODS items (those on sale for two full 52-week windows: 2014-05-26 to 2015-05-24 and 2015-05-25 to 2016-05-22). Features per item and window: log mean daily units, CV of weekly units, weekday amplitude, share of zero days, standardised. **Real retail sales, reframed as warehouse outflow**; this groups items into velocity tiers and does not model a layout, travel or slot capacity.

| Method | Groups | Silhouette (same feature space) | Year-over-year ARI | ARI vs ABC-XYZ |
|---|---|---|---|---|
| ABC-XYZ rules | 9 | 0.129 | 0.411 | 1 |
| k-means (k=5, chosen by silhouette) | 5 | 0.298 | 0.309 | 0.263 |
| HDBSCAN, min_cluster_size=25 (fixed in advance) | 0 + noise (100%) | n/a | n/a | n/a |
| HDBSCAN, min_cluster_size=25, min_samples=5 (tried after the first was all noise) | 2 + noise (26%) | 0.415 | 0.082 | 0.149 |

ABC alone is far more stable year over year than the nine cells (ARI 0.525 vs 0.411). k-means refit on random 80% subsets reproduces the full fit with ARI 0.960 on average (worst 0.878). Silhouette by k: k=3: 0.270, k=4: 0.283, k=5: 0.298, k=6: 0.263, k=7: 0.269, k=8: 0.262. Silhouette favours few, well-separated groups, so it cannot say whether more tiers would be more useful.

ABC-XYZ cell sizes (year B): {"AX": 447, "AY": 109, "AZ": 16, "BX": 346, "BY": 97, "BZ": 6, "CX": 207, "CY": 105, "CZ": 22}.

## k-means tiers (fastest first)

| Tier | Items | Median units/day | Share of volume | Median weekly CV | Median weekday amplitude | Median zero-day share | A / B / C mix |
|---|---|---|---|---|---|---|---|
| 1 | 246 | 38.3 | 55.0% | 0.24 | 0.42 | 0.00 | 100% / 0% / 0% |
| 2 | 527 | 9.5 | 21.1% | 0.24 | 0.38 | 0.01 | 32% / 50% / 18% |
| 3 | 231 | 8.0 | 11.8% | 0.74 | 0.44 | 0.29 | 32% / 30% / 38% |
| 4 | 42 | 6.6 | 1.5% | 1.37 | 0.57 | 0.59 | 36% / 14% / 50% |
| 5 | 309 | 6.5 | 10.5% | 0.30 | 0.60 | 0.04 | 21% / 36% / 43% |

HDBSCAN found no density structure with the setting fixed in advance (everything is noise: the item features form a continuum). Cluster sizes (-1 = noise), fixed setting: {"-1": 1355}; follow-up setting: {"-1": 351, "0": 965, "1": 39}. 'n/a' = fewer than two clusters, so the agreement measure is undefined.
