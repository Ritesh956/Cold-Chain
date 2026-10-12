"""Slotting on M5 FOODS: ABC-XYZ rules vs k-means / HDBSCAN on per-item demand features.

"Slotting" here means grouping items into velocity tiers for pick-face placement (fast movers near the
dock). It stops at the grouping: no warehouse layout, travel distance or slot capacity is modelled.

Protocol (fixed before looking at results):
  * Two disjoint 52-week windows ending at the last day of the data (year A before year B).
    Only items on sale for the whole of year A are used, so the same items appear in both years.
  * k for k-means = the k in 3..8 with the best silhouette in year B; HDBSCAN min_cluster_size = 25.
  * Stability: cluster each year separately and compare (adjusted Rand index); for k-means also refit on 30
    random 80% subsets and compare their predicted labels for all items with the full fit.
  * The ABC-XYZ partition is scored on the same footing (silhouette in the same standardised space, same
    year-over-year ARI).

Run:  uv run python -m coldchain_ml.slotting.run     (after scripts/download_m5.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from coldchain_ml.forecast.data import load_demand
from coldchain_ml.slotting.rules import abc_classes, abc_xyz, cell_name, item_features

WEEKS = 52
K_RANGE = range(3, 9)
N_SUBSAMPLES = 30
OUT = Path("docs/results")


def _sil(x: np.ndarray, labels: np.ndarray) -> float | None:
    keep = labels >= 0
    if len(set(labels[keep])) < 2 or keep.sum() < 3:
        return None
    return float(silhouette_score(x[keep], labels[keep]))


def _kmeans(x: np.ndarray, k: int, seed: int = 0) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)


def main() -> None:
    d = load_demand()
    on = ~np.isnan(d.price)
    first_on = np.where(on.any(axis=1), on.argmax(axis=1), d.n_days)
    b_hi = d.n_days  # year B ends on the last day of the data, year A is the 52 weeks before it
    b_lo = b_hi - 7 * WEEKS
    a_hi, a_lo = b_lo, b_lo - 7 * WEEKS
    keep = (first_on <= a_lo) & (d.units[:, a_lo:b_hi].sum(axis=1) > 0)
    fa, fb = (item_features(d.units[keep], lo, hi) for lo, hi in ((a_lo, a_hi), (b_lo, b_hi)))
    n = int(keep.sum())
    print(f"{n} items on sale for both years of {d.n_items}", flush=True)

    scaler_b = StandardScaler().fit(fb)
    xa, xb = StandardScaler().fit_transform(fa), scaler_b.transform(fb)
    vol_a, vol_b = np.expm1(fa[:, 0]), np.expm1(fb[:, 0])
    rules_a, rules_b = abc_xyz(vol_a, fa[:, 1]), abc_xyz(vol_b, fb[:, 1])
    abc_b = abc_classes(vol_b)

    res: dict = {"n_items": n, "n_items_total": d.n_items, "window_weeks": WEEKS,
                 "years": [f"{d.dates[a_lo].date()} to {d.dates[a_hi - 1].date()}",
                           f"{d.dates[b_lo].date()} to {d.dates[b_hi - 1].date()}"]}  # fmt: skip

    # ABC-XYZ on the same footing
    cells, counts = np.unique(rules_b, return_counts=True)
    res["abc_xyz"] = {
        "classes_used": len(cells),
        "sizes": {cell_name(int(c)): int(k) for c, k in zip(cells, counts, strict=True)},
        "silhouette": _sil(xb, rules_b),
        "year_over_year_ari": float(adjusted_rand_score(rules_a, rules_b)),
        "abc_only_year_over_year_ari": float(adjusted_rand_score(abc_classes(vol_a), abc_b)),
    }

    # k-means: choose k by silhouette in year B
    curve = {}
    for k in K_RANGE:
        curve[k] = float(silhouette_score(xb, _kmeans(xb, k).labels_))
    k_best = max(curve, key=lambda k: curve[k])
    km_b, km_a = _kmeans(xb, k_best), _kmeans(xa, k_best)
    rng = np.random.default_rng(0)
    sub_ari = []
    for _ in range(N_SUBSAMPLES):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        sub = _kmeans(xb[idx], k_best, seed=int(rng.integers(1_000_000)))
        sub_ari.append(adjusted_rand_score(km_b.labels_, sub.predict(xb)))
    order = np.argsort([np.median(vol_b[km_b.labels_ == c]) for c in range(k_best)])[
        ::-1
    ]  # fastest tier first
    profile = []
    for rank, c in enumerate(order):
        m = km_b.labels_ == c
        profile.append({
            "tier": rank + 1, "n": int(m.sum()), "median_units_per_day": float(np.median(vol_b[m])),
            "median_cv_week": float(np.median(fb[m, 1])), "median_dow_amp": float(np.median(fb[m, 2])),
            "median_zero_share": float(np.median(fb[m, 3])),
            "share_of_volume": float(vol_b[m].sum() / vol_b.sum()),
            "abc_mix": [float((abc_b[m] == j).mean()) for j in range(3)],
        })  # fmt: skip
    res["kmeans"] = {
        "k": k_best, "silhouette_by_k": curve, "silhouette": curve[k_best],
        "year_over_year_ari": float(adjusted_rand_score(km_a.labels_, km_b.labels_)),
        "subsample_ari_mean": float(np.mean(sub_ari)), "subsample_ari_min": float(np.min(sub_ari)),
        "ari_vs_abc_xyz": float(adjusted_rand_score(rules_b, km_b.labels_)),
        "nmi_vs_abc_xyz": float(normalized_mutual_info_score(rules_b, km_b.labels_)),
        "ari_vs_abc": float(adjusted_rand_score(abc_b, km_b.labels_)),
        "tiers": profile,
    }  # fmt: skip

    # HDBSCAN (no predict(), so only the year-over-year check applies). The first setting was fixed in
    # advance; it labels everything noise, so a second one (smaller min_samples) was tried AFTER seeing that
    # and is reported as a follow-up, not as the protocol result.
    def _hdb(label: str, **kw) -> dict:
        hb = HDBSCAN(copy=True, **kw).fit(xb)
        ha = HDBSCAN(copy=True, **kw).fit(xa)
        nc = int(hb.labels_.max() + 1)
        return {
            "setting": label, "clusters": nc, "noise_share": float((hb.labels_ < 0).mean()),
            "silhouette_excl_noise": _sil(xb, hb.labels_),
            "year_over_year_ari": float(adjusted_rand_score(ha.labels_, hb.labels_))
            if nc >= 2 and ha.labels_.max() >= 1 else None,
            "ari_vs_abc_xyz": float(adjusted_rand_score(rules_b, hb.labels_)) if nc >= 2 else None,
            "sizes": {int(c): int((hb.labels_ == c).sum()) for c in sorted(set(hb.labels_))},
        }  # fmt: skip

    res["hdbscan"] = _hdb("min_cluster_size=25 (fixed in advance)", min_cluster_size=25)
    res["hdbscan_followup"] = _hdb(
        "min_cluster_size=25, min_samples=5 (tried after the first was all noise)",
        min_cluster_size=25,
        min_samples=5,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slotting_m5.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (OUT / "slotting_m5.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res), flush=True)


def to_markdown(r: dict) -> str:
    a, k, h, h2 = r["abc_xyz"], r["kmeans"], r["hdbscan"], r["hdbscan_followup"]

    def f(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        "# Slotting on M5 FOODS: ABC-XYZ rules vs clustering",
        "",
        f"{r['n_items']} of {r['n_items_total']} FOODS items (those on sale for two full 52-week windows: "
        f"{r['years'][0]} and {r['years'][1]}). Features per item and window: log mean daily units, CV of weekly "
        "units, weekday amplitude, share of zero days, standardised. **Real retail sales, reframed as warehouse "
        "outflow**; this groups items into velocity tiers and does not model a layout, travel or slot capacity.",
        "",
        "| Method | Groups | Silhouette (same feature space) | Year-over-year ARI | ARI vs ABC-XYZ |",
        "|---|---|---|---|---|",
        f"| ABC-XYZ rules | {a['classes_used']} | {f(a['silhouette'])} | {f(a['year_over_year_ari'])} | 1 |",
        f"| k-means (k={k['k']}, chosen by silhouette) | {k['k']} | {f(k['silhouette'])} | "
        f"{f(k['year_over_year_ari'])} | {f(k['ari_vs_abc_xyz'])} |",
        f"| HDBSCAN, {h['setting']} | {h['clusters']} + noise ({h['noise_share']:.0%}) | "
        f"{f(h['silhouette_excl_noise'])} | {f(h['year_over_year_ari'])} | {f(h['ari_vs_abc_xyz'])} |",
        f"| HDBSCAN, {h2['setting']} | {h2['clusters']} + noise ({h2['noise_share']:.0%}) | "
        f"{f(h2['silhouette_excl_noise'])} | {f(h2['year_over_year_ari'])} | {f(h2['ari_vs_abc_xyz'])} |",
        "",
        f"ABC alone is far more stable year over year than the nine cells (ARI {f(a['abc_only_year_over_year_ari'])}"
        f" vs {f(a['year_over_year_ari'])}). k-means refit on random 80% subsets reproduces the full fit with "
        f"ARI {f(k['subsample_ari_mean'])} on average (worst {f(k['subsample_ari_min'])}). Silhouette by k: "
        + ", ".join(f"k={kk}: {v:.3f}" for kk, v in k["silhouette_by_k"].items())
        + ". Silhouette favours few, well-separated groups, so it cannot say whether more tiers would be more useful.",
        "",
        f"ABC-XYZ cell sizes (year B): {json.dumps(a['sizes'])}.",
        "",
        "## k-means tiers (fastest first)",
        "",
        "| Tier | Items | Median units/day | Share of volume | Median weekly CV | Median weekday amplitude | "
        "Median zero-day share | A / B / C mix |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in k["tiers"]:
        lines.append(
            f"| {t['tier']} | {t['n']} | {t['median_units_per_day']:.1f} | {t['share_of_volume']:.1%} | "
            f"{t['median_cv_week']:.2f} | {t['median_dow_amp']:.2f} | {t['median_zero_share']:.2f} | "
            + " / ".join(f"{x:.0%}" for x in t["abc_mix"])
            + " |"
        )
    lines += [
        "",
        "HDBSCAN found no density structure with the setting fixed in advance (everything is noise: the item "
        "features form a continuum). Cluster sizes (-1 = noise), fixed setting: "
        f"{json.dumps(h['sizes'])}; follow-up setting: {json.dumps(h2['sizes'])}. "
        "'n/a' = fewer than two clusters, so the agreement measure is undefined.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
