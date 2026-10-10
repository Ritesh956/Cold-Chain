import json

import pandas as pd
from coldchain_ml.nab.data import KEY, load_nab
from coldchain_ml.nab.features import UNIVARIATE_FEATURES, build_univariate_features


def _write(tmp_path, rows):
    pd.DataFrame(rows, columns=["timestamp", "value"]).to_csv(
        tmp_path / "machine_temperature_system_failure.csv", index=False
    )
    (tmp_path / "combined_windows.json").write_text(
        json.dumps({KEY: [["2025-01-01 01:00:00.000000", "2025-01-01 02:00:00.000000"]]})
    )
    (tmp_path / "combined_labels.json").write_text(json.dumps({KEY: ["2025-01-01 01:30:00"]}))


def test_loader_sorts_dedups_and_reports(tmp_path):
    base = pd.date_range("2025-01-01", periods=40, freq="5min")
    rows = [(t.isoformat(sep=" "), float(i)) for i, t in enumerate(base)]
    # clock set back: a repeated stretch appears after later timestamps
    rows += [(base[30].isoformat(sep=" "), 99.0), (base[31].isoformat(sep=" "), 99.0)]
    _write(tmp_path, rows)
    s = load_nab(tmp_path)
    assert len(s.df) == 40 and s.df["timestamp"].is_monotonic_increasing
    assert s.notes["out_of_order_steps"] == 1 and s.notes["duplicate_timestamps_dropped"] == 2
    assert s.notes["steps_not_5min_after_cleaning"] == 0
    assert s.df.loc[30, "value"] == 30.0  # first occurrence kept
    assert s.probation_end == int(0.15 * 40)
    assert len(s.windows) == 1 and s.labels == [pd.Timestamp("2025-01-01 01:30:00")]


def test_univariate_features_causal_on_5min_grid():
    temp = pd.Series(range(600), dtype=float)
    changed = temp.copy()
    changed.iloc[400:] += 100.0
    a = build_univariate_features(temp, 5)[UNIVARIATE_FEATURES]
    b = build_univariate_features(changed, 5)[UNIVARIATE_FEATURES]
    pd.testing.assert_frame_equal(a.iloc[:400], b.iloc[:400])
    assert a["d5"].iloc[10] == 1.0  # 5 minutes == 1 sample at a 5-minute step
