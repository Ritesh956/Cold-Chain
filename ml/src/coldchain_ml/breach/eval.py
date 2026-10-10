"""Episode-level evaluation of breach early warnings.

  * breach episode : run of observed temp > BREACH_C (gaps <= 10 min merged).
  * alarm          : model score > threshold on an at-risk minute (min-run debounce applied).
  * warned         : an alarm segment starts before the episode start and ends no more than
                     HORIZON minutes before it. lead = episode start - first alarm minute.
  * true segment   : an alarm segment that warned some episode; otherwise a false alarm.
  * cause          : the injected anomaly overlapping the 4 h before the breach (door_left_open,
                     power_cut, compressor_drift) or "other".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from coldchain_ml.anomaly.events_eval import apply_min_run, segments
from coldchain_ml.breach.data import BREACH_C, HORIZON

EPISODE_MERGE_GAP = 10
CAUSE_LOOKBACK_MIN = 240


@dataclass
class BreachMetrics:
    n_episodes: int = 0
    n_segments: int = 0
    n_true_segments: int = 0
    zone_days: float = 0.0
    leads: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))  # 0 = not warned

    def _all(self) -> list[float]:
        return [x for v in self.leads.values() for x in v]

    def warned_frac(self, min_lead: float = 1.0, cause: str | None = None) -> float:
        vals = self.leads[cause] if cause else self._all()
        return sum(x >= min_lead for x in vals) / len(vals) if vals else 0.0

    def count_warned(self, min_lead: float = 1.0) -> int:
        return sum(x >= min_lead for x in self._all())

    def median_lead(self, cause: str | None = None) -> float | None:
        vals = [x for x in (self.leads[cause] if cause else self._all()) if x >= 1]
        return float(np.median(vals)) if vals else None

    @property
    def precision(self) -> float:
        return self.n_true_segments / self.n_segments if self.n_segments else 0.0

    @property
    def false_alarms_per_zone_day(self) -> float:
        return (self.n_segments - self.n_true_segments) / self.zone_days if self.zone_days else 0.0

    def f1(self, min_lead: float = 5.0) -> float:
        p, r = self.precision, self.warned_frac(min_lead)
        return 2 * p * r / (p + r) if p + r else 0.0

    def merge(self, o: BreachMetrics) -> BreachMetrics:
        self.n_episodes += o.n_episodes
        self.n_segments += o.n_segments
        self.n_true_segments += o.n_true_segments
        self.zone_days += o.zone_days
        for k, v in o.leads.items():
            self.leads[k].extend(v)
        return self


def _cause(ts: pd.DatetimeIndex, start_idx: int, anomalies: pd.DataFrame) -> str:
    t = ts[start_idx]
    lo = t - pd.Timedelta(minutes=CAUSE_LOOKBACK_MIN)
    hit = anomalies[(anomalies["start"] <= t) & (anomalies["end"] >= lo)]
    if hit.empty:
        return "other"
    return str(hit.sort_values("start").iloc[-1]["type"])


def evaluate_zone(
    ts: pd.DatetimeIndex, temp: np.ndarray, alarm: np.ndarray, events: pd.DataFrame
) -> BreachMetrics:
    m = BreachMetrics(zone_days=len(ts) / 1440.0)
    breached = np.nan_to_num(temp, nan=-999.0) > BREACH_C
    ep_starts, _ = segments(breached, EPISODE_MERGE_GAP)
    a_starts, a_ends = segments(alarm, merge_gap=0)
    m.n_segments = len(a_starts)
    anomalies = events[events["is_anomaly"]]
    true_seg = np.zeros(len(a_starts), dtype=bool)

    for s in ep_starts:
        m.n_episodes += 1
        # segments that began before the breach and ended within HORIZON minutes of it
        match = (a_starts < s) & ((s - (a_ends - 1)) <= HORIZON)
        lead = float(s - a_starts[match].min()) if match.any() else 0.0
        true_seg |= match
        m.leads[_cause(ts, int(s), anomalies)].append(lead)
    m.n_true_segments = int(true_seg.sum())
    return m


def evaluate(
    readings: pd.DataFrame,
    events: pd.DataFrame,
    score: np.ndarray,
    at_risk: np.ndarray,
    thr: float,
    min_run: int = 1,
) -> BreachMetrics:
    total = BreachMetrics()
    alarm_raw = (np.nan_to_num(score, nan=-np.inf) > thr) & np.asarray(at_risk, dtype=bool)
    for zone, idx in readings.groupby("zone").indices.items():
        z = readings.iloc[idx]
        alarm = apply_min_run(alarm_raw[idx], min_run)
        total.merge(
            evaluate_zone(
                pd.DatetimeIndex(z["ts"]),
                z["temp_c"].to_numpy(),
                alarm,
                events[events["zone"] == zone],
            )
        )
    return total
