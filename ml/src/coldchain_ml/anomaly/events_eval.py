"""Event-level evaluation of anomaly alarms.

Per-minute F1 rewards flagging easy points and ignores what operators care about: was the
incident caught, how late, and how many false alarms did it cost? So we score *events*.

Definitions (all on a regular 1-minute grid, per zone):
  * alarm segment  : run of alarm minutes; gaps <= `merge_gap` minutes are merged.
  * true event     : an injected anomaly (is_anomaly=True).
  * detected       : some alarm segment overlaps [event.start, event.end + tol).
  * true segment   : an alarm segment overlapping any true event's window.
  * false segment  : an alarm segment overlapping none. Attributed to a benign cause
                     (defrost / routine_door) when it overlaps one, else "unexplained".
  * delay          : minutes from event start to the first overlapping alarm minute (>= 0).
  * recall         = detected / events;  precision = true segments / segments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BENIGN_TOL_MIN = 15  # thermal response lags the cause


def apply_min_run(alarm: np.ndarray, min_run: int) -> np.ndarray:
    """Drop alarm runs shorter than `min_run` minutes (cheap debounce)."""
    if min_run <= 1:
        return alarm
    starts, ends = segments(alarm, merge_gap=0)
    keep = (ends - starts) >= min_run
    delta = np.zeros(len(alarm) + 1, dtype=np.int32)
    np.add.at(delta, starts[keep], 1)
    np.add.at(delta, ends[keep], -1)
    return np.cumsum(delta[:-1]) > 0


def segments(alarm: np.ndarray, merge_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (starts, ends_exclusive) of alarm runs, merging gaps <= merge_gap."""
    padded = np.concatenate([[0], alarm.astype(np.int8), [0]])
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if len(starts) < 2 or merge_gap <= 0:
        return starts, ends
    keep = (starts[1:] - ends[:-1]) > merge_gap
    return starts[np.concatenate([[True], keep])], ends[np.concatenate([keep, [True]])]


@dataclass
class EventMetrics:
    n_events: int = 0
    n_detected: int = 0
    n_segments: int = 0
    n_true_segments: int = 0
    zone_days: float = 0.0
    events_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    detected_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    delays_min: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    false_by_cause: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def n_false_segments(self) -> int:
        return self.n_segments - self.n_true_segments

    @property
    def recall(self) -> float:
        return self.n_detected / self.n_events if self.n_events else 0.0

    @property
    def precision(self) -> float:
        return self.n_true_segments / self.n_segments if self.n_segments else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def false_alarms_per_zone_day(self) -> float:
        return self.n_false_segments / self.zone_days if self.zone_days else 0.0

    def median_delay(self, kind: str | None = None) -> float | None:
        vals = (
            [d for v in self.delays_min.values() for d in v] if kind is None else self.delays_min[kind]
        )
        return float(np.median(vals)) if vals else None

    def merge(self, other: EventMetrics) -> EventMetrics:
        """Pool counts (use this to combine zones/seeds; never average the ratios)."""
        self.n_events += other.n_events
        self.n_detected += other.n_detected
        self.n_segments += other.n_segments
        self.n_true_segments += other.n_true_segments
        self.zone_days += other.zone_days
        for src, dst in (
            (other.events_by_type, self.events_by_type),
            (other.detected_by_type, self.detected_by_type),
            (other.false_by_cause, self.false_by_cause),
        ):
            for k, v in src.items():
                dst[k] += v
        for k, v in other.delays_min.items():
            self.delays_min[k].extend(v)
        return self


def _overlaps(a: int, b: int, w0: int, w1: int) -> bool:
    return a < w1 and b > w0


def evaluate_zone(
    ts: pd.DatetimeIndex | pd.Series,
    alarm: np.ndarray,
    events: pd.DataFrame,
    tol_min: int = 30,
    merge_gap: int = 5,
    attribute_false: bool = True,
    step_min: int = 1,
) -> EventMetrics:
    """Score one zone's alarm series. `events` must be this zone's rows only.

    `tol_min` / `merge_gap` are in minutes; `step_min` is the sampling interval, so the same
    definitions work on 1-minute simulator data and on 5-minute real data (NAB).
    """
    ts = pd.DatetimeIndex(ts)
    n = len(ts)
    tol = -(-tol_min // step_min)  # ceil, in samples
    gap = merge_gap // step_min
    benign_tol = -(-BENIGN_TOL_MIN // step_min)

    def pos(t: pd.Timestamp) -> int:
        return int(ts.searchsorted(t))

    m = EventMetrics(zone_days=n * step_min / 1440.0)
    seg_starts, seg_ends = segments(alarm, gap)
    m.n_segments = len(seg_starts)

    truth = events[events["is_anomaly"]]
    windows = [(pos(s), pos(e) + tol) for s, e in zip(truth["start"], truth["end"], strict=True)]
    kinds = [str(k) for k in truth["type"]]

    for (w0, w1), kind in zip(windows, kinds, strict=True):
        m.n_events += 1
        m.events_by_type[kind] += 1
        hits = [a for a, b in zip(seg_starts, seg_ends, strict=True) if _overlaps(a, b, w0, w1)]
        if hits:
            m.n_detected += 1
            m.detected_by_type[kind] += 1
            # delay counted from the true start; w0 is that start
            m.delays_min[kind].append(float(max(0, min(hits) - w0) * step_min))

    benign = events[~events["is_anomaly"]]
    b_windows = [
        (pos(s), pos(e) + benign_tol, str(k))
        for s, e, k in zip(benign["start"], benign["end"], benign["type"], strict=True)
    ]

    for a, b in zip(seg_starts, seg_ends, strict=True):
        if any(_overlaps(a, b, w0, w1) for w0, w1 in windows):
            m.n_true_segments += 1
            continue
        if not attribute_false:
            continue
        causes = {k for w0, w1, k in b_windows if _overlaps(a, b, w0, w1)}
        # a defrost is the more specific explanation than a routine door
        m.false_by_cause["defrost" if "defrost" in causes else (next(iter(causes), "unexplained"))] += 1
    return m


def evaluate(
    readings: pd.DataFrame,
    events: pd.DataFrame,
    alarm: np.ndarray,
    tol_min: int = 30,
    merge_gap: int = 5,
    min_run: int = 1,
    attribute_false: bool = True,
) -> EventMetrics:
    """Pool event metrics over all zones. `alarm` is aligned with `readings` rows.

    attribute_false=False skips the (slow) benign-cause breakdown; use it inside tuning loops."""
    total = EventMetrics()
    alarm = np.asarray(alarm, dtype=bool)
    for zone, idx in readings.groupby("zone").indices.items():
        z_alarm = apply_min_run(alarm[idx], min_run)
        z_events = events[events["zone"] == zone]
        total.merge(
            evaluate_zone(readings["ts"].iloc[idx], z_alarm, z_events, tol_min, merge_gap, attribute_false)
        )
    return total
