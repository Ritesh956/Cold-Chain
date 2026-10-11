"""Prediction drift: has the distribution of a model's output scores moved?

The ml-api exports each score histogram as cumulative counters. This tracker keeps timestamped
snapshots of them, freezes the first `baseline_count` observations as the baseline, and compares
the observations of the last `window_s` seconds against it with PSI over the histogram buckets.

Limitation, stated plainly: the baseline is the model's own early behaviour after this monitor
started, not its behaviour on the training data. It detects "the scores changed since deployment",
not "the scores differ from validation". A counter reset (ml-api restart) restarts the baseline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from coldchain_monitor.drift import psi_from_counts
from coldchain_monitor.prom import per_bucket


@dataclass
class ScoreDrift:
    psi: float | None
    n_baseline: int
    n_recent: int
    state: str  # "ok" | "warn" | "drift" | "building_baseline" | "insufficient_data"


@dataclass
class ScoreTracker:
    baseline_count: int
    window_s: float
    min_count: int
    warn: float
    drift: float
    _bounds: list[float] = field(default_factory=list)
    _baseline: list[float] | None = None
    _snaps: deque = field(default_factory=deque)  # (t, per-bucket counts since the exporter started)

    def observe(self, now: float, bounds: list[float], cumulative: list[float]) -> ScoreDrift:
        counts = per_bucket(cumulative)
        total = sum(counts)
        if self._snaps and (bounds != self._bounds or total < sum(self._snaps[-1][1])):
            # the exporter restarted or its buckets changed: the old snapshots are meaningless
            self._baseline = None
            self._snaps.clear()
        self._bounds = list(bounds)
        self._snaps.append((now, counts))
        while len(self._snaps) > 1 and self._snaps[1][0] <= now - self.window_s:
            self._snaps.popleft()

        if self._baseline is None:
            if total < self.baseline_count:
                return ScoreDrift(None, int(total), 0, "building_baseline")
            self._baseline = counts
        base = self._baseline
        oldest = self._snaps[0][1]
        # recent = what arrived since the window start, but never counting from inside the baseline
        floor = base if sum(oldest) < sum(base) else oldest
        recent = [max(c - f, 0.0) for c, f in zip(counts, floor, strict=True)]
        n_recent = int(sum(recent))
        if n_recent < self.min_count:
            return ScoreDrift(None, int(sum(base)), n_recent, "insufficient_data")
        value = psi_from_counts([b / sum(base) for b in base], recent)
        state = "drift" if value >= self.drift else "warn" if value >= self.warn else "ok"
        return ScoreDrift(value, int(sum(base)), n_recent, state)
