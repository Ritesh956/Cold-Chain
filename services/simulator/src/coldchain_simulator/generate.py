"""Offline generation of a synthetic sensor history plus its ground-truth event table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coldchain_simulator.config import SimConfig
from coldchain_simulator.faults import build_forcings
from coldchain_simulator.zone import simulate_zone

READING_COLUMNS = ["ts", "zone", "temp_c", "compressor_on", "compressor_current_a"]


def generate(cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (readings, events). Deterministic for a given cfg.seed.

    `readings` is what a detector may see. `events` is ground truth and must only be
    used for evaluation, never as a feature.
    """
    seeds = np.random.SeedSequence(cfg.seed).spawn(2 * len(cfg.zones))
    index = pd.date_range(cfg.start, periods=cfg.n_steps, freq="min")
    frames: list[pd.DataFrame] = []
    events: list[dict] = []

    for k, zone in enumerate(cfg.zones):
        forcings = build_forcings(cfg, zone, k, np.random.default_rng(seeds[2 * k]))
        trace = simulate_zone(cfg, forcings, np.random.default_rng(seeds[2 * k + 1]))
        frames.append(
            pd.DataFrame(
                {
                    "ts": index,
                    "zone": zone,
                    "temp_c": trace.temp_obs_c,
                    "compressor_on": trace.compressor_on,
                    "compressor_current_a": trace.current_a,
                }
            )
        )
        events.extend(e.model_dump(mode="json") for e in forcings.events)

    readings = pd.concat(frames, ignore_index=True).sort_values(["ts", "zone"], ignore_index=True)
    ev = pd.DataFrame(events)
    for col in ("start", "end"):
        ev[col] = pd.to_datetime(ev[col])
    ev = ev.sort_values(["zone", "start"], ignore_index=True)
    return readings[READING_COLUMNS], ev


def point_labels(readings: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Per-row boolean: is this reading inside an injected ANOMALY (benign events excluded)?"""
    labels = pd.Series(False, index=readings.index)
    anomalies = events[events["is_anomaly"]]
    for zone, grp in anomalies.groupby("zone"):
        in_zone = readings["zone"] == zone
        for start, end in zip(grp["start"], grp["end"], strict=True):
            labels |= in_zone & (readings["ts"] >= start) & (readings["ts"] < end)
    return labels


def write_dataset(cfg: SimConfig, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    readings, events = generate(cfg)
    r_path, e_path = out_dir / "readings.parquet", out_dir / "events.parquet"
    readings.to_parquet(r_path, index=False)
    events.to_parquet(e_path, index=False)
    return r_path, e_path
