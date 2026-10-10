"""First-order thermal model of one freezer zone with a thermostat-controlled compressor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from coldchain_simulator.config import MINUTES_PER_DAY, SimConfig
from coldchain_simulator.faults import Forcings


@dataclass(frozen=True)
class ZoneTrace:
    temp_true_c: NDArray[np.float64]
    temp_obs_c: NDArray[np.float64]  # NaN where the sensor dropped out
    compressor_on: NDArray[np.bool_]
    current_a: NDArray[np.float64]


def simulate_zone(cfg: SimConfig, f: Forcings, rng: np.random.Generator) -> ZoneTrace:
    """dT = leak + door exchange - cooling + defrost heater + noise, 1 minute per step.

    The compressor is a hysteresis thermostat; it is forced off during power cuts and
    defrost. Reduced `efficiency` lowers cooling and raises current draw.
    """
    n = cfg.n_steps
    minute = np.arange(n) % MINUTES_PER_DAY
    # sinusoid peaking at 15:00
    ambient = cfg.ambient_mean_c + cfg.ambient_amp_c * np.sin(2 * np.pi * (minute - 540) / MINUTES_PER_DAY)
    noise = rng.normal(0.0, cfg.process_noise_c, n)

    temp = np.empty(n)
    on_arr = np.empty(n, dtype=bool)
    t = cfg.setpoint_c
    on = False
    hi = cfg.setpoint_c + cfg.hysteresis_c
    lo = cfg.setpoint_c - cfg.hysteresis_c

    for i in range(n):
        if f.power_off[i] or f.defrost[i]:
            on = False
        elif t >= hi:
            on = True
        elif t <= lo:
            on = False

        dt = cfg.leak_k * (ambient[i] - t)
        dt += cfg.door_k * f.door_frac[i] * (cfg.dock_temp_c - t)
        if on:
            dt -= cfg.cool_rate_c * f.efficiency[i]
        if f.defrost[i]:
            dt += cfg.defrost_heat_c
        t += dt + noise[i]
        temp[i] = t
        on_arr[i] = on

    obs = temp + rng.normal(0.0, cfg.sensor_noise_c, n)
    obs[rng.random(n) < cfg.dropout_prob] = np.nan

    # A struggling compressor draws more current for the same duty.
    on_current = cfg.current_on_a * (1.0 + 0.6 * (1.0 - f.efficiency))
    current = np.where(on_arr, on_current, 0.3) + rng.normal(0.0, cfg.current_noise_a, n)
    current = np.clip(current, 0.0, None)
    return ZoneTrace(temp, obs, on_arr, current)
