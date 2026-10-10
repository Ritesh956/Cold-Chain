"""Simulation parameters. Everything the simulator produces is SYNTHETIC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class SimConfig:
    seed: int = 42
    zones: tuple[str, ...] = ("A", "B", "C", "D")
    start: datetime = datetime(2025, 1, 1, tzinfo=UTC)
    days: int = 90

    # --- thermal model (all per-minute, degC) -------------------------------
    setpoint_c: float = -25.0
    hysteresis_c: float = 1.0  # compressor on at setpoint+h, off at setpoint-h
    ambient_mean_c: float = 10.0  # air outside the freezer envelope
    ambient_amp_c: float = 4.0  # daily swing, peaking ~15:00
    leak_k: float = 0.0016 # envelope heat leak coefficient
    cool_rate_c: float = 0.12  # cooling at full compressor efficiency
    dock_temp_c: float = 5.0  # air that rushes in when a door opens
    door_k: float = 0.02  # heat exchange when the door is fully open
    process_noise_c: float = 0.01
    sensor_noise_c: float = 0.15
    dropout_prob: float = 0.001  # sensor returns no value

    # --- benign periodic / routine behaviour (hard negatives) ---------------
    defrost_every_min: int = 480
    defrost_len_min: int = 20
    defrost_heat_c: float = 0.08  # evaporator heater, degC/min
    routine_doors_per_hour: float = 3.0  # 06:00-22:00 forklift traffic
    routine_door_intensity: float = 0.3  # strip curtains: partial exchange

    # --- injected anomalies, per zone over the whole horizon ----------------
    n_door_left_open: int = 5
    n_power_cut: int = 4
    n_compressor_drift: int = 2
    door_open_min: tuple[int, int] = (12, 35)  # minutes
    power_cut_min: tuple[int, int] = (60, 180)  # minutes
    drift_days: tuple[int, int] = (3, 6)
    drift_final_efficiency: float = 0.45
    anomaly_guard_min: int = 720  # quiet gap between anomalies in one zone

    # --- compressor current (amps) ------------------------------------------
    current_on_a: float = 12.0
    current_noise_a: float = 0.3

    @property
    def n_steps(self) -> int:
        return self.days * MINUTES_PER_DAY
