"""Synthetic freezer-sensor simulator with injected, labelled faults."""

from coldchain_simulator.config import SimConfig
from coldchain_simulator.generate import generate, point_labels

__all__ = ["SimConfig", "generate", "point_labels"]
