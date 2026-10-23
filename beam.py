"""Statics for a simply supported beam: pin at x = 0, roller at x = L.

SI units throughout (m, N, N/m). Downward loads are positive. Sagging
moment is positive.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointLoad:
    position: float
    magnitude: float


@dataclass(frozen=True)
class DistributedLoad:
    start: float
    end: float
    intensity: float

    @property
    def total(self) -> float:
        return self.intensity * (self.end - self.start)

    @property
    def centroid(self) -> float:
        return (self.start + self.end) / 2


Load = PointLoad | DistributedLoad


def _resultant(load: Load) -> tuple[float, float]:
    """Return (force, position) of the single force equivalent to the load."""
    if isinstance(load, PointLoad):
        return load.magnitude, load.position
    return load.total, load.centroid


def reactions(length: float, loads: list[Load]) -> tuple[float, float]:
    """Return (R_A, R_B), the upward support forces at the left and right ends."""
    total = 0.0
    moment_about_a = 0.0
    for load in loads:
        force, position = _resultant(load)
        total += force
        moment_about_a += force * position
    r_b = moment_about_a / length
    return total - r_b, r_b


def _covered(x: np.ndarray, load: DistributedLoad) -> np.ndarray:
    """Length of the UDL lying to the left of each x."""
    return np.clip(x, load.start, load.end) - load.start


def shear(x, length: float, loads: list[Load]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r_a, _ = reactions(length, loads)
    v = np.full(x.shape, r_a)
    for load in loads:
        if isinstance(load, PointLoad):
            # A load exactly at x is counted on the right side of the jump.
            v -= np.where(x > load.position, load.magnitude, 0.0)
        else:
            v -= load.intensity * _covered(x, load)
    return v


def moment(x, length: float, loads: list[Load]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r_a, _ = reactions(length, loads)
    m = r_a * x
    for load in loads:
        if isinstance(load, PointLoad):
            m -= load.magnitude * np.maximum(x - load.position, 0.0)
        else:
            covered = _covered(x, load)
            lever_arm = x - (load.start + covered / 2)
            m -= load.intensity * covered * lever_arm
    return m
