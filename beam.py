"""Simply supported beam: pin at x = 0, roller at x = L.

SI units throughout (m, N, N/m, Pa). Downward loads are positive, sagging
moment is positive, and deflection is positive upward, so a loaded beam
has negative deflection.
"""

from dataclasses import dataclass

import numpy as np

from materials import Material


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


def load_problem(load: Load, length: float) -> str | None:
    """Return why the load can't be applied to this beam, or None if it can."""
    if isinstance(load, PointLoad):
        if not 0 <= load.position <= length:
            return f"Point load at {load.position:g} m is off the beam (span is {length:g} m)."
        return None
    where = f"Distributed load from {load.start:g} to {load.end:g} m"
    if load.end <= load.start:
        return f"{where} has to end after it starts."
    if load.start < 0 or load.end > length:
        return f"{where} runs past the beam (span is {length:g} m)."
    return None


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


@dataclass(frozen=True)
class RectangularSection:
    width: float
    height: float

    @property
    def second_moment(self) -> float:
        return self.width * self.height**3 / 12

    @property
    def extreme_fiber(self) -> float:
        return self.height / 2


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    steps = (y[1:] + y[:-1]) / 2 * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(steps)))


def deflection(
    x,
    length: float,
    loads: list[Load],
    elastic_modulus: float,
    second_moment: float,
    samples: int = 2001,
) -> np.ndarray:
    """Solve EI y'' = M(x) with y(0) = y(L) = 0."""
    grid = np.linspace(0.0, length, samples)
    curvature = moment(grid, length, loads) / (elastic_modulus * second_moment)
    y = _cumulative_trapezoid(_cumulative_trapezoid(curvature, grid), grid)
    # The double integral already satisfies y(0) = 0. Subtracting a straight
    # line through its end value picks the slope constant that makes y(L) = 0.
    y -= y[-1] * grid / length
    return np.interp(x, grid, y)


def max_bending_stress(max_moment: float, section: RectangularSection) -> float:
    return abs(max_moment) * section.extreme_fiber / section.second_moment


def safety_factor(strength: float, stress: float) -> float:
    return strength / stress if stress > 0 else float("inf")


def safety_rating(factor: float) -> str:
    if factor >= 2:
        return "safe"
    if factor >= 1:
        return "marginal"
    return "failing"


@dataclass(frozen=True)
class Analysis:
    x: np.ndarray
    shear: np.ndarray
    moment: np.ndarray
    deflection: np.ndarray
    reactions: tuple[float, float]
    max_moment: float
    max_moment_position: float
    max_deflection: float
    max_deflection_position: float
    max_stress: float
    safety_factor: float


def analyze(
    length: float,
    loads: list[Load],
    material: Material,
    section: RectangularSection,
    samples: int = 2001,
) -> Analysis:
    if length <= 0:
        raise ValueError("Span must be positive.")
    if section.width <= 0 or section.height <= 0:
        raise ValueError("Section width and height must be positive.")
    for load in loads:
        if problem := load_problem(load, length):
            raise ValueError(problem)

    point_positions = [ld.position for ld in loads if isinstance(ld, PointLoad)]
    # Each point load gets a sample on both sides of its shear jump so the
    # plotted diagram shows a vertical step instead of a slanted line.
    just_after = [np.nextafter(p, np.inf) for p in point_positions]
    x = np.linspace(0.0, length, samples)
    x = np.unique(np.clip(np.concatenate((x, point_positions, just_after)), 0.0, length))

    v = shear(x, length, loads)
    m = moment(x, length, loads)
    y = deflection(x, length, loads, material.elastic_modulus, section.second_moment, samples)

    i_m = int(np.argmax(np.abs(m)))
    i_y = int(np.argmax(np.abs(y)))
    stress = max_bending_stress(m[i_m], section)

    return Analysis(
        x=x,
        shear=v,
        moment=m,
        deflection=y,
        reactions=reactions(length, loads),
        max_moment=float(m[i_m]),
        max_moment_position=float(x[i_m]),
        max_deflection=float(y[i_y]),
        max_deflection_position=float(x[i_y]),
        max_stress=stress,
        safety_factor=safety_factor(material.strength, stress),
    )
