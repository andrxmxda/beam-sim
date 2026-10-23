import numpy as np
import pytest

from beam import (
    DistributedLoad,
    PointLoad,
    RectangularSection,
    analyze,
    deflection,
    moment,
    reactions,
    safety_rating,
    shear,
)
from materials import MATERIALS

L = 6.0
P = 10_000.0
W = 2_000.0
STEEL = MATERIALS["Structural steel (A36)"]
SECTION = RectangularSection(width=0.1, height=0.3)
EI = STEEL.elastic_modulus * SECTION.second_moment


def test_midspan_point_load():
    loads = [PointLoad(L / 2, P)]

    assert reactions(L, loads) == pytest.approx((P / 2, P / 2))
    assert moment(L / 2, L, loads) == pytest.approx(P * L / 4)
    assert shear(L / 2 - 1e-9, L, loads) == pytest.approx(P / 2)
    assert shear(L / 2 + 1e-9, L, loads) == pytest.approx(-P / 2)


def test_off_center_point_load():
    a, b = 2.0, L - 2.0
    loads = [PointLoad(a, P)]

    assert reactions(L, loads) == pytest.approx((P * b / L, P * a / L))
    assert moment(a, L, loads) == pytest.approx(P * a * b / L)


def test_full_span_udl():
    loads = [DistributedLoad(0.0, L, W)]

    assert reactions(L, loads) == pytest.approx((W * L / 2, W * L / 2))
    assert moment(L / 2, L, loads) == pytest.approx(W * L**2 / 8)
    assert shear(0.0, L, loads) == pytest.approx(W * L / 2)
    assert shear(L, L, loads) == pytest.approx(-W * L / 2)
    assert shear(L / 2, L, loads) == pytest.approx(0.0, abs=1e-9)


def test_partial_udl_reactions():
    start, end = 1.0, 3.0
    loads = [DistributedLoad(start, end, W)]
    total = W * (end - start)
    r_b = total * (start + end) / 2 / L

    assert reactions(L, loads) == pytest.approx((total - r_b, r_b))


def test_no_loads():
    x = np.linspace(0, L, 11)

    assert reactions(L, []) == (0.0, 0.0)
    assert np.all(shear(x, L, []) == 0)
    assert np.all(moment(x, L, []) == 0)


MIXED = [PointLoad(1.5, P), PointLoad(4.0, 0.5 * P), DistributedLoad(2.0, 5.0, W)]


def test_equilibrium_and_zero_end_moments():
    r_a, r_b = reactions(L, MIXED)
    total = P + 0.5 * P + W * 3.0

    assert r_a + r_b == pytest.approx(total)
    assert moment(0.0, L, MIXED) == pytest.approx(0.0, abs=1e-9)
    assert moment(L, L, MIXED) == pytest.approx(0.0, abs=1e-6)


def test_superposition():
    x = np.linspace(0, L, 101)
    combined = moment(x, L, MIXED)
    separate = sum(moment(x, L, [load]) for load in MIXED)

    np.testing.assert_allclose(combined, separate, atol=1e-6)


def test_shear_is_slope_of_moment():
    # Point loads land on grid points, so no interval straddles a shear jump,
    # and the midpoint of each interval is where the difference quotient is exact.
    x = np.linspace(0, L, 6001)
    midpoints = (x[:-1] + x[1:]) / 2
    slope = np.diff(moment(x, L, MIXED)) / np.diff(x)

    np.testing.assert_allclose(slope, shear(midpoints, L, MIXED), atol=1e-3)


def test_rectangular_section():
    section = RectangularSection(width=0.2, height=0.4)

    assert section.second_moment == pytest.approx(0.2 * 0.4**3 / 12)
    assert section.extreme_fiber == pytest.approx(0.2)


def test_midspan_point_load_deflection():
    result = analyze(L, [PointLoad(L / 2, P)], STEEL, SECTION)

    assert result.max_deflection == pytest.approx(-P * L**3 / (48 * EI), rel=1e-5)
    assert result.max_deflection_position == pytest.approx(L / 2, abs=L / 1000)


def test_full_span_udl_deflection():
    result = analyze(L, [DistributedLoad(0.0, L, W)], STEEL, SECTION)

    assert result.max_deflection == pytest.approx(-5 * W * L**4 / (384 * EI), rel=1e-5)
    assert result.max_moment == pytest.approx(W * L**2 / 8, rel=1e-6)


def test_off_center_point_load_deflection():
    # Textbook form puts the load a from the left, b from the right, with a > b.
    a, b = 4.0, 2.0
    result = analyze(L, [PointLoad(a, P)], STEEL, SECTION)
    expected = P * b * (L**2 - b**2) ** 1.5 / (9 * np.sqrt(3) * L * EI)

    assert result.max_deflection == pytest.approx(-expected, rel=1e-5)
    assert result.max_deflection_position == pytest.approx(
        np.sqrt((L**2 - b**2) / 3), abs=L / 1000
    )


def test_deflection_is_zero_at_supports():
    y = deflection([0.0, L], L, MIXED, STEEL.elastic_modulus, SECTION.second_moment)

    np.testing.assert_allclose(y, 0.0, atol=1e-12)


def test_deflection_scales_inversely_with_stiffness():
    aluminum = MATERIALS["Aluminum 6061-T6"]
    steel_result = analyze(L, MIXED, STEEL, SECTION)
    aluminum_result = analyze(L, MIXED, aluminum, SECTION)
    ratio = aluminum_result.max_deflection / steel_result.max_deflection

    assert ratio == pytest.approx(STEEL.elastic_modulus / aluminum.elastic_modulus)


def test_stress_and_safety_factor():
    result = analyze(L, [PointLoad(L / 2, P)], STEEL, SECTION)
    expected_stress = (P * L / 4) * SECTION.extreme_fiber / SECTION.second_moment

    assert result.max_stress == pytest.approx(expected_stress)
    assert result.safety_factor == pytest.approx(STEEL.strength / expected_stress)


def test_unloaded_beam_is_infinitely_safe():
    result = analyze(L, [], STEEL, SECTION)

    assert result.max_stress == 0
    assert result.safety_factor == float("inf")
    assert safety_rating(result.safety_factor) == "safe"


@pytest.mark.parametrize(
    "factor, rating",
    [(3.0, "safe"), (2.0, "safe"), (1.5, "marginal"), (1.0, "marginal"), (0.9, "failing")],
)
def test_safety_rating_thresholds(factor, rating):
    assert safety_rating(factor) == rating


def test_shear_diagram_has_vertical_step_at_point_load():
    result = analyze(L, [PointLoad(2.0, P)], STEEL, SECTION)
    i = int(np.searchsorted(result.x, 2.0))

    assert result.x[i] == 2.0
    assert result.shear[i] - result.shear[i + 1] == pytest.approx(P)
