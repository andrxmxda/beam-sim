import numpy as np
import pytest

from beam import DistributedLoad, PointLoad, moment, reactions, shear

L = 6.0
P = 10_000.0
W = 2_000.0


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
