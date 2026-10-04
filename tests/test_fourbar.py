import numpy as np
import pytest

from mechdesign import FourBar


def test_loop_closure():
    """Every solved position must keep each link at its true length."""
    fb = FourBar(7, 2, 6, 5, coupler_point=(3, 0.5))
    for th2 in np.linspace(0, 2 * np.pi, 73):
        p = fb.position(th2)
        assert np.hypot(*(p["B"] - p["A"])) == pytest.approx(6)
        assert np.hypot(*(p["B"] - p["O4"])) == pytest.approx(5)


def test_parallelogram_hand_check():
    """Open parallelogram: rocker copies the crank, coupler only translates."""
    fb = FourBar(4, 2, 4, 2)
    th2 = np.radians(60)
    assert fb.position(th2)["th4"] == pytest.approx(th2)
    w3, w4 = fb.velocity(th2, w2=10)
    assert w3 == pytest.approx(0, abs=1e-9)
    assert w4 == pytest.approx(10)


def test_velocity_matches_finite_difference():
    fb = FourBar(7, 2, 6, 5)
    th2, h = 1.1, 1e-6
    dth4 = (fb.position(th2 + h)["th4"] - fb.position(th2 - h)["th4"]) / (2 * h)
    assert fb.velocity(th2, w2=1)[1] == pytest.approx(dth4, rel=1e-5)


def test_transmission_angle_hand_check():
    """At th2 = 0 the diagonal is r1 - r2 = 5, so cos(mu) = (6^2 + 5^2 - 5^2) / (2*6*5) = 0.6."""
    fb = FourBar(7, 2, 6, 5)
    assert np.degrees(fb.position(0)["mu"]) == pytest.approx(53.130, abs=1e-3)


@pytest.mark.parametrize("links, kind", [
    ((7, 2, 6, 5), "crank-rocker"),
    ((2, 7, 6, 5), "double-crank"),
    ((7, 6, 2, 5), "double-rocker"),
    ((4, 5, 6, 9), "triple-rocker (non-Grashof)"),
    ((4, 5, 6, 7), "change-point"),  # 4 + 7 == 5 + 6
    ((4, 2, 4, 2), "change-point"),
])
def test_grashof_classification(links, kind):
    assert FourBar(*links).grashof()[1] == kind


def test_non_grashof_cannot_fully_rotate():
    fb = FourBar(4, 5, 6, 9)
    assert not fb.summary()["full_rotation"]
    assert fb.position(0) is None  # crank along the ground: diagonal 1 < |6 - 9|, coupler and rocker cannot meet


def test_time_ratio_matches_sweep():
    """Closed-form Q must match the crank angles where the swept rocker actually reverses."""
    fb = FourBar(90, 30, 60, 100)
    s = fb.sweep(n=36001)
    th4 = s["th4"]
    fwd = (s["th2"][np.argmin(th4)] - s["th2"][np.argmax(th4)]) % (2 * np.pi)
    q = max(fwd, 2 * np.pi - fwd) / min(fwd, 2 * np.pi - fwd)
    assert fb.time_ratio() == pytest.approx(q, rel=1e-3)
    assert FourBar(2, 7, 6, 5).time_ratio() is None   # only defined for crank-rockers
