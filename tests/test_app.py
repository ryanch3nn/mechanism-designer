"""Smoke test: the web app runs every scenario and mode without raising."""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from mechdesign.scenarios import FOURBAR_SCENARIOS, GEAR_SCENARIOS  # noqa: E402


def run(**state):
    at = st_testing.AppTest.from_file("../app.py", default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("name", list(FOURBAR_SCENARIOS))
def test_fourbar_scenarios(name):
    sc = FOURBAR_SCENARIOS[name]
    run(fb_scenario=name, **{f"fb_{k}": float(sc[k]) if k != "branch" else sc[k]
                             for k in ("ground", "crank", "coupler", "rocker", "point_dist", "point_deg", "branch", "rpm")})


@pytest.mark.parametrize("name", list(GEAR_SCENARIOS))
def test_gear_scenarios(name):
    sc = GEAR_SCENARIOS[name]
    run(g_scenario=name, **{f"g_{k}": float(sc[k]) for k in ("motor_rpm", "motor_torque", "out_rpm", "load_torque")})


def test_bad_inputs_show_errors_not_crashes():
    at = run(fb_ground=500.0, g_mode="Check my own gears", g_meshes="nonsense")
    assert len(at.error) >= 2   # can't-assemble linkage + unparseable gears, and the page still renders


def test_check_mode_flags_interference():
    at = run(g_mode="Check my own gears", g_meshes="10:60")
    assert any("Mesh 1" in e.value for e in at.error)
