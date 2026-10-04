import pytest

from mechdesign import FourBar, design_train
from mechdesign.scenarios import FOURBAR_SCENARIOS, GEAR_SCENARIOS


@pytest.mark.parametrize("name, kind", [
    ("Windshield wiper", "crank-rocker"),
    ("Quick-return shaper", "crank-rocker"),
    ("Toggle clamp (non-Grashof)", "triple-rocker (non-Grashof)"),
    ("Drag-link (double-crank)", "double-crank"),
])
def test_fourbar_scenarios_are_what_they_claim(name, kind):
    sc = FOURBAR_SCENARIOS[name]
    assert FourBar(sc["ground"], sc["crank"], sc["coupler"], sc["rocker"]).grashof()[1] == kind


def test_wiper_and_quick_return_numbers():
    w = FOURBAR_SCENARIOS["Windshield wiper"]
    assert FourBar(w["ground"], w["crank"], w["coupler"], w["rocker"]).summary()["output_swing_deg"] > 95
    q = FOURBAR_SCENARIOS["Quick-return shaper"]
    assert FourBar(q["ground"], q["crank"], q["coupler"], q["rocker"]).time_ratio() == pytest.approx(1.45, abs=0.01)


@pytest.mark.parametrize("name", list(GEAR_SCENARIOS))
def test_gear_scenarios_design_cleanly(name):
    sc = GEAR_SCENARIOS[name]
    target = sc["motor_rpm"] / sc["out_rpm"]
    g = design_train(target)
    assert g.ratio == pytest.approx(target, rel=0.001)
    assert not g.interference_problems()
