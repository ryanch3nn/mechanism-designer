import pytest

from mechdesign import GearTrain, design_train, min_pinion_teeth


def test_compound_train_hand_check():
    """20T->60T then 15T->45T: ratio 3 x 3 = 9, two external meshes keep direction."""
    g = GearTrain([(20, 60), (15, 45)], module=1.5, efficiency=0.98)
    assert g.ratio == pytest.approx(9)
    assert g.direction == 1
    n_out, t_out = g.output(1800, 2.0)
    assert n_out == pytest.approx(200)
    assert t_out == pytest.approx(2.0 * 9 * 0.98**2)
    assert g.center_distances() == pytest.approx([60, 45])  # m (N1 + N2) / 2


def test_idler_cancels_and_reverses():
    g = GearTrain([(20, 35), (35, 40)])
    assert g.ratio == pytest.approx(2)        # idler teeth cancel
    assert g.direction == 1                    # two external meshes


def test_internal_ring_keeps_direction():
    g = GearTrain([(20, 80)], internal=[0])
    assert g.direction == 1
    assert g.center_distances() == pytest.approx([30])


@pytest.mark.parametrize("ratio, expected", [
    (None, 18),  # rack: 2 / sin^2(20 deg) = 17.1
    (1, 13),     # equal gears: 12.3 (Shigley)
    (4, 16),     # 15.4
])
def test_min_pinion_teeth(ratio, expected):
    assert min_pinion_teeth(ratio) == expected


def test_interference_flagged():
    assert GearTrain([(12, 60)]).interference_problems()
    assert not GearTrain([(18, 54)]).interference_problems()


@pytest.mark.parametrize("target", [3, 12.5, 37, 100, 0.25])
def test_designer_hits_target(target):
    g = design_train(target, tol=0.001)
    assert g.ratio == pytest.approx(target, rel=0.001)
    assert not g.interference_problems()


def test_designer_uses_fewest_stages():
    assert len(design_train(4).meshes) == 1
    assert len(design_train(20).meshes) == 2
