"""Design a gearbox: a 3000 rpm motor driving a 100 rpm output (30:1 reduction)."""
from pathlib import Path

from mechdesign import GearTrain, design_train, plot_train

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

motor_rpm, motor_torque = 3000, 0.25      # rpm, N·m
target = motor_rpm / 100

train = design_train(target, module=1.0, efficiency=0.98)
print(f"Target ratio {target}:1\n")
print(train.report(motor_rpm, motor_torque))
plot_train(train, OUT / "gear_train.png", speed_in_rpm=motor_rpm)

print("\nWhat a bad design looks like:")
print(GearTrain([(10, 60)]).report(motor_rpm, motor_torque))
