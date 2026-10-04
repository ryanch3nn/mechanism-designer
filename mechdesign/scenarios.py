"""Ready-made design scenarios: real machines to start from, each with what to look for.

Four-bar inputs are link lengths in mm plus a coupler point (distance along AB in mm,
offset angle in degrees). Gear inputs are motor speed/torque and the output speed wanted.
"""

FOURBAR_SCENARIOS = {
    "Custom": dict(
        ground=70, crank=20, coupler=60, rocker=50, point_dist=40, point_deg=35, branch=1, rpm=60,
        story="Start from a balanced crank-rocker and change anything you like.",
        notice="Watch the checks update as you drag the sliders."),
    "Windshield wiper": dict(
        ground=100, crank=15, coupler=100, rocker=20, point_dist=50, point_deg=0, branch=1, rpm=45,
        story="A motor turns a short crank continuously; the rocker swings the wiper arm back and forth.",
        notice="A big ~98° swing from a tiny crank, and almost equal speed in both directions "
               "(time ratio ≈ 1). The price is a transmission angle just under 40° at one end."),
    "Quick-return shaper": dict(
        ground=90, crank=30, coupler=60, rocker=100, point_dist=60, point_deg=0, branch=1, rpm=30,
        story="Metal shapers cut on the slow stroke and race back on the return stroke to save time.",
        notice="Time ratio ≈ 1.45: the cutting stroke takes about 45 % longer than the return."),
    "Walking robot leg (Hoekens)": dict(
        ground=40, crank=20, coupler=50, rocker=50, point_dist=100, point_deg=0, branch=1, rpm=40,
        story="Hoekens' straight-line linkage. The foot is the coupler point, extended past joint B.",
        notice="The bottom of the coupler curve is almost perfectly straight for about 220° of crank "
               "rotation. That flat part is the foot on the ground, so the body moves without bobbing. "
               "The trade-off is a low transmission angle (~23°): fine for a light toy robot, "
               "not for a heavy machine."),
    "Toggle clamp (non-Grashof)": dict(
        ground=50, crank=30, coupler=40, rocker=35, point_dist=20, point_deg=0, branch=1, rpm=20,
        story="No link can turn all the way round, so the input rocks between two limit positions.",
        notice="Near the limits the torque amplification shoots up: a small push on the input holds "
               "a huge force. That is exactly how a toggle clamp locks a workpiece."),
    "Drag-link (double-crank)": dict(
        ground=20, crank=40, coupler=50, rocker=45, point_dist=25, point_deg=30, branch=1, rpm=60,
        story="The ground is the shortest link, so both cranks rotate fully. Used in presses and feeders.",
        notice="The output turns continuously but at a varying speed: it speeds up and slows down "
               "every turn even though the input turns steadily."),
}

GEAR_SCENARIOS = {
    "Custom": dict(
        motor_rpm=3000, motor_torque=0.25, out_rpm=100, load_torque=0.0,
        story="Pick a motor and the output speed you want.",
        notice=""),
    "Robot arm joint": dict(
        motor_rpm=3000, motor_torque=0.25, out_rpm=30, load_torque=20.0,
        story="A small brushless motor driving a shoulder joint that has to lift the arm.",
        notice="A 100:1 reduction needs three stages, and efficiency losses add up across them. "
               "Check that the output torque beats the 20 N·m the arm needs."),
    "Conveyor belt drive": dict(
        motor_rpm=1450, motor_torque=5.0, out_rpm=58, load_torque=100.0,
        story="A standard 4-pole induction motor (1450 rpm) driving a conveyor drum.",
        notice="Industrial motors run at fixed speeds, so the gearbox sets the belt speed."),
    "Electric winch": dict(
        motor_rpm=4000, motor_torque=1.2, out_rpm=40, load_torque=100.0,
        story="A winch drum has to pull a heavy load slowly.",
        notice="Torque goes up by roughly the same factor that speed goes down, minus losses."),
    "Clock (minute → hour hand)": dict(
        motor_rpm=1 / 60, motor_torque=0.001, out_rpm=1 / 720, load_torque=0.0,
        story="The minute hand turns once an hour, and the hour hand once every 12 hours.",
        notice="A 12:1 ratio fits in two small stages. Real clocks use exactly this kind of "
               "'motion work'."),
    "Wind turbine (speed increaser)": dict(
        motor_rpm=20, motor_torque=50_000.0, out_rpm=1500, load_torque=0.0,
        story="Slow, huge-torque blades have to spin a generator at 1500 rpm.",
        notice="The ratio is below 1, so the designer flips the train and the output spins 75 times "
               "faster. Real turbines use planetary stages for this."),
}
