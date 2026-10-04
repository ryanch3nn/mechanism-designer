"""Spur gear trains: ratios, speed/torque, tooth interference and automatic tooth-count design.

A train is a list of meshes (driver_teeth, driven_teeth). In a compound train the driven
gear of one mesh shares a shaft with the driver of the next. An idler gear shows up as the
driven gear of one mesh and the driver of the next, so its tooth count cancels out.
"""
from dataclasses import dataclass, field

import numpy as np


def min_pinion_teeth(ratio=None, pressure_angle_deg=20.0, k=1.0):
    """Fewest pinion teeth that avoid interference (full-depth spur gears, Shigley eqs. 13-10/13-11).

    ratio = gear teeth / pinion teeth (>= 1). ratio=None means the pinion meshes with a rack.
    """
    s2 = np.sin(np.radians(pressure_angle_deg)) ** 2
    if ratio is None:
        n = 2 * k / s2
    else:
        m = ratio
        n = 2 * k / ((1 + 2 * m) * s2) * (m + np.sqrt(m**2 + (1 + 2 * m) * s2))
    return int(np.ceil(n - 1e-9))


@dataclass
class GearTrain:
    meshes: list                       # [(driver_teeth, driven_teeth), ...]
    module: float = 1.0                # mm (pitch diameter = module * teeth)
    efficiency: float = 0.98           # per mesh, typical for spur gears
    pressure_angle_deg: float = 20.0
    internal: list = field(default_factory=list)  # indexes of meshes with an internal (ring) gear

    @property
    def ratio(self):
        """Speed reduction n_in / n_out (> 1 means the output turns slower)."""
        return float(np.prod([b / a for a, b in self.meshes]))

    @property
    def direction(self):
        """+1 if the output turns the same way as the input, -1 if reversed."""
        external = len(self.meshes) - len(self.internal)
        return -1 if external % 2 else 1

    def output(self, speed_in_rpm, torque_in_Nm):
        n_out = self.direction * speed_in_rpm / self.ratio
        t_out = torque_in_Nm * self.ratio * self.efficiency ** len(self.meshes)
        return n_out, t_out

    def center_distances(self):
        return [self.module * (b - a if i in self.internal else a + b) / 2
                for i, (a, b) in enumerate(self.meshes)]

    def interference_problems(self):
        """List external meshes whose smaller gear has too few teeth."""
        bad = []
        for i, (a, b) in enumerate(self.meshes):
            if i in self.internal:
                continue
            small, big = sorted((a, b))
            need = min_pinion_teeth(big / small, self.pressure_angle_deg)
            if small < need:
                bad.append((i, small, need))
        return bad

    def report(self, speed_in_rpm=1000.0, torque_in_Nm=1.0):
        n_out, t_out = self.output(speed_in_rpm, torque_in_Nm)
        lines = [f"Meshes: {self.meshes}  (module {self.module} mm)",
                 f"Ratio: {self.ratio:.4f} : 1   direction: {'same' if self.direction > 0 else 'reversed'}",
                 f"Input  {speed_in_rpm:8.1f} rpm  {torque_in_Nm:8.3f} N·m",
                 f"Output {n_out:8.1f} rpm  {t_out:8.3f} N·m  (efficiency {self.efficiency**len(self.meshes):.1%})",
                 f"Center distances (mm): {[round(c, 2) for c in self.center_distances()]}"]
        bad = self.interference_problems()
        lines.append("Interference: OK" if not bad else
                     "Interference: " + "; ".join(f"mesh {i}: {n}T < {need}T min" for i, n, need in bad))
        return "\n".join(lines)


def _valid_pairs(min_teeth, max_teeth, pressure_angle_deg):
    """All (driver, driven) reduction pairs with driver <= driven and no interference."""
    pairs = [(a, b) for a in range(min_teeth, max_teeth + 1) for b in range(a, max_teeth + 1)
             if a >= min_pinion_teeth(b / a, pressure_angle_deg)]
    return np.array(pairs)


def _unique_by_ratio(ratio, teeth, payload):
    """Sort by ratio and keep only the fewest-teeth option for each distinct ratio."""
    key = np.round(ratio, 10)
    order = np.lexsort((teeth, key))
    keep = np.r_[True, np.diff(key[order]) != 0]
    idx = order[keep]
    return ratio[idx], teeth[idx], payload[idx]


def _search(r, t, target, tol):
    """In a ratio-sorted table, return (index, rel_error): fewest teeth within tol, else nearest."""
    lo = np.searchsorted(r, target * (1 - tol))
    hi = np.searchsorted(r, target * (1 + tol), side="right")
    if hi > lo:
        i = lo + int(np.argmin(t[lo:hi]))
    else:
        near = [j for j in (lo - 1, lo) if 0 <= j < len(r)]
        i = min(near, key=lambda j: abs(r[j] / target - 1))
    return i, abs(r[i] / target - 1)


def design_train(target_ratio, stages=None, min_teeth=12, max_teeth=80,
                 pressure_angle_deg=20.0, tol=0.001, **train_kwargs):
    """Pick tooth counts for a 1-, 2- or 3-stage compound train that hits target_ratio.

    Tries the fewest stages first and stops at the first one within `tol` (relative error).
    Among solutions within tolerance it picks the fewest total teeth (smaller, lighter gears).
    If nothing is within tolerance, returns the most accurate design found.
    Ratios below 1 (speed increasers) are designed as reducers and then flipped.
    """
    flip = target_ratio < 1
    target = 1 / target_ratio if flip else target_ratio

    pairs = _valid_pairs(min_teeth, max_teeth, pressure_angle_deg)
    r1, t1, p1 = _unique_by_ratio(pairs[:, 1] / pairs[:, 0], pairs.sum(axis=1), pairs)
    n1 = len(r1)
    # two-stage table: every combination of two single stages, stored as index pairs into p1
    combo = np.stack(np.divmod(np.arange(n1 * n1), n1), axis=1)
    r2, t2, c2 = _unique_by_ratio(np.outer(r1, r1).ravel(), np.add.outer(t1, t1).ravel(), combo)

    def solve(n):
        if n == 1:
            i, e = _search(r1, t1, target, tol)
            return e, t1[i], [p1[i]]
        if n == 2:
            i, e = _search(r2, t2, target, tol)
            return e, t2[i], [p1[c2[i, 0]], p1[c2[i, 1]]]
        best = None                      # n == 3: first stage from r1, other two from r2
        for k in range(n1):
            i, e = _search(r2, t2, target / r1[k], tol)
            score = (e > tol, e if e > tol else 0, t1[k] + t2[i])
            if best is None or score < best[0]:
                best = (score, e, t1[k] + t2[i], [p1[k], p1[c2[i, 0]], p1[c2[i, 1]]])
        return best[1:]

    results = []
    for n in ([stages] if stages else [1, 2, 3]):
        results.append(solve(n))
        if results[-1][0] <= tol:
            break
    _, _, meshes = results[-1] if results[-1][0] <= tol else min(results, key=lambda res: res[0])
    meshes = [tuple(int(x) for x in m) for m in meshes]
    if flip:
        meshes = [(b, a) for a, b in reversed(meshes)]
    return GearTrain(meshes, pressure_angle_deg=pressure_angle_deg, **train_kwargs)


# ---------------------------------------------------------------- drawing
def _gear_outline(teeth, module, phase):
    """Approximate tooth profile (good for diagrams, not for manufacturing)."""
    th = np.linspace(0, 2 * np.pi, teeth * 24 + 1)
    rp = module * teeth / 2
    wave = np.clip(1.8 * np.sin(teeth * (th - phase)), -1, 1)
    r = rp + np.where(wave > 0, module, 1.25 * module) * wave
    return r * np.cos(th), r * np.sin(th)


def plot_train(train, path, speed_in_rpm=None):
    """Front-view schematic: meshes run left to right, gears on one shaft are drawn concentric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#1f6feb", "#f0883e", "#2da44e", "#8250df", "#d1242f"]
    from matplotlib.colors import to_rgb

    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    x, m = 0.0, train.module
    for i, (a, b) in enumerate(train.meshes):
        cd = m * (a + b) / 2
        c = np.array(to_rgb(colors[i % len(colors)]))
        tint = 0.3 * c + 0.7  # solid light fill so each later stage reads as sitting in front
        # driver tooth points at the mesh (angle 0), driven gear has a gap there (angle pi)
        for teeth, cx, phase, z in ((a, x, -np.pi / (2 * a), 2 * i + 2),
                                    (b, x + cd, np.pi + np.pi / (2 * b), 2 * i + 1)):
            gx, gy = _gear_outline(teeth, m, phase)
            ax.fill(gx + cx, gy, color=tint, zorder=z)
            ax.plot(gx + cx, gy, color=c, lw=0.9, zorder=z)
        ax.text(x, -m * a / 2 - 3 * m, f"{a}T", ha="center", va="top", fontsize=9, zorder=20)
        ax.text(x + cd, m * b / 2 + 2 * m, f"{b}T", ha="center", fontsize=9, zorder=20)
        ax.plot([x], [0], "k+", ms=9, zorder=20)
        x += cd
    ax.plot([x], [0], "k+", ms=9, zorder=20)
    title = f"Gear train  {' → '.join(f'{a}:{b}' for a, b in train.meshes)}   ratio {train.ratio:.3f}:1"
    if speed_in_rpm is not None:
        n_out, _ = train.output(speed_in_rpm, 1)
        title += f"\n{speed_in_rpm:.0f} rpm in → {abs(n_out):.1f} rpm out"
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
