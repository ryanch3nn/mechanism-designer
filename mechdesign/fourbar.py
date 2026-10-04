"""Four-bar linkage kinematics: position, velocity, transmission angle, coupler curves.

Convention (all angles in radians, measured CCW from the +x axis):

    O2 = (0, 0)  ground pivot of the input crank
    O4 = (r1, 0) ground pivot of the output rocker
    r1 = ground, r2 = input crank, r3 = coupler, r4 = output rocker

    Vector loop:  r2 e^{i th2} + r3 e^{i th3} - r4 e^{i th4} - r1 = 0
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FourBar:
    ground: float
    crank: float
    coupler: float
    rocker: float
    coupler_point: tuple = (0.0, 0.0)  # (distance from joint A along coupler, offset angle in rad)
    branch: int = 1                     # +1 = open configuration, -1 = crossed

    @property
    def links(self):
        return np.array([self.ground, self.crank, self.coupler, self.rocker])

    # ---------- classification ----------
    def grashof(self):
        """Return (is_grashof, type) using s + l <= p + q."""
        s, p, q, l = np.sort(self.links)
        total = s + l - (p + q)
        if np.isclose(total, 0):
            return True, "change-point"
        if total > 0:
            return False, "triple-rocker (non-Grashof)"
        shortest = int(np.argmin(self.links))
        return True, ["double-crank", "crank-rocker", "double-rocker", "rocker-crank"][shortest]

    # ---------- position ----------
    def position(self, th2):
        """Joint positions for input angle th2. Returns dict, or None if the linkage can't assemble."""
        O2 = np.array([0.0, 0.0])
        O4 = np.array([self.ground, 0.0])
        A = self.crank * np.array([np.cos(th2), np.sin(th2)])

        # B is where a circle of radius r3 about A meets a circle of radius r4 about O4
        d_vec = O4 - A
        d = np.hypot(*d_vec)
        r3, r4 = self.coupler, self.rocker
        if d > r3 + r4 or d < abs(r3 - r4) or d == 0:
            return None
        a = (r3**2 - r4**2 + d**2) / (2 * d)
        h = np.sqrt(max(r3**2 - a**2, 0.0))
        u = d_vec / d
        perp = np.array([-u[1], u[0]])
        B = A + a * u + self.branch * h * perp

        th3 = np.arctan2(*(B - A)[::-1])
        th4 = np.arctan2(*(B - O4)[::-1])
        p, alpha = self.coupler_point
        P = A + p * np.array([np.cos(th3 + alpha), np.sin(th3 + alpha)])

        # transmission angle: angle between coupler and rocker at B (law of cosines)
        mu = np.arccos(np.clip((r3**2 + r4**2 - d**2) / (2 * r3 * r4), -1, 1))
        return dict(O2=O2, A=A, B=B, O4=O4, P=P, th3=th3, th4=th4, mu=mu)

    # ---------- velocity ----------
    def velocity(self, th2, w2=1.0):
        """Angular velocities (w3, w4) from the time derivative of the vector loop."""
        pos = self.position(th2)
        if pos is None:
            return None
        th3, th4 = pos["th3"], pos["th4"]
        r2, r3, r4 = self.crank, self.coupler, self.rocker
        M = np.array([[-r3 * np.sin(th3), r4 * np.sin(th4)],
                      [ r3 * np.cos(th3), -r4 * np.cos(th4)]])
        rhs = r2 * w2 * np.array([np.sin(th2), -np.cos(th2)])
        if abs(np.sin(th3 - th4)) < 1e-9:   # toggle: coupler and rocker in line, velocity undefined
            return np.array([np.nan, np.nan])
        return np.linalg.solve(M, rhs)

    # ---------- full sweep ----------
    def sweep(self, n=361, w2=1.0):
        """Analyze the linkage over a full turn of the crank. Unreachable angles are NaN."""
        th2 = np.linspace(0, 2 * np.pi, n)
        out = {k: np.full(n, np.nan) for k in ("th3", "th4", "mu", "w3", "w4", "Px", "Py")}
        for i, t in enumerate(th2):
            pos = self.position(t)
            if pos is None:
                continue
            out["th3"][i], out["th4"][i], out["mu"][i] = pos["th3"], pos["th4"], pos["mu"]
            out["Px"][i], out["Py"][i] = pos["P"]
            out["w3"][i], out["w4"][i] = self.velocity(t, w2)
        out["th2"] = th2
        return out

    def summary(self):
        s = self.sweep()
        ok, kind = self.grashof()
        th4 = np.unwrap(s["th4"][~np.isnan(s["th4"])])
        mu = np.degrees(s["mu"])
        return {
            "type": kind,
            "grashof": ok,
            "full_rotation": bool(np.all(~np.isnan(s["th4"]))),
            "output_swing_deg": float(np.degrees(th4.max() - th4.min())) if th4.size else 0.0,
            "min_transmission_deg": float(np.nanmin(np.minimum(mu, 180 - mu))),
        }


def animate(fb, path, frames=120, fps=30):
    """Save an animated GIF of the linkage with its coupler curve traced."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    s = fb.sweep(frames + 1)
    valid = ~np.isnan(s["th3"])
    th2 = s["th2"][valid][:-1]

    pts = np.array([[fb.position(t)[k] for k in ("O2", "A", "P", "B", "O4")] for t in th2])
    lim_lo = pts.reshape(-1, 2).min(axis=0) - 0.15 * fb.ground
    lim_hi = pts.reshape(-1, 2).max(axis=0) + 0.15 * fb.ground

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=90)
    ax.set_xlim(lim_lo[0], lim_hi[0]); ax.set_ylim(lim_lo[1], lim_hi[1])
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot(s["Px"], s["Py"], "--", color="#9aa5b1", lw=1)
    ax.plot([0, fb.ground], [0, 0], color="#52606d", lw=3)
    ax.plot([0, fb.ground], [-0.04 * fb.ground] * 2, "^", color="#9aa5b1", ms=16, zorder=0)  # ground pivots
    links, = ax.plot([], [], "-o", color="#1f6feb", lw=3, ms=7, mfc="white")
    coupler, = ax.fill([], [], color="#f0883e", alpha=0.35)[0:1]
    tracer, = ax.plot([], [], "o", color="#d1242f", ms=6)
    ax.set_title(f"Four-bar linkage — {fb.grashof()[1]}", fontsize=11)

    def draw(i):
        O2, A, P, B, O4 = pts[i]
        links.set_data([O2[0], A[0], B[0], O4[0]], [O2[1], A[1], B[1], O4[1]])
        coupler.set_xy([A, P, B])
        tracer.set_data([P[0]], [P[1]])
        return links, coupler, tracer

    FuncAnimation(fig, draw, frames=len(pts), blit=True).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
