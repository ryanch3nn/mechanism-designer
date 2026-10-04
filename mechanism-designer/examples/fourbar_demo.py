"""Analyze a crank-rocker and save its animation plus analysis plots to docs/img/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mechdesign import FourBar, animate

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# link lengths in mm: ground, crank, coupler, rocker; tracer point 40 mm along coupler, 35 deg off it
fb = FourBar(ground=70, crank=20, coupler=60, rocker=50, coupler_point=(40, np.radians(35)))

for k, v in fb.summary().items():
    print(f"{k:>22}: {v:.2f}" if isinstance(v, float) else f"{k:>22}: {v}")

s = fb.sweep(n=721, w2=10.0)  # crank at 10 rad/s
deg = np.degrees(s["th2"])
mu = np.degrees(s["mu"])

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), dpi=110)
ax[0].plot(s["Px"], s["Py"], color="#d1242f")
ax[0].set_title("Coupler curve (mm)"); ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3)
ax[1].plot(deg, s["w4"], color="#1f6feb", label="ω₄ rocker")
ax[1].plot(deg, s["w3"], color="#f0883e", label="ω₃ coupler")
ax[1].set_title("Angular velocity at ω₂ = 10 rad/s"); ax[1].set_xlabel("crank angle θ₂ (deg)")
ax[1].set_ylabel("rad/s"); ax[1].legend(frameon=False); ax[1].grid(alpha=0.3)
ax[2].plot(deg, mu, color="#2da44e")
ax[2].axhspan(40, 140, color="#2da44e", alpha=0.08, label="recommended 40°–140°")
ax[2].set_title("Transmission angle μ"); ax[2].set_xlabel("crank angle θ₂ (deg)")
ax[2].set_ylabel("deg"); ax[2].set_ylim(0, 180); ax[2].legend(frameon=False); ax[2].grid(alpha=0.3)
for a in ax[1:]:
    a.set_xlim(0, 360); a.set_xticks(range(0, 361, 90))
fig.tight_layout()
fig.savefig(OUT / "fourbar_analysis.png")

animate(fb, OUT / "fourbar.gif")
print("saved", OUT / "fourbar_analysis.png", "and", OUT / "fourbar.gif")
