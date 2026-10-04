"""Mechanism Designer — interactive web app.

Run locally:   streamlit run app.py
Every number on screen comes from the mechdesign package, which the test suite checks
against hand calculations.
"""
from dataclasses import replace
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from mechdesign import FourBar, GearTrain, design_train, min_pinion_teeth, parse_meshes
from mechdesign.gears import _gear_outline
from mechdesign.scenarios import FOURBAR_SCENARIOS, GEAR_SCENARIOS

BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f6feb", "#f0883e", "#2da44e", "#d1242f", "#8250df", "#8c959f"
REPO = "https://github.com/ryanch3nn/mechanism-designer"

st.set_page_config(page_title="Mechanism Designer", page_icon="⚙️", layout="wide")


# ================================================================ shareable state
# Every input lives in session_state under a fixed key, and is mirrored into the URL so a
# design can be shared by copying the address bar.
FB_FIELDS = ["ground", "crank", "coupler", "rocker", "point_dist", "point_deg", "branch", "rpm"]
G_FIELDS = ["motor_rpm", "motor_torque", "out_rpm", "load_torque"]
DEFAULTS = {
    "fb_scenario": "Custom", "fb_inspect": 30, "fb_torque": 1.0,
    **{f"fb_{k}": FOURBAR_SCENARIOS["Custom"][k] for k in FB_FIELDS},
    "g_scenario": "Custom", "g_mode": "Design it for me", "g_stages": "Fewest possible",
    "g_min_teeth": 12, "g_max_teeth": 80, "g_module": 1.0, "g_pa": 20.0, "g_eff": 0.98,
    "g_meshes": "15:40, 15:45, 16:60",
    **{f"g_{k}": GEAR_SCENARIOS["Custom"][k] for k in G_FIELDS},
}
CHOICES = {"fb_scenario": list(FOURBAR_SCENARIOS), "g_scenario": list(GEAR_SCENARIOS),
           "fb_branch": [1, -1], "g_mode": ["Design it for me", "Check my own gears"],
           "g_stages": ["Fewest possible", "1", "2", "3"],
           "g_module": [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], "g_pa": [14.5, 20.0, 25.0]}


def init_state():
    for key, default in DEFAULTS.items():
        if key in st.session_state:
            continue
        value = default
        if key in st.query_params:
            try:
                value = type(default)(float(st.query_params[key]) if isinstance(default, (int, float))
                                      else st.query_params[key])
                if key in CHOICES and value not in CHOICES[key]:
                    value = default
            except ValueError:
                value = default
        st.session_state[key] = float(value) if isinstance(default, float) else value


def load_fourbar_scenario():
    sc = FOURBAR_SCENARIOS[st.session_state.fb_scenario]
    for k in FB_FIELDS:
        st.session_state[f"fb_{k}"] = float(sc[k]) if isinstance(DEFAULTS[f"fb_{k}"], float) else sc[k]


def load_gear_scenario():
    sc = GEAR_SCENARIOS[st.session_state.g_scenario]
    for k in G_FIELDS:
        st.session_state[f"g_{k}"] = float(sc[k])
    st.session_state.g_mode = "Design it for me"


def check(level, text):
    {"ok": st.success, "warn": st.warning, "bad": st.error, "info": st.info}[level](
        text, icon={"ok": "✅", "warn": "⚠️", "bad": "❌", "info": "💡"}[level])


init_state()


# ================================================================ four-bar helpers
def crank_frames(fb, step_deg=5):
    """(crank angle, branch) for each animation frame.

    A crank that turns fully just goes round twice. If it can't, the input rocks between two
    limits, and passing through each limit flips the linkage onto its other assembly branch.
    """
    th = np.radians(np.arange(0, 360, step_deg))
    ok = np.array([fb.position(t) is not None for t in th])
    if not ok.any():
        return []
    if ok.all():
        return [(t, fb.branch) for t in np.r_[th, th]]
    start = int(np.flatnonzero(ok & ~np.roll(ok, 1))[0])   # first reachable angle after a gap
    th, ok = np.roll(th, -start), np.roll(ok, -start)
    reach = th[: int(np.argmin(ok))]
    return [(t, fb.branch) for t in reach] + [(t, -fb.branch) for t in reach[::-1]]


def linkage_figure(fb, frames, inspect):
    poses = [replace(fb, branch=b).position(t) for t, b in frames]

    def pose_traces(p):
        return [go.Scatter(x=[p[k][0] for k in ("O2", "A", "B", "O4")], y=[p[k][1] for k in ("O2", "A", "B", "O4")],
                           mode="lines+markers", line=dict(color=BLUE, width=5),
                           marker=dict(size=10, color="white", line=dict(color=BLUE, width=2.5)),
                           name="links", hoverinfo="skip"),
                go.Scatter(x=[p["A"][0], p["P"][0], p["B"][0], p["A"][0]],
                           y=[p["A"][1], p["P"][1], p["B"][1], p["A"][1]],
                           fill="toself", fillcolor="rgba(240,136,62,0.3)", line=dict(color=ORANGE, width=2),
                           mode="lines", name="coupler", hoverinfo="skip"),
                go.Scatter(x=[p["P"][0]], y=[p["P"][1]], mode="markers", name="coupler point P",
                           marker=dict(size=12, color=RED), hoverinfo="skip")]

    curves = []
    for b in sorted({b for _, b in frames}, reverse=fb.branch < 0):
        s = replace(fb, branch=b).sweep(721)
        curves.append(go.Scatter(x=s["Px"], y=s["Py"], mode="lines", name="path of P",
                                 line=dict(color=RED, width=1.5, dash="dot" if b != fb.branch else "solid"),
                                 hovertemplate="P = (%{x:.1f}, %{y:.1f})<extra></extra>",
                                 showlegend=b == fb.branch))
    ground = go.Scatter(x=[0, fb.ground], y=[0, 0], mode="lines+markers", name="ground",
                        line=dict(color=GREY, width=3, dash="dash"),
                        marker=dict(symbol="triangle-up", size=18, color=GREY), hoverinfo="skip")
    static = curves + [ground]
    n = len(static)

    fig = go.Figure(data=static + pose_traces(inspect if inspect is not None else poses[0]),
                    frames=[go.Frame(data=pose_traces(p), traces=[n, n + 1, n + 2], name=str(i))
                            for i, p in enumerate(poses)])
    pts = np.array([p[k] for p in poses for k in ("O2", "A", "B", "O4", "P")])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.08 * max(hi - lo) + 1e-9
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=10, b=70),
        xaxis=dict(range=[lo[0] - pad, hi[0] + pad], zeroline=False, title="x (mm)"),
        yaxis=dict(range=[lo[1] - pad, hi[1] + pad], zeroline=False, scaleanchor="x", title="y (mm)"),
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.13, xanchor="left", yanchor="top",
                          pad=dict(t=4, r=6), showactive=False, buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=45, redraw=False), transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.2, len=0.8, y=-0.11, yanchor="top", pad=dict(t=4), tickcolor="rgba(0,0,0,0)",
                      font=dict(color="rgba(0,0,0,0)"), currentvalue=dict(prefix="θ₂ = ", font=dict(color=GREY)),
                      steps=[dict(method="animate", label=f"{np.degrees(t):.0f}°",
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                       transition=dict(duration=0))])
                             for i, (t, _) in enumerate(frames)])])
    return fig


def angle_chart(deg, series, ytitle, mark, band=None, log=False):
    fig = go.Figure()
    if band:
        fig.add_hrect(y0=band[0], y1=band[1], fillcolor=GREEN, opacity=0.08, line_width=0,
                      annotation_text="recommended", annotation_position="top left")
    for name, y, color in series:
        fig.add_scatter(x=deg, y=y, name=name, line=dict(color=color, width=2.5),
                        hovertemplate=f"θ₂ = %{{x:.0f}}°<br>{name} = %{{y:.3g}}<extra></extra>")
    fig.add_vline(x=mark, line=dict(color=GREY, dash="dot"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0),
                      xaxis=dict(title="crank angle θ₂ (deg)", range=[0, 360], tickvals=list(range(0, 361, 45))),
                      yaxis=dict(title=ytitle, type="log" if log else "linear"))
    return fig


# ================================================================ gear helpers
def gear_animation_html(train):
    """Spinning SVG gearbox. Relative shaft speeds are exact; the whole thing is slowed down."""
    m = train.module
    speeds = train.shaft_speeds(1.0)
    fastest = max(abs(s) for s in speeds)
    colors = [BLUE, ORANGE, GREEN, PURPLE, RED]
    parts, labels, x = [], [], 0.0
    for i, (a, b) in enumerate(train.meshes):
        cd = m * (a + b) / 2
        color = colors[i % len(colors)]
        for teeth, cx, phase, shaft, below in ((b, x + cd, np.pi + np.pi / (2 * b), i + 1, False),
                                               (a, x, -np.pi / (2 * a), i, True)):
            gx, gy = _gear_outline(teeth, m, phase)
            d = "M" + "L".join(f"{px + cx:.2f},{-py:.2f}" for px, py in zip(gx, gy)) + "Z"
            period = 3.0 * fastest / abs(speeds[shaft])     # fastest shaft: one turn every 3 s
            direction = "reverse" if speeds[shaft] > 0 else "normal"   # +speed = CCW on screen
            rp = m * teeth / 2
            parts.append(
                f'<g class="spin" style="transform-origin:{cx:.2f}px 0px;animation-duration:{period:.3f}s;'
                f'animation-direction:{direction}">'
                f'<path d="{d}" fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="{0.08 * m:.3f}"/>'
                f'<line x1="{cx:.2f}" y1="0" x2="{cx:.2f}" y2="{-0.7 * rp:.2f}" stroke="{color}" '
                f'stroke-width="{0.35 * m:.2f}" stroke-linecap="round"/>'
                f'<circle cx="{cx:.2f}" cy="0" r="{max(0.12 * rp, 1.2 * m):.2f}" fill="{color}"/></g>')
            y = (rp + 5 * m) if below else -(rp + 2.4 * m)
            labels.append(f'<text x="{cx:.2f}" y="{y:.2f}" font-size="{3.6 * m:.2f}" font-weight="600" text-anchor="middle" '
                          f'fill="#8c959f">{teeth}T</text>')
        x += cd
    r_max = max(m * (n + 2) / 2 for mesh in train.meshes for n in mesh)
    x0 = -m * (train.meshes[0][0] + 2) / 2 - 3 * m
    x1 = x + m * (train.meshes[-1][1] + 2) / 2 + 3 * m
    y0, h = -r_max - 6 * m, 2 * r_max + 12 * m
    return f"""
<div style="font-family:sans-serif;color:#8c959f;font-size:13px">
  <button onclick="const s=document.getElementById('box').classList.toggle('paused');this.textContent=s?'▶ Play':'⏸ Pause'"
    style="border:1px solid #8c959f55;border-radius:6px;background:transparent;color:inherit;padding:3px 10px;cursor:pointer">⏸ Pause</button>
  <span style="margin-left:8px">Slowed down; the speed of every gear relative to the others is exact.</span>
</div>
<style>
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .spin {{ animation-name: spin; animation-timing-function: linear; animation-iteration-count: infinite; }}
  .paused .spin {{ animation-play-state: paused; }}
</style>
<svg id="box" viewBox="{x0:.2f} {y0:.2f} {x1 - x0:.2f} {h:.2f}" width="100%" height="330"
     preserveAspectRatio="xMidYMid meet" font-family="sans-serif">{''.join(parts)}{''.join(labels)}</svg>"""


@st.cache_data(show_spinner="Searching tooth counts…")
def cached_design(ratio, stages, min_teeth, max_teeth, pa, tol, module, eff):
    return design_train(ratio, stages=stages, min_teeth=min_teeth, max_teeth=max_teeth,
                        pressure_angle_deg=pa, tol=tol, module=module, efficiency=eff)


# ================================================================ page
st.title("⚙️ Mechanism Designer")
st.caption(f"Analyze four-bar linkages and design gearboxes in your browser, with every step of the math "
           f"shown. Free and open source on [GitHub]({REPO}).")

tab_fb, tab_gear, tab_help = st.tabs(["🔗 Four-bar linkage", "⚙️ Gearbox designer", "📘 Guide"])

# ---------------------------------------------------------------- four-bar
def render_fourbar():
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.selectbox("Start from a real machine", CHOICES["fb_scenario"], key="fb_scenario",
                     on_change=load_fourbar_scenario)
        sc = FOURBAR_SCENARIOS[st.session_state.fb_scenario]
        st.caption(sc["story"])
        st.markdown("**Link lengths (mm)**")
        c1, c2 = st.columns(2)
        c1.number_input("Ground r₁", min_value=0.1, step=1.0, key="fb_ground", help="Distance between the two fixed pivots O₂ and O₄.")
        c2.number_input("Crank r₂", min_value=0.1, step=1.0, key="fb_crank", help="Input link, driven by the motor.")
        c1.number_input("Coupler r₃", min_value=0.1, step=1.0, key="fb_coupler", help="Floating link from A to B.")
        c2.number_input("Rocker r₄", min_value=0.1, step=1.0, key="fb_rocker", help="Output link, pivoting about O₄.")
        st.markdown("**Coupler point P** (the point whose path you want)")
        c1, c2 = st.columns(2)
        c1.number_input("Distance from A (mm)", min_value=0.0, step=1.0, key="fb_point_dist")
        c2.number_input("Angle off AB (°)", min_value=-180.0, max_value=180.0, step=5.0, key="fb_point_deg")
        st.radio("Assembly", CHOICES["fb_branch"], key="fb_branch", horizontal=True,
                 format_func=lambda b: "Open" if b == 1 else "Crossed",
                 help="Most crank angles have two ways to assemble the linkage. Flip it if yours looks inside-out.")
        c1, c2 = st.columns(2)
        c1.number_input("Crank speed (rpm)", min_value=0.1, step=5.0, key="fb_rpm")
        c2.number_input("Input torque (N·m)", min_value=0.0, step=0.5, key="fb_torque",
                        help="Used to estimate the output torque at the inspected angle (no friction).")

    fb = FourBar(st.session_state.fb_ground, st.session_state.fb_crank, st.session_state.fb_coupler,
                 st.session_state.fb_rocker,
                 coupler_point=(st.session_state.fb_point_dist, np.radians(st.session_state.fb_point_deg)),
                 branch=st.session_state.fb_branch)
    w2 = st.session_state.fb_rpm * 2 * np.pi / 60
    frames = crank_frames(fb)

    with right:
        if not frames:
            check("bad", "These links can't be connected at any crank angle: one link is longer than the other "
                         "three put together, or the coupler and rocker can't reach each other. Try a longer "
                         "coupler or rocker.")
            return
        summ, (grashof, kind), Q = fb.summary(), fb.grashof(), fb.time_ratio()
        s = fb.sweep(n=721, w2=w2)
        deg = np.degrees(s["th2"])
        mu_eff = np.minimum(np.degrees(s["mu"]), 180 - np.degrees(s["mu"]))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Linkage type", kind.split(" (")[0].capitalize(), "Grashof" if grashof else "non-Grashof",
                  delta_color="off", delta_arrow="off")
        m2.metric("Output swing", f"{summ['output_swing_deg']:.1f}°" if summ["output_swing_deg"] < 359 else "full turn")
        m3.metric("Worst transmission angle", f"{summ['min_transmission_deg']:.1f}°",
                  "good" if summ["min_transmission_deg"] >= 40 else "low", delta_arrow="off",
                  delta_color="normal" if summ["min_transmission_deg"] >= 40 else "inverse")
        m4.metric("Time ratio", f"{Q:.3f}" if Q else "—", help="Slow stroke ÷ fast stroke (crank-rockers only).")

        inspect_deg = st.slider("Inspect crank angle θ₂ (°)", 0, 360, key="fb_inspect")
        th = np.radians(inspect_deg)
        p = fb.position(th)
        st.plotly_chart(linkage_figure(fb, frames, p), width="stretch", config={"displaylogo": False})
        if p is None:
            st.caption(f"θ₂ = {inspect_deg}° can't be reached; the drawing shows the first reachable position.")

    with left:
        st.markdown("**Checks**")
        if summ["min_transmission_deg"] >= 40:
            check("ok", f"Transmission angle never drops below {summ['min_transmission_deg']:.0f}°, so force "
                        "passes smoothly to the output.")
        elif summ["min_transmission_deg"] >= 30:
            check("warn", f"Transmission angle dips to {summ['min_transmission_deg']:.0f}° (rule of thumb: stay "
                          "above 40°). It will work, but it gets stiff and sensitive to friction there.")
        else:
            check("bad", f"Transmission angle drops to {summ['min_transmission_deg']:.0f}°. Near that position "
                         "the coupler pushes almost along the rocker, so the linkage can jam.")
        if summ["full_rotation"]:
            check("ok", "The crank turns a full 360°, so a motor can drive it directly.")
        else:
            check("warn", "The input can't turn all the way round, so a motor can't drive it continuously. "
                          "Use a rocking input (a lever or an actuator) or change the lengths.")
        if kind == "change-point":
            check("warn", "Change-point linkage: at two positions every link lines up and the linkage can "
                          "flip between open and crossed. Real designs add a guide or avoid this.")
        if Q and Q > 1.05:
            check("info", f"Quick return: one stroke takes {Q:.2f}× as long as the other. Great for "
                          "cutting or pressing on the slow stroke.")
        st.caption(sc["notice"])

    with right:
        if p is not None:
            w3, w4 = fb.velocity(th, w2)
            a, al = st.session_state.fb_point_dist, np.radians(st.session_state.fb_point_deg)
            vP = (fb.crank * w2 * np.array([-np.sin(th), np.cos(th)])
                  + a * w3 * np.array([-np.sin(p["th3"] + al), np.cos(p["th3"] + al)]))
            mu_d = np.degrees(p["mu"])
            t_out = abs(w2 / w4) * st.session_state.fb_torque if abs(w4) > 1e-9 else np.inf
            st.markdown(f"**At θ₂ = {inspect_deg}°** (crank at {st.session_state.fb_rpm:g} rpm "
                        f"= {w2:.3f} rad/s)")
            v1, v2, v3, v4, v5, v6 = st.columns(6)
            v1.metric("θ₃ coupler", f"{np.degrees(p['th3']):.2f}°")
            v2.metric("θ₄ rocker", f"{np.degrees(p['th4']):.2f}°")
            v3.metric("ω₃", f"{w3:.3f} rad/s")
            v4.metric("ω₄", f"{w4:.3f} rad/s")
            v5.metric("μ", f"{mu_d:.1f}°")
            v6.metric("Speed of P", f"{np.hypot(*vP):.1f} mm/s")
            st.caption(f"P is at ({p['P'][0]:.2f}, {p['P'][1]:.2f}) mm. Output torque ≈ "
                       f"{t_out:.3g} N·m from {st.session_state.fb_torque:g} N·m in (power in = power out, no friction).")

        g1, g2, g3, g4 = st.tabs(["Angular velocity", "Transmission angle", "Torque amplification", "Output angle"])
        with g1:
            st.plotly_chart(angle_chart(deg, [("ω₄ rocker", s["w4"], BLUE), ("ω₃ coupler", s["w3"], ORANGE)],
                                        "rad/s", inspect_deg), width="stretch")
        with g2:
            st.plotly_chart(angle_chart(deg, [("μ", np.degrees(s["mu"]), GREEN)], "deg", inspect_deg,
                                        band=(40, 140)), width="stretch")
        with g3:
            amp = np.clip(np.abs(w2 / s["w4"]), 1e-3, 1e3)
            st.plotly_chart(angle_chart(deg, [("T_out / T_in", amp, PURPLE)], "torque ratio (log)", inspect_deg,
                                        log=True), width="stretch")
            st.caption("Output torque ÷ input torque = ω₂ / ω₄ (no friction). It spikes at the toggle "
                       "positions where the rocker momentarily stops; clamps and presses use this.")
        with g4:
            th4 = np.degrees(s["th4"])
            st.plotly_chart(angle_chart(deg, [("θ₄", th4, BLUE)], "deg", inspect_deg), width="stretch")

        with st.expander("📐 How these numbers were calculated", expanded=False):
            r1, r2, r3, r4 = fb.links
            srt = np.sort(fb.links)
            st.markdown("**1. Grashof check.** Shortest + longest vs. the other two:")
            rel = r"\le" if grashof else ">"
            st.latex(rf"s + l = {srt[0]:g} + {srt[3]:g} = {srt[0] + srt[3]:g} \quad"
                     rf"{rel}\quad p + q = {srt[1]:g} + {srt[2]:g} = {srt[1] + srt[2]:g}")
            st.markdown(f"→ **{kind}**. " + ("At least one link can rotate fully; which one depends on where "
                        "the shortest link is." if grashof else "No link can make a full turn."))
            st.markdown("**2. Transmission angle.** Diagonal from A to O₄, then the law of cosines:")
            st.latex(r"d^2 = r_1^2 + r_2^2 - 2r_1r_2\cos\theta_2 \qquad \cos\mu = \frac{r_3^2 + r_4^2 - d^2}{2r_3r_4}")
            for t_deg, d in ((0, r1 - r2), (180, r1 + r2)):
                c = (r3**2 + r4**2 - d**2) / (2 * r3 * r4)
                if abs(c) <= 1:
                    st.latex(rf"\theta_2 = {t_deg}^\circ:\ d = {abs(d):g},\ \cos\mu = "
                             rf"\frac{{{r3:g}^2 + {r4:g}^2 - {abs(d):g}^2}}{{2({r3:g})({r4:g})}} = {c:.4f}"
                             rf"\ \Rightarrow\ \mu = {np.degrees(np.arccos(c)):.2f}^\circ")
            st.caption("For a crank-rocker the extremes of μ happen at θ₂ = 0° and 180°.")
            if Q:
                st.markdown("**3. Time ratio.** The rocker reverses when crank and coupler line up "
                            "(O₂B = r₃ + r₂ and r₃ − r₂). The crank angle between those two positions is 180° + δ:")
                delta = 180 * (Q - 1) / (Q + 1)
                st.latex(rf"Q = \frac{{180^\circ + \delta}}{{180^\circ - \delta}} = "
                         rf"\frac{{{180 + delta:.2f}^\circ}}{{{180 - delta:.2f}^\circ}} = {Q:.3f}")
            if p is not None:
                st.markdown(f"**4. Velocity at θ₂ = {inspect_deg}°.** Differentiate the vector loop "
                            r"$r_2e^{i\theta_2} + r_3e^{i\theta_3} - r_4e^{i\theta_4} - r_1 = 0$:")
                t3, t4 = p["th3"], p["th4"]
                st.latex(rf"\omega_4 = \frac{{r_2\omega_2\sin(\theta_2-\theta_3)}}{{r_4\sin(\theta_4-\theta_3)}} = "
                         rf"\frac{{{r2:g}({w2:.3f})\sin({inspect_deg:g}^\circ - {np.degrees(t3):.2f}^\circ)}}"
                         rf"{{{r4:g}\sin({np.degrees(t4):.2f}^\circ - {np.degrees(t3):.2f}^\circ)}} = {w4:.4f}\ \text{{rad/s}}")
                st.latex(rf"\omega_3 = \frac{{r_2\omega_2\sin(\theta_4-\theta_2)}}{{r_3\sin(\theta_3-\theta_4)}} = {w3:.4f}\ \text{{rad/s}}")

        df = pd.DataFrame({"theta2_deg": deg, "theta3_deg": np.degrees(s["th3"]), "theta4_deg": np.degrees(s["th4"]),
                           "mu_deg": np.degrees(s["mu"]), "w3_rad_s": s["w3"], "w4_rad_s": s["w4"],
                           "Px_mm": s["Px"], "Py_mm": s["Py"]}).round(5)
        report = "\n".join([
            "Four-bar linkage report (Mechanism Designer)", "",
            f"Links (mm): ground {fb.ground:g}, crank {fb.crank:g}, coupler {fb.coupler:g}, rocker {fb.rocker:g}",
            f"Coupler point: {st.session_state.fb_point_dist:g} mm from A, {st.session_state.fb_point_deg:g} deg off AB",
            f"Type: {kind}", f"Crank turns fully: {summ['full_rotation']}",
            f"Output swing: {summ['output_swing_deg']:.2f} deg",
            f"Worst transmission angle: {summ['min_transmission_deg']:.2f} deg",
            f"Time ratio: {Q:.3f}" if Q else "Time ratio: n/a"])
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Full sweep (CSV)", df.to_csv(index=False), "fourbar_sweep.csv", "text/csv",
                           width="stretch")
        d2.download_button("⬇️ Design report (TXT)", report, "fourbar_report.txt", width="stretch")

# ---------------------------------------------------------------- gearbox
def render_gearbox():
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.selectbox("Start from a real machine", CHOICES["g_scenario"], key="g_scenario",
                     on_change=load_gear_scenario)
        gsc = GEAR_SCENARIOS[st.session_state.g_scenario]
        st.caption(gsc["story"])
        st.radio("Mode", CHOICES["g_mode"], key="g_mode", horizontal=True)
        c1, c2 = st.columns(2)
        c1.number_input("Motor speed (rpm)", min_value=1e-6, step=100.0, format="%.6g", key="g_motor_rpm")
        c2.number_input("Motor torque (N·m)", min_value=0.0, step=0.05, format="%.6g", key="g_motor_torque")
        if st.session_state.g_mode == "Design it for me":
            c1.number_input("Output speed wanted (rpm)", min_value=1e-6, step=10.0, format="%.6g", key="g_out_rpm")
        else:
            st.text_input("Your gears (driver:driven, one pair per mesh)", key="g_meshes",
                          help="e.g. 15:40, 15:45. Gears in the same compound stage share a shaft.")
        c2.number_input("Load torque needed (N·m)", min_value=0.0, step=1.0, format="%.6g", key="g_load_torque",
                        help="Optional. Set it to check whether the motor is strong enough.")
        with st.expander("Gear settings", expanded=False):
            st.select_slider("Module (mm)", CHOICES["g_module"], key="g_module",
                             help="Tooth size. Pitch diameter = module × teeth. Bigger module = stronger, bigger gears.")
            st.radio("Pressure angle", CHOICES["g_pa"], key="g_pa", horizontal=True, format_func=lambda a: f"{a:g}°",
                     help="20° is today's standard; 14.5° is older and needs more teeth to avoid interference.")
            st.slider("Efficiency per mesh", 0.90, 0.995, step=0.005, key="g_eff",
                      help="0.98 is typical for spur gears.")
            if st.session_state.g_mode == "Design it for me":
                st.radio("Stages", CHOICES["g_stages"], key="g_stages", horizontal=True)
                c1, c2 = st.columns(2)
                c1.number_input("Min teeth", 6, 60, step=1, key="g_min_teeth")
                c2.number_input("Max teeth", 20, 200, step=1, key="g_max_teeth")

    mode_design = st.session_state.g_mode == "Design it for me"
    rpm_in, t_in = st.session_state.g_motor_rpm, st.session_state.g_motor_torque
    with right:
        if mode_design:
            target = rpm_in / st.session_state.g_out_rpm
            stages = None if st.session_state.g_stages == "Fewest possible" else int(st.session_state.g_stages)
            if st.session_state.g_min_teeth >= st.session_state.g_max_teeth:
                check("bad", "Min teeth has to be smaller than max teeth.")
                return
            train = cached_design(target, stages, int(st.session_state.g_min_teeth), int(st.session_state.g_max_teeth),
                                  st.session_state.g_pa, 0.001, st.session_state.g_module, st.session_state.g_eff)
        else:
            try:
                meshes = parse_meshes(st.session_state.g_meshes)
            except ValueError as e:
                check("bad", str(e).capitalize())
                return
            train = GearTrain(meshes, module=st.session_state.g_module, efficiency=st.session_state.g_eff,
                              pressure_angle_deg=st.session_state.g_pa)
            target = None

        n_out, t_out = train.output(rpm_in, t_in)
        eta = train.efficiency ** len(train.meshes)
        dims = train.dimensions()
        cds = train.center_distances()
        length = dims[0]["outside_d"] / 2 + sum(cds) + dims[-1]["outside_d"] / 2
        err = abs(train.ratio / target - 1) if target else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ratio", f"{train.ratio:.4g} : 1", f"{err:.2%} off target" if target else None,
                  delta_color="off", delta_arrow="off")
        m2.metric("Output speed", f"{abs(n_out):.4g} rpm", "same direction" if n_out > 0 else "reversed",
                  delta_color="off", delta_arrow="off")
        m3.metric("Output torque", f"{t_out:.4g} N·m", f"{eta:.1%} efficient", delta_color="off", delta_arrow="off")
        m4.metric("Gearbox size", f"{length:.0f} × {max(r['outside_d'] for r in dims):.0f} mm",
                  f"{len(train.meshes)} stage{'s' * (len(train.meshes) > 1)}", delta_color="off", delta_arrow="off")

        components.html(gear_animation_html(train), height=375)
        st.caption("Meshes: " + "  →  ".join(f"{a}:{b}" for a, b in train.meshes))

    with left:
        st.markdown("**Checks**")
        bad = train.interference_problems()
        if not bad:
            check("ok", "No tooth interference: every pinion has enough teeth for its mesh.")
        for i, n, need in bad:
            check("bad", f"Mesh {i + 1}: a {n}-tooth pinion needs at least {need} teeth at this ratio and "
                         f"pressure angle, or its tips dig into the other gear's roots.")
        if target and err > 0.001:
            check("warn", f"Closest ratio found is {err:.2%} off. Allow more teeth or another stage.")
        big = [i + 1 for i, (a, b) in enumerate(train.meshes) if max(a, b) / min(a, b) > 6]
        if big:
            check("warn", f"Stage {', '.join(map(str, big))} reduces more than 6:1, which makes one gear huge. "
                          "Splitting it into two stages is usually smaller overall.")
        load = st.session_state.g_load_torque
        if load > 0:
            need = load / (train.ratio * eta)
            if t_out >= load:
                check("ok", f"Strong enough: {t_out:.3g} N·m out vs. {load:g} N·m needed "
                            f"({t_out / load - 1:.0%} margin). The motor only needs {need:.3g} N·m.")
            else:
                check("bad", f"Too weak: {t_out:.3g} N·m out vs. {load:g} N·m needed. The motor would need "
                             f"{need:.3g} N·m, or the ratio has to go up to {load / (t_in * eta):.3g}:1.")
        if gsc["notice"]:
            st.caption(gsc["notice"])

    with right:
        speeds = train.shaft_speeds(rpm_in)
        shafts = pd.DataFrame({
            "shaft": range(1, len(speeds) + 1),
            "speed (rpm)": [abs(v) for v in speeds],
            "direction": ["same as motor" if v > 0 else "reversed" for v in speeds],
            "torque (N·m)": [t_in * rpm_in / abs(v) * train.efficiency ** k for k, v in enumerate(speeds)],
        })
        gears = pd.DataFrame(dims).rename(columns={"pitch_d": "pitch Ø (mm)", "outside_d": "outside Ø (mm)",
                                                   "root_d": "root Ø (mm)"})
        gears.insert(3, "center dist. (mm)", [cds[r["mesh"] - 1] for r in dims])
        t1, t2 = st.tabs(["Gears (for CAD)", "Shafts (speed & torque)"])
        t1.dataframe(gears, hide_index=True, width="stretch")
        t2.dataframe(shafts.round(4), hide_index=True, width="stretch")

        with st.expander("📐 How these numbers were calculated", expanded=False):
            frac = r" \times ".join(rf"\frac{{{b}}}{{{a}}}" for a, b in train.meshes)
            st.markdown("**1. Ratio**: multiply driven ÷ driver for every mesh.")
            st.latex(rf"e = {frac} = {train.ratio:.4f}")
            k = len(train.meshes)
            st.markdown("**2. Speed and torque**: speed drops by e; torque rises by e, minus losses at each mesh.")
            st.latex(rf"n_\text{{out}} = \frac{{{rpm_in:g}}}{{{train.ratio:.4f}}} = {abs(n_out):.4g}\ \text{{rpm}}"
                     rf"\qquad T_\text{{out}} = {t_in:g} \times {train.ratio:.4f} \times {train.efficiency:g}^{{{k}}}"
                     rf" = {t_out:.4g}\ \text{{N·m}}")
            st.markdown(f"**3. Direction**: each external mesh reverses rotation. {k} mesh{'es' * (k > 1)} → "
                        f"output turns **{'the same way as' if train.direction > 0 else 'opposite to'}** the motor.")
            st.markdown("**4. Center distance**: half the sum of the pitch diameters.")
            st.latex(r"\quad ".join(rf"C_{i + 1} = \frac{{{train.module:g}({a}+{b})}}{{2}} = {c:g}"
                                    for i, ((a, b), c) in enumerate(zip(train.meshes, cds))) + r"\ \text{mm}")
            a, b = min(train.meshes, key=min)
            mg = max(a, b) / min(a, b)
            st.markdown(f"**5. Interference** (Shigley eq. 13-11), for the smallest pinion ({min(a, b)}T, "
                        f"ratio {mg:.3g}):")
            st.latex(rf"N_P = \frac{{2k}}{{(1+2m_G)\sin^2\phi}}\left(m_G + \sqrt{{m_G^2 + (1+2m_G)\sin^2\phi}}\right)"
                     rf" \Rightarrow {min_pinion_teeth(mg, train.pressure_angle_deg)}\ \text{{teeth minimum}}")

        buf = io.StringIO()
        buf.write("Gearbox report (Mechanism Designer)\n\n" + train.report(rpm_in, t_in) + "\n\n")
        gears.to_csv(buf, index=False)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Gear dimensions (CSV)", gears.to_csv(index=False), "gears.csv", "text/csv",
                           width="stretch")
        d2.download_button("⬇️ Design report (TXT)", buf.getvalue(), "gearbox_report.txt", width="stretch")

with tab_fb:
    render_fourbar()
with tab_gear:
    render_gearbox()

# ---------------------------------------------------------------- guide
with tab_help:
    st.markdown(f"""
### What can I do here?
**Four-bar linkage**: type in four link lengths and see how the linkage moves, what path a point on it
traces, and whether it will run smoothly. Good for wipers, robot legs, presses, clamps, toys and
checking kinematics homework.

**Gearbox designer**: say how fast your motor spins and how fast the output should turn, and it picks
the tooth counts, checks that the teeth won't clash, and tells you the output torque. Switch to
**Check my own gears** to analyze a gearbox you already have.

### Tips
- **Start from a real machine** in the drop-down, then change one number at a time and watch the checks.
- **Inspect crank angle** gives you every angle and velocity at one position: handy for checking a hand calculation.
- **📐 How these numbers were calculated** shows each formula with your numbers plugged in.
- **Share a design** by copying the address bar: every input is saved in the link.
- **Download** the full sweep as CSV for Excel or MATLAB, or the gear table for your CAD model.

### Rules of thumb used in the checks
| Check | Rule |
|---|---|
| Transmission angle | keep μ between 40° and 140° |
| Gear stage ratio | ≤ 6:1 per stage keeps gears a sensible size |
| Tooth interference | pinion teeth ≥ Shigley's minimum for that ratio and pressure angle |
| Spur gear efficiency | about 98 % per mesh |

### Limits
Gear drawings are approximate outlines, not true involute profiles for manufacturing. The app doesn't size
gears for strength (Lewis bending stress) yet, and the linkage analysis covers position and velocity but
not acceleration or forces. See the [roadmap]({REPO}#limitations-and-roadmap).
""")

# keep the URL in sync so any design can be shared as a link
st.query_params.from_dict({k: str(st.session_state[k]) for k in DEFAULTS})
