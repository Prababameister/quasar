"""
fk_interactive.py - interactive FORWARD kinematics, ONE leg at a time, 3D.

Run from the quasar-1 folder:   python fk_interactive.py

Drag the three joint sliders (theta1, theta_c, hip). The real FK code
(leg_kinematics.fk_leg) runs on every move and the display shows where the
foot lands in X, Y, Z, plus a geometry check on the linkage.
Pick the leg with the radio buttons.

ARDUINO: if an Arduino running cobs_motor_prog.ino is plugged in, the joint
angles are also sent to the selected leg's servos on every move. Untick
"Send to Arduino" to stop. Nothing is sent if the linkage can't assemble,
theta1/theta_c are outside limits, or the hip is past the swing limit.
  python fk_interactive.py [--port COM4] [--no-serial]

Axes (leg-local, hip at origin):
  X = forward (mm)    Y = sideways, positive = OUTWARD from the body (mm)
  Z = height (mm), negative = below the hip
hip = 90 deg means the leg hangs straight down.
"""

import argparse
import math

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons

from leg_kinematics import fk_leg, leg_names, leg_side, hip_dir, hip_fixed_deg
from workspace import compute_workspace
from arduino_link import ArduinoLink, add_cli_args

CFG = yaml.safe_load(open("linkage_config.yaml"))
BARS = [("O", "A"), ("A", "B"), ("Q", "B"), ("Q", "C"), ("Q", "P1"),
        ("C", "P2"), ("P1", "P2"), ("P1", "D")]
LIMITS = {i["name"]: (i.get("min_deg", -999), i.get("max_deg", 999)) for i in CFG["inputs"]}
SWING_MAX = float(CFG.get("hip_swing_max_deg", 30.0))
HIP0 = hip_fixed_deg(CFG)

state = {"leg": "FL"}

ap = argparse.ArgumentParser(description="Interactive FK tester (+ live Arduino output)")
add_cli_args(ap)
args = ap.parse_args()
LINK = ArduinoLink(CFG)
LINK.connect_from_args(args)

print("Computing workspace (a few seconds)...")
WS = compute_workspace(CFG)[::4]
WSX = [p["x"] for p in WS]
WSY = [p["y"] for p in WS]

T1_0, TC_0 = 150.0, -60.0   # starting pose (a valid one)

fig = plt.figure(figsize=(15, 8))
ax2d = fig.add_axes([0.04, 0.36, 0.28, 0.58])
ax3d = fig.add_axes([0.33, 0.30, 0.40, 0.66], projection="3d")
axtxt = fig.add_axes([0.74, 0.36, 0.25, 0.58])
axtxt.axis("off")

t1_lo, t1_hi = LIMITS["theta1"]
tc_lo, tc_hi = LIMITS["theta_c"]
s_t1 = Slider(fig.add_axes([0.10, 0.20, 0.50, 0.03]), "theta1 (deg)", t1_lo, t1_hi, valinit=T1_0, valstep=0.5)
s_tc = Slider(fig.add_axes([0.10, 0.14, 0.50, 0.03]), "theta_c (deg)", tc_lo, tc_hi, valinit=TC_0, valstep=0.5)
s_hip = Slider(fig.add_axes([0.10, 0.08, 0.50, 0.03]), "hip (deg)", HIP0 - 60, HIP0 + 60, valinit=HIP0, valstep=0.5)
radio = RadioButtons(fig.add_axes([0.72, 0.06, 0.10, 0.20]), leg_names(CFG))
btn = Button(fig.add_axes([0.62, 0.08, 0.07, 0.04]), "Reset")
send_box = CheckButtons(fig.add_axes([0.84, 0.10, 0.14, 0.06]), ["Send to Arduino"],
                        [LINK.enabled])


def approx_us(leg, servo, t1, tc, hip):
    """Exactly the pulse send_leg() would use (offsets + calibration frame + calibration)."""
    return LINK.mgr.compute_leg_us(CFG, leg, t1, tc, hip)[servo]


def to_3d(leg, pts, hip_deg):
    """planar (x, y) points -> leg-local 3D. Right legs go toward -Y."""
    s = 1.0 if leg_side(CFG, leg) == "left" else -1.0
    swing = math.radians((hip_deg - HIP0) / hip_dir(CFG, leg))
    return {k: np.array([p["x"], s * (-p["y"]) * math.sin(swing), p["y"] * math.cos(swing)])
            for k, p in pts.items()}


def update(_=None):
    leg = state["leg"]
    side = leg_side(CFG, leg)
    s = 1.0 if side == "left" else -1.0
    t1, tc, hip = s_t1.val, s_tc.val, s_hip.val

    elev, azim = ax3d.elev, ax3d.azim
    ax2d.clear(); ax3d.clear(); axtxt.clear(); axtxt.axis("off")
    lines = [f"LEG {leg}  ({side})", "",
             "JOINT ANGLES (inputs)",
             f"  theta1  {t1:8.2f} deg",
             f"  theta_c {tc:8.2f} deg",
             f"  hip     {hip:8.2f} deg", ""]

    try:
        pts = fk_leg(CFG, leg, t1, tc)["points"]
    except Exception as e:
        lines += ["LINKAGE CANNOT ASSEMBLE at these angles:", str(e)[:140], "",
                  "Move a slider to a different value.", "",
                  LINK.send(leg, t1, tc, hip, ["linkage can't assemble"])]
        axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
        ax2d.scatter(WSX, WSY, s=1, c="lightgray")
        ax2d.set_xlim(-100, 260); ax2d.set_ylim(-260, 60); ax2d.set_aspect("equal")
        ax3d.view_init(elev, azim)
        fig.canvas.draw_idle()
        return

    p3 = to_3d(leg, pts, hip)
    foot = p3["D"]
    y_out = s * foot[1]   # positive = outward for both sides

    # linkage geometry check: every bar must keep its configured length
    worst = 0.0
    for e in CFG["edges"]:
        a, b = e["from"], e["to"]
        got = math.hypot(pts[a]["x"] - pts[b]["x"], pts[a]["y"] - pts[b]["y"])
        worst = max(worst, abs(got - e["length"]))

    # ---- 2D: linkage in the leg's own plane
    ax2d.scatter(WSX, WSY, s=1, c="lightgray")
    for a, b in BARS:
        ax2d.plot([pts[a]["x"], pts[b]["x"]], [pts[a]["y"], pts[b]["y"]], "b-")
    ax2d.scatter([pts["D"]["x"]], [pts["D"]["y"]], c="r")
    ax2d.set_xlim(-100, 260); ax2d.set_ylim(-260, 60); ax2d.set_aspect("equal")
    ax2d.set_title("Leg plane (2D linkage)")

    # ---- 3D
    for a, b in BARS:
        ax3d.plot(*zip(p3[a], p3[b]), "b-")
    ax3d.scatter(*foot, c="r", s=40)
    ax3d.set_xlim(-80, 220); ax3d.set_ylim(-80, 80); ax3d.set_zlim(-240, 40)
    ax3d.set_xlabel("X fwd"); ax3d.set_ylabel("Y (+left)"); ax3d.set_zlabel("Z up")
    ax3d.set_title("3D (red dot = foot)")
    ax3d.view_init(elev, azim)

    lines += [
        "FOOT POSITION (output)",
        f"  X (forward)  {foot[0]:8.1f} mm",
        f"  Y (outward)  {y_out:8.1f} mm",
        f"  Z (height)   {foot[2]:8.1f} mm", "",
        f"in leg plane: x {pts['D']['x']:.1f}  y {pts['D']['y']:.1f}", "",
        f"bar-length check: worst dev {worst:.4f} mm  {'OK' if worst < 0.01 else 'BAD'}", "",
        "approx servo pulses (us):",
        f"  theta1  {approx_us(leg, 'theta1', t1, tc, hip):6.0f}",
        f"  theta_c {approx_us(leg, 'theta_c', t1, tc, hip):6.0f}",
        f"  hip     {approx_us(leg, 'hip', t1, tc, hip):6.0f}", "",
    ]
    warn = []
    if abs(hip - HIP0) > SWING_MAX:
        warn.append(f"hip beyond +/-{SWING_MAX:.0f} deg swing limit")
    blockers = [w for w in warn]
    for name, val in (("theta1", t1), ("theta_c", tc)):
        lo, hi = LIMITS[name]
        if not lo <= val <= hi:
            blockers.append(f"{name} out of limits")
    if worst >= 0.01:
        blockers.append("bad linkage geometry")
    lines += ["WARNINGS: " + ("none" if not warn else "")] + ["  - " + w for w in warn]
    lines += ["", LINK.send(leg, t1, tc, hip, blockers)]

    axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.canvas.draw_idle()


def on_leg(label):
    state["leg"] = label
    update()


def on_send(_):
    LINK.set_enabled(send_box.get_status()[0])
    update()


def on_reset(_):
    s_t1.reset(); s_tc.reset(); s_hip.reset()


for sl in (s_t1, s_tc, s_hip):
    sl.on_changed(update)
radio.on_clicked(on_leg)
btn.on_clicked(on_reset)
send_box.on_clicked(on_send)
update()
try:
    plt.show()
finally:
    LINK.close()