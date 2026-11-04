"""
ik_interactive.py - interactive IK tester, ONE leg at a time, 2D + 3D.

Run from the quasar-1 folder:   python ik_interactive.py

Drag the sliders to move the target foot position. The real solver
(leg_kinematics.solve_leg_3d) runs on every move and the display shows:
  - left   : the planar linkage in the leg's own plane (+ reachable workspace)
  - middle : the leg in 3D (hip at the origin)
  - right  : target, IK angles (theta1, theta_c, hip), FK check, servo pulses
Pick the leg with the radio buttons.

ARDUINO: if an Arduino running cobs_motor_prog.ino is plugged in, every
slider move also sends the solved angles to that leg's servos. Untick
"Send to Arduino" to stop sending. A pose is only sent when IK is valid and
theta1/theta_c are inside their limits.
  python ik_interactive.py                  # auto-detect the port
  python ik_interactive.py --port COM5      # pick the port
  python ik_interactive.py --no-serial      # screen only

Axes (leg-local, hip at origin):
  X = forward (mm)        Y = sideways, positive = OUTWARD from the body (mm)
  Z = height (mm), negative = below the hip
"""

import argparse
import math

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

from leg_kinematics import (solve_leg_3d, fk_leg, leg_names, leg_side,
                            hip_dir, hip_fixed_deg)
from workspace import compute_workspace
from arduino_link import ArduinoLink, add_cli_args

CFG = yaml.safe_load(open("linkage_config.yaml"))
BARS = [("O", "A"), ("A", "B"), ("Q", "B"), ("Q", "C"), ("Q", "P1"),
        ("C", "P2"), ("P1", "P2"), ("P1", "D")]
LIMITS = {i["name"]: (i.get("min_deg", -999), i.get("max_deg", 999)) for i in CFG["inputs"]}
SWING_MAX = float(CFG.get("hip_swing_max_deg", 30.0))
OFFSET_KEY = {"theta1": "theta1_deg", "theta_c": "thetac_deg", "hip": "hip_deg"}

state = {"leg": "FL"}

# ---- Arduino connection (see arduino_link.py)
ap = argparse.ArgumentParser(description="Interactive IK tester (+ live Arduino output)")
add_cli_args(ap)
args = ap.parse_args()
LINK = ArduinoLink(CFG)
LINK.connect_from_args(args)

print("Computing workspace (a few seconds)...")
WS = compute_workspace(CFG)[::4]
WSX = [p["x"] for p in WS]
WSY = [p["y"] for p in WS]

# start at the first gait waypoint
X0, Z0 = CFG["gait"]["waypoints"][0]

fig = plt.figure(figsize=(15, 8))
ax2d = fig.add_axes([0.04, 0.36, 0.28, 0.58])
ax3d = fig.add_axes([0.33, 0.30, 0.40, 0.66], projection="3d")
axtxt = fig.add_axes([0.74, 0.36, 0.25, 0.58])
axtxt.axis("off")

s_x = Slider(fig.add_axes([0.10, 0.20, 0.50, 0.03]), "X fwd (mm)", -80, 220, valinit=X0, valstep=1)
s_y = Slider(fig.add_axes([0.10, 0.14, 0.50, 0.03]), "Y outward (mm)", -60, 60, valinit=0, valstep=1)
s_z = Slider(fig.add_axes([0.10, 0.08, 0.50, 0.03]), "Z height (mm)", -240, -50, valinit=Z0, valstep=1)
s_h = Slider(fig.add_axes([0.10, 0.02, 0.50, 0.03]), "Hip swing (deg)", -SWING_MAX, SWING_MAX,
             valinit=0, valstep=0.5)
radio = RadioButtons(fig.add_axes([0.72, 0.06, 0.10, 0.20]), leg_names(CFG))
mode_radio = RadioButtons(fig.add_axes([0.62, 0.02, 0.09, 0.14]), ["Y slider", "Hip slider"])
state["mode"] = "Y slider"
send_box = CheckButtons(fig.add_axes([0.84, 0.10, 0.14, 0.06]), ["Send to Arduino"],
                        [LINK.enabled])


def approx_us(leg, servo, deg):
    cal = CFG["legs"][leg]["calibration"][servo]
    deg = deg + float(CFG["legs"][leg]["offsets"].get(OFFSET_KEY[servo], 0.0))
    return cal["us0"] + (deg - cal["deg0"]) * (cal["us1"] - cal["us0"]) / (cal["deg1"] - cal["deg0"])


def to_3d(leg, pts, hip_deg):
    """planar (x, y) points -> leg-local 3D. Right legs go toward -Y."""
    s = 1.0 if leg_side(CFG, leg) == "left" else -1.0
    swing = math.radians((hip_deg - hip_fixed_deg(CFG)) / hip_dir(CFG, leg))
    return {k: np.array([p["x"], s * (-p["y"]) * math.sin(swing), p["y"] * math.cos(swing)])
            for k, p in pts.items()}


def update(_=None):
    leg = state["leg"]
    side = leg_side(CFG, leg)
    s = 1.0 if side == "left" else -1.0
    x, lat, z = s_x.val, s_y.val, s_z.val
    if state["mode"] == "Hip slider":
        # hip swing drives the sideways offset: lateral = depth * tan(swing)
        lat = (-z) * math.tan(math.radians(s_h.val))

    elev, azim = ax3d.elev, ax3d.azim
    ax2d.clear(); ax3d.clear(); axtxt.clear(); axtxt.axis("off")
    lines = [f"LEG {leg}  ({side})", "",
             f"TARGET  X {x:7.1f}   Y {lat:6.1f}   Z {z:7.1f}", ""]

    if z >= 0:
        lines += ["Z must be negative (below the hip).", "",
                  LINK.send(leg, 0, 0, 0, ["Z >= 0"])]
        axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
        fig.canvas.draw_idle()
        return

    try:
        r = solve_leg_3d(CFG, leg, x, z, lateral=lat)
        pts = fk_leg(CFG, leg, r["theta1"], r["theta_c"])["points"]
    except Exception as e:
        lines += ["IK/FK failed:", str(e)[:120], "",
                  LINK.send(leg, 0, 0, 0, ["IK failed"])]
        axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
        fig.canvas.draw_idle()
        return

    p3 = to_3d(leg, pts, r["hip"])
    foot = p3["D"]
    want = np.array([x, s * lat, z])
    err3 = float(np.linalg.norm(foot - want))

    # ---- 2D: linkage in the leg's own (tilted) plane
    ax2d.scatter(WSX, WSY, s=1, c="lightgray")
    for a, b in BARS:
        ax2d.plot([pts[a]["x"], pts[b]["x"]], [pts[a]["y"], pts[b]["y"]], "b-")
    reach = math.hypot(lat, -z)
    ax2d.scatter([x], [-reach], marker="x", c="g", s=80, label="planar target")
    ax2d.scatter([pts["D"]["x"]], [pts["D"]["y"]], c="r", label="foot")
    ax2d.set_xlim(-100, 260); ax2d.set_ylim(-260, 60); ax2d.set_aspect("equal")
    ax2d.set_title("Leg plane (2D linkage)"); ax2d.legend(loc="upper right", fontsize=8)

    # ---- 3D
    for a, b in BARS:
        ax3d.plot(*zip(p3[a], p3[b]), "b-")
    ax3d.scatter(*foot, c="r", s=40)
    ax3d.scatter(*want, c="g", marker="x", s=80)
    ax3d.set_xlim(-80, 220); ax3d.set_ylim(-80, 80); ax3d.set_zlim(-240, 40)
    ax3d.set_xlabel("X fwd"); ax3d.set_ylabel("Y (+left)"); ax3d.set_zlabel("Z up")
    ax3d.set_title("3D (green x = target, red = foot)")
    ax3d.view_init(elev, azim)

    # ---- text
    lines += [
        f"IK ANGLES  theta1  {r['theta1']:8.2f} deg",
        f"           theta_c {r['theta_c']:8.2f} deg",
        f"           hip     {r['hip']:8.2f} deg",
        "",
        f"IK valid: {r['valid']}   planar err {r['error_mm']:.2f} mm",
        f"FK foot   X {foot[0]:7.1f}  Y {foot[1]:6.1f}  Z {foot[2]:7.1f}",
        f"3D error  {err3:.2f} mm", "",
        "approx servo pulses (us):",
        f"  theta1  {approx_us(leg, 'theta1', r['theta1']):6.0f}",
        f"  theta_c {approx_us(leg, 'theta_c', r['theta_c']):6.0f}",
        f"  hip     {approx_us(leg, 'hip', r['hip']):6.0f}", "",
    ]
    warn = []
    blockers = []
    for name, val in (("theta1", r["theta1"]), ("theta_c", r["theta_c"])):
        lo, hi = LIMITS[name]
        if not lo <= val <= hi:
            warn.append(f"{name} outside limits {lo}..{hi}")
            blockers.append(f"{name} out of limits")
    if abs(math.degrees(math.atan2(lat, -z))) > SWING_MAX:
        warn.append(f"hip swing clamped at {SWING_MAX:.0f} deg")
    if not r["valid"]:
        warn.append("target NOT reachable")
        blockers.insert(0, "unreachable")
    lines += ["WARNINGS: " + ("none" if not warn else "")] + ["  - " + w for w in warn]
    lines += ["", LINK.send(leg, r["theta1"], r["theta_c"], r["hip"], blockers)]

    axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.canvas.draw_idle()


def on_leg(label):
    state["leg"] = label
    update()


def on_mode(label):
    state["mode"] = label
    update()


def on_send(_):
    LINK.set_enabled(send_box.get_status()[0])
    update()


send_box.on_clicked(on_send)
for sl in (s_x, s_y, s_z, s_h):
    sl.on_changed(update)
radio.on_clicked(on_leg)
mode_radio.on_clicked(on_mode)
update()
try:
    plt.show()
finally:
    LINK.close()