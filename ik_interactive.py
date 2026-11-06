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
                            leg_plane_to_3d, lateral_for_swing, hip_offset_mm)
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

fig = plt.figure(figsize=(17, 8))
ax2d = fig.add_axes([0.03, 0.36, 0.22, 0.58])
axyz = fig.add_axes([0.27, 0.36, 0.17, 0.58])
ax3d = fig.add_axes([0.45, 0.30, 0.31, 0.66], projection="3d")
axtxt = fig.add_axes([0.77, 0.36, 0.22, 0.58])
axtxt.axis("off")

s_x = Slider(fig.add_axes([0.10, 0.20, 0.50, 0.03]), "X fwd (mm)", -150, 250, valinit=X0, valstep=1)
s_y = Slider(fig.add_axes([0.10, 0.14, 0.50, 0.03]), "Y outward (mm)", -60, 60, valinit=0, valstep=1)
s_z = Slider(fig.add_axes([0.10, 0.08, 0.50, 0.03]), "Z height (mm)", -270, -50, valinit=Z0, valstep=1)
s_h = Slider(fig.add_axes([0.10, 0.02, 0.50, 0.03]), "Hip swing (deg)", -SWING_MAX, SWING_MAX,
             valinit=0, valstep=0.5)
radio = RadioButtons(fig.add_axes([0.72, 0.06, 0.10, 0.20]), leg_names(CFG))
mode_radio = RadioButtons(fig.add_axes([0.62, 0.02, 0.09, 0.14]), ["Y slider", "Hip slider"])
state["mode"] = "Y slider"
send_box = CheckButtons(fig.add_axes([0.84, 0.10, 0.14, 0.06]), ["Send to Arduino"],
                        [LINK.enabled])


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def doc_angles(r):
    """
    Solver angles -> the convention in 'Quadruped V1 Documentation' (photos).

    Solver frame: standard math angles in the leg plane, +x forward, +y up.
      theta_c = thigh (Q -> P1),        theta1 = knee servo horn (O -> A)
    Doc frame:
      Theta 1 = thigh: 0 = parallel to ground (pointing forward),
                       90 = perpendicular, leg straight down    -> -theta_c
      Theta c = horn:  0 = parallel to ground (pointing back),
                       90 = horn pointing up                    -> 180 - theta1
      Hip:             90 = leg hanging straight down (neutral) -> hip (unchanged)
    Display only: the Arduino still gets solver-frame angles, which is
    what the per-servo calibration in linkage_config.yaml is written in.
    """
    return {"theta1": _wrap180(-r["theta_c"]),
            "theta_c": _wrap180(180.0 - r["theta1"]),
            "hip": r["hip"]}


def approx_us(leg, servo, deg):
    cal = CFG["legs"][leg]["calibration"][servo]
    deg = deg + float(CFG["legs"][leg]["offsets"].get(OFFSET_KEY[servo], 0.0))
    return cal["us0"] + (deg - cal["deg0"]) * (cal["us1"] - cal["us0"]) / (cal["deg1"] - cal["deg0"])


def to_3d(leg, pts, hip_deg):
    """planar (x, y) points -> leg-local 3D (hip axis at origin). Right legs go toward -Y."""
    p3 = {k: np.array(leg_plane_to_3d(CFG, leg, p["x"], p["y"], hip_deg))
          for k, p in pts.items()}
    p3["H"] = np.zeros(3)               # hip axis; H -> plane origin is the hip offset
    p3["H0"] = np.array(leg_plane_to_3d(CFG, leg, 0.0, 0.0, hip_deg))  # bar end, rotates with the leg
    return p3


def update(_=None):
    leg = state["leg"]
    side = leg_side(CFG, leg)
    s = 1.0 if side == "left" else -1.0
    x, lat, z = s_x.val, s_y.val, s_z.val
    if state["mode"] == "Hip slider":
        # hip swing drives the sideways offset (accounts for the hip offset)
        lat = lateral_for_swing(CFG, z, s_h.val)

    elev, azim = ax3d.elev, ax3d.azim
    ax2d.clear(); axyz.clear(); ax3d.clear(); axtxt.clear(); axtxt.axis("off")
    lines = [f"LEG {leg}  ({side})", "",
             f"TARGET  X {x:7.1f}   Y {lat:6.1f}   Z {z:7.1f}", ""]

    if z >= 0:
        lines += ["Z must be negative (below the hip).", "",
                  LINK.send(leg, 0, 0, 0, ["Z >= 0"])]
        axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
        fig.canvas.draw_idle()
        return

    try:
        r = solve_leg_3d(CFG, leg, x, z, lateral=lat)
        pts = fk_leg(CFG, leg, r["theta1"], r["theta_c"])["points"]
    except Exception as e:
        lines += ["IK/FK failed:", str(e)[:120], "",
                  LINK.send(leg, 0, 0, 0, ["IK failed"])]
        axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
        fig.canvas.draw_idle()
        return

    p3 = to_3d(leg, pts, r["hip"])
    foot = p3["D"]
    want = np.array([x, s * (hip_offset_mm(CFG) + lat), z])
    err3 = float(np.linalg.norm(foot - want))

    # ---- 2D: linkage in the leg's own (tilted) plane
    ax2d.scatter(WSX, WSY, s=1, c="lightgray")
    for a, b in BARS:
        ax2d.plot([pts[a]["x"], pts[b]["x"]], [pts[a]["y"], pts[b]["y"]], "b-")
    reach = r["reach"]
    ax2d.scatter([x], [-reach], marker="x", c="g", s=80, label="planar target")
    ax2d.scatter([pts["D"]["x"]], [pts["D"]["y"]], c="r", label="foot")
    ax2d.set_xlim(-160, 280); ax2d.set_ylim(-280, 80); ax2d.set_aspect("equal")
    ax2d.set_title("Leg plane (2D linkage)"); ax2d.legend(loc="upper right", fontsize=8)

    # ---- Y-Z: looking along the hip axis (front view), Y = outward from body
    def yz(p):
        return s * p[1], p[2]
    h0 = np.array(yz(p3["H0"]))
    u_bar = h0 / np.linalg.norm(h0)               # hip axis -> leg plane
    u_leg = np.array([u_bar[1], -u_bar[0]])       # leg-plane "down", exactly 90 deg to the bar
    for a, b in BARS:
        axyz.plot(*zip(yz(p3[a]), yz(p3[b])), "b-")
    axyz.plot([0, h0[0]], [0, h0[1]], "k-", lw=3, label="hip offset")
    ln = np.array([h0 + 40 * u_leg, h0 - 300 * u_leg])
    axyz.plot(ln[:, 0], ln[:, 1], "k:", lw=0.8, label="leg plane")
    m = 10.0                                      # right-angle marker
    sq = np.array([h0 - m * u_bar, h0 - m * u_bar + m * u_leg, h0 + m * u_leg])
    axyz.plot(sq[:, 0], sq[:, 1], "m-", lw=1.5)
    axyz.scatter([0], [0], c="k", marker="o", s=30)
    axyz.scatter(*yz(foot), c="r", s=40, label="foot")
    axyz.scatter(*yz(want), c="g", marker="x", s=80, label="target")
    axyz.axhline(0, c="lightgray", lw=0.5); axyz.axvline(0, c="lightgray", lw=0.5)
    axyz.set_xlim(-60, 160); axyz.set_ylim(-280, 80); axyz.set_aspect("equal")
    axyz.set_xlabel("Y outward (mm)"); axyz.set_ylabel("Z (mm)")
    axyz.set_title("Front view (Y-Z)"); axyz.legend(loc="upper right", fontsize=7)
    foot_dir = np.array(yz(foot)) - h0
    bar_leg_deg = math.degrees(math.atan2(abs(u_bar[1] * foot_dir[0] - u_bar[0] * foot_dir[1]),
                                          float(np.dot(-u_bar, foot_dir))))

    # ---- 3D
    for a, b in BARS:
        ax3d.plot(*zip(p3[a], p3[b]), "b-")
    ax3d.plot(*zip(p3["H"], p3["H0"]), "k-", lw=3)   # hip offset
    ax3d.scatter(*foot, c="r", s=40)
    ax3d.scatter(*want, c="g", marker="x", s=80)
    ax3d.set_xlim(-150, 250); ax3d.set_ylim(-160, 160); ax3d.set_zlim(-280, 40)
    ax3d.set_box_aspect((400, 320, 320))   # equal mm scale on all axes, so 90 deg looks 90 deg
    ax3d.set_xlabel("X fwd"); ax3d.set_ylabel("Y (+left)"); ax3d.set_zlabel("Z up")
    ax3d.set_title("3D (green x = target, red = foot)")
    ax3d.view_init(elev, azim)

    # ---- text
    d = doc_angles(r)
    lines += [
        "ANGLES (doc convention)",
        f"  Theta 1 (thigh) {d['theta1']:8.2f} deg",
        f"  Theta c (horn)  {d['theta_c']:8.2f} deg",
        f"  Hip             {d['hip']:8.2f} deg",
        f"  solver: theta1 {r['theta1']:.1f}  theta_c {r['theta_c']:.1f}",
        "",
        f"IK valid: {r['valid']}   planar err {r['error_mm']:.2f} mm",
        f"FK foot   X {foot[0]:7.1f}  Y {foot[1]:6.1f}  Z {foot[2]:7.1f}",
        f"3D error  {err3:.2f} mm",
        f"hip offset {np.linalg.norm(h0):.3f} mm, at {bar_leg_deg:.6f} deg to leg", "",
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
    if abs(r["swing_deg"]) > SWING_MAX:
        warn.append(f"hip swing clamped at {SWING_MAX:.0f} deg")
    if not r["valid"]:
        warn.append("target NOT reachable")
        blockers.insert(0, "unreachable")
    lines += ["WARNINGS: " + ("none" if not warn else "")] + ["  - " + w for w in warn]
    lines += ["", LINK.send(leg, r["theta1"], r["theta_c"], r["hip"], blockers)]

    axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
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