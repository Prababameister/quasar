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

HARDWARE FRAME: the angles shown are the ones the servos are calibrated in
(calibration_frame: hardware in linkage_config.yaml). Seen from the side:
  horn  servo (config key theta1):  0 = level, pointing one way, 90 = up
  thigh servo (config key theta_c): 0 = level, pointing the other way
                                    (180 deg from horn zero), 90 = straight down
  hip   servo: 90 = leg hanging straight down, 0 = leg horizontal, swung outward
Left/right are as seen from the leg's own outside. Each leg's x_forward_sign
in linkage_config.yaml says whether the solver's +x is the robot's front (+1,
right legs) or back (-1, left legs).
The solver works internally in its own frame; serial_manager converts to the
hardware frame right before the angle -> us step, and the pulses shown here
are exactly the ones sent (same function).

Axes (leg-local, hip at origin), all in ROBOT terms:
  X = toward the robot's front (mm); converted to the solver's frame with
      the leg's x_forward_sign, and every plot is drawn with the robot's
      front toward +X        Y = sideways, positive = OUTWARD from the body (mm)
  Z = height (mm), negative = below the hip
"""

import argparse
import math

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

from leg_kinematics import (solve_leg_3d, fk_leg, leg_names, leg_side,
                            leg_plane_to_3d, lateral_for_swing, hip_offset_mm,
                            calibration_frame, solver_to_hw, x_forward_sign)
from serial_manager import SERVO_MIN_US, SERVO_MAX_US
from workspace import compute_workspace
from arduino_link import ArduinoLink, add_cli_args

CFG = yaml.safe_load(open("linkage_config.yaml"))
BARS = [("O", "A"), ("A", "B"), ("Q", "B"), ("Q", "C"), ("Q", "P1"),
        ("C", "P2"), ("P1", "P2"), ("P1", "D")]
LIMITS = {i["name"]: (i.get("min_deg", -999), i.get("max_deg", 999)) for i in CFG["inputs"]}
SWING_MAX = float(CFG.get("hip_swing_max_deg", 30.0))
if calibration_frame(CFG) != "hardware":
    print("WARNING: linkage_config.yaml calibration_frame is "
          f"{calibration_frame(CFG)!r}, not 'hardware' - servo angles shown below are "
          "still the hardware frame, but pulses follow the config as written.")

state = {"leg": "FL"}

# ---- Arduino connection (see arduino_link.py)
ap = argparse.ArgumentParser(description="Interactive IK tester (+ live Arduino output)")
add_cli_args(ap)
args = ap.parse_args()
LINK = ArduinoLink(CFG)
LINK.connect_from_args(args)

print("Computing workspace (a few seconds)...")
WS = compute_workspace(CFG)[::4]
WSX = [p["x"] for p in WS]                # solver frame; flipped per leg when drawn
WSY = [p["y"] for p in WS]

# start at the first gait waypoint (stored in the solver frame)
X0, Z0 = CFG["gait"]["waypoints"][0]
X0 = x_forward_sign(CFG, "FL") * X0          # FL is selected first
X_RANGE = (-250, 250)                        # robot X; left and right legs reach opposite ways

fig = plt.figure(figsize=(17, 8))
ax2d = fig.add_axes([0.03, 0.36, 0.22, 0.58])
axyz = fig.add_axes([0.27, 0.36, 0.17, 0.58])
ax3d = fig.add_axes([0.45, 0.30, 0.31, 0.66], projection="3d")
axtxt = fig.add_axes([0.77, 0.36, 0.22, 0.58])
axtxt.axis("off")

s_x = Slider(fig.add_axes([0.10, 0.20, 0.50, 0.03]), "X fwd (mm)", *X_RANGE, valinit=X0, valstep=1)
s_y = Slider(fig.add_axes([0.10, 0.14, 0.50, 0.03]), "Y outward (mm)", -60, 60, valinit=0, valstep=1)
s_z = Slider(fig.add_axes([0.10, 0.08, 0.50, 0.03]), "Z height (mm)", -270, -50, valinit=Z0, valstep=1)
s_h = Slider(fig.add_axes([0.10, 0.02, 0.50, 0.03]), "Hip swing (deg)", -SWING_MAX, SWING_MAX,
             valinit=0, valstep=0.5)
radio = RadioButtons(fig.add_axes([0.72, 0.06, 0.10, 0.20]), leg_names(CFG))
mode_radio = RadioButtons(fig.add_axes([0.62, 0.02, 0.09, 0.14]), ["Y slider", "Hip slider"])
state["mode"] = "Y slider"
send_box = CheckButtons(fig.add_axes([0.84, 0.10, 0.14, 0.06]), ["Send to Arduino"],
                        [LINK.enabled])


def hw_angles(r):
    """Solver result -> hardware-frame angles (what the servos are calibrated in)."""
    horn, thigh = solver_to_hw(r["theta1"], r["theta_c"])
    return {"horn": horn, "thigh": thigh, "hip": r["hip"]}


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
    x, lat, z = s_x.val, s_y.val, s_z.val     # x: robot-forward
    FWD = x_forward_sign(CFG, leg)            # this leg: robot X = FWD * solver x
    xs = FWD * x                              # same target in the solver frame
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
        r = solve_leg_3d(CFG, leg, xs, z, lateral=lat)
        pts = fk_leg(CFG, leg, r["theta1"], r["theta_c"])["points"]
        pts = {k: {"x": FWD * p["x"], "y": p["y"]} for k, p in pts.items()}  # robot X
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
    ax2d.scatter([FWD * v for v in WSX], WSY, s=1, c="lightgray")
    for a, b in BARS:
        ax2d.plot([pts[a]["x"], pts[b]["x"]], [pts[a]["y"], pts[b]["y"]], "b-")
    reach = r["reach"]
    ax2d.scatter([x], [-reach], marker="x", c="g", s=80, label="planar target")
    ax2d.scatter([pts["D"]["x"]], [pts["D"]["y"]], c="r", label="foot")
    ax2d.set_xlim(*((-160, 280) if FWD > 0 else (-280, 160))); ax2d.set_ylim(-280, 80)
    if not r["valid"]:
        ax2d.text(0.5, 0.97, "OUT OF REACH - not sent,\nrobot holds its last pose",
                  transform=ax2d.transAxes, ha="center", va="top", color="red",
                  fontsize=10, weight="bold")
    ax2d.set_aspect("equal"); ax2d.set_xlabel("X (robot front ->)")
    ax2d.set_title("Leg plane (2D linkage)"); ax2d.legend(loc="upper right", fontsize=8)

    # ---- Y-Z: looking along the hip axis (front view), Y = outward from body
    def yz(p):
        return s * p[1], p[2]
    h0 = np.array(yz(p3["H0"]))
    n0 = np.linalg.norm(h0)
    if n0 > 1e-9:
        u_bar = h0 / n0                           # hip axis -> leg plane
    else:                                         # hip_offset_mm = 0: use the leg's own rotation
        phi = math.radians(r["hip"] - float(CFG.get("hip_fixed_deg", 90.0)))
        u_bar = np.array([math.cos(phi), math.sin(phi)])
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
    ax3d.set_xlim(*X_RANGE); ax3d.set_ylim(-160, 160); ax3d.set_zlim(-280, 40)
    ax3d.set_box_aspect((X_RANGE[1] - X_RANGE[0], 320, 320))   # equal mm scale on all axes, so 90 deg looks 90 deg
    ax3d.set_xlabel("X fwd"); ax3d.set_ylabel("Y (+left)"); ax3d.set_zlabel("Z up")
    ax3d.set_title("3D (green x = target, red = foot)")
    ax3d.view_init(elev, azim)

    # ---- text
    d = hw_angles(r)
    ch = CFG["legs"][leg]["channels"]
    us = LINK.mgr.compute_leg_us(CFG, leg, r["theta1"], r["theta_c"], r["hip"])  # exactly what is sent
    lines += [
        "SERVO ANGLES (hardware frame)",
        f"  thigh  ch{ch['theta_c']:<2} {d['thigh']:7.2f} deg  -> {us['theta_c']:4d} us",
        f"  horn   ch{ch['theta1']:<2} {d['horn']:7.2f} deg  -> {us['theta1']:4d} us",
        f"  hip    ch{ch['hip']:<2} {d['hip']:7.2f} deg  -> {us['hip']:4d} us",
        f"  (solver: theta1 {r['theta1']:.1f}  theta_c {r['theta_c']:.1f})",
        "",
        f"IK valid: {r['valid']}   planar err {r['error_mm']:.2f} mm",
        f"FK foot   X {foot[0]:7.1f}  Y {foot[1]:6.1f}  Z {foot[2]:7.1f}",
        f"3D error  {err3:.2f} mm",
        f"hip offset {np.linalg.norm(h0):.3f} mm, at {bar_leg_deg:.6f} deg to leg", "",
    ]
    warn = []
    blockers = []
    for name, val in (("theta1", r["theta1"]), ("theta_c", r["theta_c"])):
        lo, hi = LIMITS[name]
        if not lo <= val <= hi:
            warn.append(f"{name} outside limits {lo}..{hi}")
            blockers.append(f"{name} out of limits")
    for name, v in us.items():
        if v in (SERVO_MIN_US, SERVO_MAX_US):
            warn.append(f"{name} pulse clamped at {v} us (past servo range)")
            blockers.append(f"{name} pulse clamped")
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