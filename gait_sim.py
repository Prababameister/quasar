"""
gait_sim.py - gait generation simulator / visualizer.

Run from the quasar-1 folder:   python gait_sim.py
                                python gait_sim.py --port COM5   (also drive the robot)

Runs gait_generator.GaitGenerator in real time and runs the REAL solver
(solve_leg_3d) + FK for all four legs every frame, so what you see is what the
servos would be told to do.

Panels
------
  3D robot      : body + the four linkages from FK (body frame, ground plane dashed)
  Top view      : the robot walking in the world. Filled feet = on the ground,
                  hollow = swinging. Shaded = support polygon, dot = centre of
                  mass (green = statically stable, red = not). Gray dots =
                  footprints. A good gait leaves each footprint exactly where
                  the foot landed -> the "slip" readout stays ~0.
  Leg plane     : every leg's foot path over one cycle in the canonical solver
                  frame, on top of the reachable workspace. Red = unreachable.
  Gait diagram  : black = stance, per leg, over one cycle; red line = now.

Controls: gait type (strafe_left / strafe_right turn the vx slider into sideways
speed), body speed (vx fwd, vy left, turn), cycle time, duty factor
(0 = gait default), step height, stand height, sim speed. Keyboard:
  space = play/pause   arrows = drive (up/down vx, left/right turn)   r = reset
  [ / ] = gait period -/+ 0.05 s

ARDUINO: if connected, tick "Send to Arduino" to stream the same frames to the
robot (off by default). Frames with any unreachable leg are never sent.

STREAMING: the window only redraws a few times a second, so sending one pose
per redraw would make every servo jump to each pose at full speed. Instead,
every redraw steps the gait in small sub-steps (KEY_S) and queues each one as
a WAYPOINT (all 12 pulse widths, time-stamped). A background thread plays the
waypoints back in real time at OUT_HZ, LINEARLY INTERPOLATING the pulses
between neighbouring waypoints, so the Arduino gets every point in between.
Playback runs a little behind the simulation (about two redraws) so the next
waypoint is always already queued.

GAIT PERIOD: the "gait period s" slider (or [ and ] keys, 0.05 s steps) is
the time for one full stride, in real seconds (at sim speed 1.0).

CURRENT LIMITING: what goes to the servos is rate-limited so the robot doesn't
pull big current spikes (which brown out the supply / reset the Arduino):
  * soft start  - when sending starts (or resumes after a pause / unreachable
                  frame) the servos glide from where they are to the gait pose
                  at SOFT_START_DEG_S instead of all 12 snapping there at once.
  * speed limit - "max servo deg/s" caps how fast any servo is told to move.
                  Fast moves = big acceleration = big current. If the gait asks
                  for more than this (fast walk swings), the readout says
                  SLEW-LIMITED; slow the gait down (longer cycle) instead.
Lower current also comes from the gait itself: higher duty factor (more feet
share the weight), longer cycle time, lower step height.
"""

import argparse
import math
import os
import threading
from collections import deque
import time

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

from gait_generator import (GAITS, GaitGenerator, hip_position, stance_intervals,
                            support_margin, _convex_hull, x_forward_sign)
from leg_kinematics import fk_leg, leg_plane_to_3d, solver_to_hw
from workspace import compute_workspace
from arduino_link import ArduinoLink, add_cli_args

CONFIG_PATH = "linkage_config.yaml"
CFG = yaml.safe_load(open(CONFIG_PATH))
CFG_LOADED = {"at": time.strftime("%H:%M:%S"), "mtime": os.path.getmtime(CONFIG_PATH)}
BARS = [("O", "A"), ("A", "B"), ("Q", "B"), ("Q", "C"), ("Q", "P1"),
        ("C", "P2"), ("P1", "P2"), ("P1", "D")]
COLORS = {"FL": "tab:blue", "FR": "tab:orange", "BL": "tab:green", "BR": "tab:purple"}
FRAME_S = 1 / 30
US_PER_DEG = 2000 / 270                 # servo pulse mapping: 0-270 deg -> 500-2500 us
SOFT_START_DEG_S = 45                   # glide speed when sending (re)starts
SETTLED_US = 10                         # soft start ends once every servo is this close
OUT_HZ = 50                             # servo frames per second sent to the Arduino
KEY_S = 0.04                            # longest gait sub-step between two waypoints
MAX_TICK_S = 0.5                        # longer gaps (window dragged etc.) don't jump the gait
# free the arrow keys / r / l from matplotlib's default navigation shortcuts
for _k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.yscale"):
    plt.rcParams[_k] = [k for k in plt.rcParams[_k] if k not in ("left", "right", "r", "l")]

ap = argparse.ArgumentParser(description="Gait simulator (+ optional Arduino output)")
add_cli_args(ap)
args = ap.parse_args()
LINK = ArduinoLink(CFG)
LINK.connect_from_args(args)
LINK.set_enabled(False)                 # never start walking the real robot by surprise

GEN = GaitGenerator(CFG)
LEGS = GEN.legs

print("Computing workspace (a few seconds)...")
WS = compute_workspace(CFG)[::3]

# world state
state = {"playing": True, "pose": [0.0, 0.0, 0.0], "trail": [], "path": [],
         "was_stance": {leg: False for leg in LEGS}, "touch": {},
         "slip": {leg: 0.0 for leg in LEGS}, "last_wall": time.monotonic(),
         "path_key": None, "tick_ema": FRAME_S}

# ── layout ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9.5))
fig.canvas.manager.set_window_title("Gait simulator")
ax3d = fig.add_axes([0.0, 0.36, 0.35, 0.58], projection="3d")
axtop = fig.add_axes([0.40, 0.40, 0.26, 0.55])
axleg = fig.add_axes([0.70, 0.58, 0.28, 0.38])
axgait = fig.add_axes([0.70, 0.40, 0.28, 0.12])
axtxt = fig.add_axes([0.70, 0.01, 0.29, 0.31]); axtxt.axis("off")

p0 = GEN.p
def slider(y, label, lo, hi, v, step=None):
    return Slider(fig.add_axes([0.08, y, 0.36, 0.022]), label, lo, hi, valinit=v, valstep=step)

vmax = 150
s_vx = slider(0.30, "vx fwd mm/s", -vmax, vmax, 40, 1)
s_vy = slider(0.265, "vy left mm/s", -vmax, vmax, 0, 1)
s_wz = slider(0.23, "turn deg/s", -60, 60, 0, 1)
s_ct = slider(0.195, "gait period s", 0.3, 5.0, p0.cycle_time_s, 0.05)
s_df = slider(0.16, "duty (0=dflt)", 0.0, 0.95, p0.duty_factor, 0.01)
s_sh = slider(0.125, "step height mm", 0, 60, p0.step_height_mm, 1)
s_st = slider(0.09, "stand height mm", 110, 185, p0.stand_height_mm, 1)
s_slew = slider(0.055, "max servo deg/s", 30, 400, 200, 5)
s_sp = slider(0.02, "sim speed", 0.1, 2.0, 1.0, 0.05)

radio = RadioButtons(fig.add_axes([0.50, 0.115, 0.10, 0.27]), list(GAITS),
                     active=list(GAITS).index(p0.gait))
checks = CheckButtons(fig.add_axes([0.50, 0.01, 0.17, 0.10]),
                      ["Play", "Step in place", "Send to Arduino", "Calibration view"],
                      [True, False, False, False])

# ── 3D artists ────────────────────────────────────────────────────────────
leg_lines = {leg: [ax3d.plot([], [], [], "-", c=COLORS[leg], lw=2)[0] for _ in BARS]
             for leg in LEGS}
hipbar = {leg: ax3d.plot([], [], [], "k-", lw=3)[0] for leg in LEGS}
foot3d = {leg: ax3d.plot([], [], [], "o", c=COLORS[leg], ms=6)[0] for leg in LEGS}
body3d = ax3d.plot([], [], [], "k-", lw=2.5)[0]
ground3d = ax3d.plot([], [], [], "--", c="gray", lw=0.8)[0]
nose3d = ax3d.plot([], [], [], "r-", lw=3)[0]
ax3d.set_xlim(-250, 250); ax3d.set_ylim(-250, 250); ax3d.set_zlim(-220, 80)
ax3d.set_box_aspect((500, 500, 300))
ax3d.set_xlabel("X fwd"); ax3d.set_ylabel("Y left"); ax3d.set_zlabel("Z up")
ax3d.set_title("Robot (body frame, red = front)")
ax3d.view_init(22, -60)

# ── top view artists ──────────────────────────────────────────────────────
support = Polygon(np.zeros((3, 2)), closed=True, fc="tab:green", alpha=0.18, ec="tab:green")
axtop.add_patch(support)
trail_sc = axtop.scatter([], [], s=12, c="0.6", zorder=1)
path_ln, = axtop.plot([], [], "-", c="0.75", lw=1)
body_top, = axtop.plot([], [], "k-", lw=2)
nose_top, = axtop.plot([], [], "r-", lw=3)
feet_top = {leg: axtop.plot([], [], "o", ms=9, mec=COLORS[leg], mew=2)[0] for leg in LEGS}
com_top, = axtop.plot([], [], "o", ms=7)
axtop.set_aspect("equal"); axtop.set_title("Top view (world)")
axtop.set_xlabel("world X (mm)"); axtop.set_ylabel("world Y (mm)")

# ── leg plane ─────────────────────────────────────────────────────────────
axleg.scatter([p["x"] for p in WS], [p["y"] for p in WS], s=1, c="0.88")
leg_paths = {leg: axleg.plot([], [], "-", c=COLORS[leg], lw=1.5, label=leg)[0] for leg in LEGS}
leg_bad = axleg.plot([], [], "rx", ms=6)[0]
leg_now = {leg: axleg.plot([], [], "o", c=COLORS[leg], ms=6)[0] for leg in LEGS}
axleg.set_aspect("equal"); axleg.legend(loc="upper right", fontsize=7)
axleg.set_title("Foot paths, leg plane (canonical frame)", fontsize=9)


def zoom_leg_plane():
    p = GEN.p
    axleg.set_xlim(p.neutral_x_mm - 110, p.neutral_x_mm + 110)
    axleg.set_ylim(-p.stand_height_mm - 40, -p.stand_height_mm + 90)


# ── helpers ───────────────────────────────────────────────────────────────
def to_body(leg, px, py, hip_deg):
    """Solver-frame point of one leg -> body frame 3D."""
    fwd = x_forward_sign(CFG, leg)
    lx, ly, lz = leg_plane_to_3d(CFG, leg, fwd * px, py, hip_deg)
    hx, hy = hip_position(CFG, leg, GEN.p)
    return np.array([hx + lx, hy + ly, lz])


def world(xy):
    x, y, th = state["pose"]
    c, s = math.cos(th), math.sin(th)
    return x + c * xy[0] - s * xy[1], y + s * xy[0] + c * xy[1]


def read_controls():
    p = GEN.p
    p.cycle_time_s = s_ct.val
    p.duty_factor = s_df.val
    p.step_height_mm = s_sh.val
    p.stand_height_mm = s_st.val
    GEN.set_command(s_vx.val, s_vy.val, math.radians(s_wz.val),
                    step_in_place=checks.get_status()[1])


def draw_gait_diagram():
    axgait.clear()
    for i, leg in enumerate(LEGS):
        axgait.broken_barh(stance_intervals(GEN, leg), (i - 0.35, 0.7), fc="k")
    axgait.set_yticks(range(len(LEGS)), LEGS); axgait.set_xlim(0, 1); axgait.set_ylim(-0.6, len(LEGS) - 0.4)
    axgait.invert_yaxis()
    axgait.set_title(f"Gait diagram: {GEN.p.gait}, duty {GEN.p.duty:.2f} (black = stance)", fontsize=9)
    axgait.set_xlabel("cycle phase", fontsize=8)
    state["cursor"] = axgait.axvline(GEN.phase, c="r")


def update_paths():
    """Redraw the one-cycle foot paths when velocity / params change."""
    p = GEN.p
    key = (tuple(round(v, 1) for v in GEN.vel), p.gait, p.duty, p.step_height_mm,
           p.stand_height_mm, round(GEN.activity, 2))
    if key == state["path_key"]:
        return
    state["path_key"] = key
    bad = []
    for leg in LEGS:
        pts = GEN.cycle_path(leg, 40)
        leg_paths[leg].set_data([t.x for t in pts], [t.y for t in pts])
        for t in pts[::2]:
            if not GEN.solve({leg: t})[leg]["valid"]:
                bad.append((t.x, t.y))
    leg_bad.set_data([b[0] for b in bad], [b[1] for b in bad])
    state["path_bad"] = len(bad)


def frame_pulses(res):
    """All 12 channels' pulse widths (us) for one solved pose, via the same
    compute_leg_us() the Arduino frames are built with."""
    legs = CFG["legs"]
    out = list(LINK.mgr._last_us)
    for leg, r in res.items():
        us = LINK.mgr.compute_leg_us(CFG, leg, r["theta1"], r["theta_c"], r["hip"])
        for joint, ch in legs[leg]["channels"].items():
            out[ch] = us[joint]
    return out


class Streamer(threading.Thread):
    """
    Plays queued waypoints (wall time, 12 pulses) to the Arduino at OUT_HZ,
    linearly interpolating between the two waypoints around the playback time.
    The speed limit / soft start are applied on top, using the real time
    between frames.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.keys = deque()
        self.active = False           # set by tick(): sending allowed right now
        self.soft_start = True
        self.slew_limited = False
        self.error = None
        self.sent = 0
        self.underrun = False         # playback caught up with the newest waypoint

    def push(self, t, pulses):
        with self.lock:
            self.keys.append((t, pulses))

    def clear(self):
        with self.lock:
            self.keys.clear()
        self.soft_start = True
        self.slew_limited = False

    def _sample(self, play_t):
        """Linear interpolation of the waypoints at play_t (None if none queued)."""
        with self.lock:
            k = self.keys
            if not k:
                return None
            while len(k) >= 2 and k[1][0] <= play_t:   # drop waypoints already passed
                k.popleft()
            if self.soft_start:
                return list(k[-1][1])                  # glide to the newest pose first
            t0, a = k[0]
            if play_t <= t0 or len(k) == 1:
                self.underrun = play_t > t0
                return list(a)
            t1, b = k[1]
            self.underrun = False
            u = (play_t - t0) / max(t1 - t0, 1e-6)
            return [round(x + (y - x) * u) for x, y in zip(a, b)]

    def run(self):
        period = 1.0 / OUT_HZ
        last = time.monotonic()
        while True:
            time.sleep(max(0.0, period - (time.monotonic() - last)))
            now = time.monotonic()
            dt, last = min(now - last, 0.1), now
            if not (self.active and LINK.connected):
                continue
            delay = max(2 * state["tick_ema"], 2 * KEY_S)
            target = self._sample(now - delay)
            if target is None:
                continue
            mgr = LINK.mgr
            deg_s = SOFT_START_DEG_S if self.soft_start else s_slew.val
            step = max(1, int(deg_s * US_PER_DEG * dt))
            held, out = False, []
            for cur, want in zip(mgr._last_us, target):
                d = max(-step, min(step, want - cur))
                held |= d != want - cur
                out.append(cur + d)
            if out == mgr._last_us:
                continue
            try:
                mgr._last_us[:] = out
                mgr._transmit("gait_sim")
                self.sent += 1
            except Exception as e:
                self.error = str(e)[:30]
                self.active = False
                mgr.disconnect()
                continue
            if self.soft_start and max(abs(a - b) for a, b in zip(out, target)) <= SETTLED_US:
                self.soft_start = False
            self.slew_limited = held and not self.soft_start


STREAM = Streamer()
STREAM.start()


def reset(_=None):
    state.update(pose=[0.0, 0.0, 0.0], trail=[], path=[],
                 was_stance={leg: False for leg in LEGS}, touch={},
                 slip={leg: 0.0 for leg in LEGS})
    GEN.phase = 0.0
    GEN.vel = [0.0, 0.0, 0.0]


# ── main tick ─────────────────────────────────────────────────────────────
def tick(_frame):
    now = time.monotonic()
    t_prev = state["last_wall"]
    wall = min(now - t_prev, MAX_TICK_S)
    state["last_wall"] = now
    state["tick_ema"] += 0.2 * (wall - state["tick_ema"])
    read_controls()
    playing = checks.get_status()[0]
    want_send = checks.get_status()[2]
    streaming = LINK.connected and want_send and playing
    dt = wall * s_sp.val if playing else 0.0

    # Step the gait in sub-steps of at most KEY_S. Each sub-step is a waypoint
    # for the streamer (time-stamped at the wall time it belongs to); the last
    # one is also what gets drawn.
    n = max(1, math.ceil(dt / KEY_S)) if dt > 0 else 1
    x, y, th = state["pose"]
    bad = []
    for k in range(1, n + 1):
        sub = dt / n
        targets = GEN.step(sub) if dt > 0 else GEN.targets()
        res = GEN.solve(targets)
        vx, vy, wz = GEN.vel                       # integrate body pose with the velocity used
        x += (vx * math.cos(th) - vy * math.sin(th)) * sub
        y += (vx * math.sin(th) + vy * math.cos(th)) * sub
        th += wz * sub
        bad = [leg for leg, r in res.items() if not r["valid"]]
        if streaming and dt > 0:
            if bad:
                STREAM.clear()                    # never send it; glide back in after
            else:
                STREAM.push(now - wall + wall * k / n, frame_pulses(res))
    state["pose"] = [x, y, th]
    if dt > 0:
        state["path"].append((x, y)); state["path"] = state["path"][-600:]

    # ---- 3D
    feet_body = {}
    for leg in LEGS:
        r = res[leg]
        pts = fk_leg(CFG, leg, r["theta1"], r["theta_c"])["points"]
        p3 = {k: to_body(leg, v["x"], v["y"], r["hip"]) for k, v in pts.items()}
        for ln, (a, b) in zip(leg_lines[leg], BARS):
            ln.set_data_3d(*zip(p3[a], p3[b]))
        hx, hy = hip_position(CFG, leg, GEN.p)
        hipbar[leg].set_data_3d(*zip((hx, hy, 0.0), to_body(leg, 0.0, 0.0, r["hip"])))
        foot3d[leg].set_data_3d([p3["D"][0]], [p3["D"][1]], [p3["D"][2]])
        feet_body[leg] = p3["D"]
    hips = {leg: hip_position(CFG, leg, GEN.p) for leg in LEGS}
    ring = [hips[k] for k in ("FL", "FR", "BR", "BL") if k in hips]
    ring.append(ring[0])
    body3d.set_data_3d([h[0] for h in ring], [h[1] for h in ring], [0] * len(ring))
    L = GEN.p.body_length_mm / 2
    nose3d.set_data_3d([L, L + 40], [0, 0], [0, 0])
    gz = ground_z(targets)
    g = 230
    ground3d.set_data_3d([-g, g, g, -g, -g], [-g, -g, g, g, -g], [gz] * 5)

    # ---- contacts, slip, footprints
    stance_pts = []
    for leg, t in targets.items():
        fb = feet_body[leg]
        fw = world(fb[:2])
        if t.stance:
            stance_pts.append((fb[0], fb[1]))
            if not state["was_stance"][leg]:            # touchdown
                state["touch"][leg] = fw
                state["trail"].append(fw); state["trail"] = state["trail"][-60:]
                state["slip"][leg] = 0.0
            elif leg in state["touch"]:
                state["slip"][leg] = max(state["slip"][leg], math.dist(fw, state["touch"][leg]))
        state["was_stance"][leg] = t.stance
        line = feet_top[leg]
        line.set_data([fw[0]], [fw[1]])
        line.set_markerfacecolor(COLORS[leg] if t.stance else "white")

    margin = support_margin(stance_pts)
    if len(stance_pts) >= 3:
        support.set_xy([world(p) for p in _hull(stance_pts)])
    elif len(stance_pts) == 2:
        support.set_xy([world(stance_pts[0]), world(stance_pts[1]), world(stance_pts[1])])
    else:
        support.set_xy(np.zeros((3, 2)))
    com_top.set_data([x], [y]); com_top.set_color("tab:green" if margin > 0 else "tab:red")

    ring_w = [world(h) for h in ring]
    body_top.set_data([p[0] for p in ring_w], [p[1] for p in ring_w])
    nose_top.set_data(*zip(world((L, 0)), world((L + 50, 0))))
    if state["trail"]:
        trail_sc.set_offsets(state["trail"])
    if state["path"]:
        path_ln.set_data(*zip(*state["path"]))
    axtop.set_xlim(x - 280, x + 280); axtop.set_ylim(y - 280, y + 280)

    # ---- leg plane + gait diagram
    zoom_leg_plane()
    update_paths()
    for leg, t in targets.items():
        leg_now[leg].set_data([t.x], [t.y])
    if state.get("gait_key") != (GEN.p.gait, GEN.p.duty):
        state["gait_key"] = (GEN.p.gait, GEN.p.duty)
        draw_gait_diagram()
    state["cursor"].set_xdata([GEN.phase, GEN.phase])

    # ---- Arduino
    if LINK.enabled != want_send:
        LINK.set_enabled(want_send)
    if streaming != STREAM.active:
        if streaming:
            STREAM.clear()                # fresh start: soft start, empty queue
        STREAM.active = streaming
    if not LINK.connected:
        ard = ("ARDUINO: not connected" if not STREAM.error
               else f"ARDUINO: send failed, disconnected ({STREAM.error})")
    elif not want_send:
        ard = f"ARDUINO: {LINK.mgr.port_name} (sending off)"
    elif bad:
        ard = f"ARDUINO: NOT SENT ({', '.join(bad)} unreachable)"
    elif not playing:
        ard = f"ARDUINO: {LINK.mgr.port_name} paused"
    else:
        ard = (f"ARDUINO: {LINK.mgr.port_name} "
               + ("soft start..." if STREAM.soft_start else f"streaming {OUT_HZ} Hz, interpolated")
               + ("  SLEW-LIMITED" if STREAM.slew_limited else "")
               + ("  UNDERRUN" if STREAM.underrun and not STREAM.soft_start else ""))

    # ---- text
    p = GEN.p
    lines = [
        f"{p.gait.upper()}  duty {p.duty:.2f}  period {p.cycle_time_s:.2f}s  phase {GEN.phase:.2f}"
        f"  ({1 / max(state['tick_ema'], 1e-3):.0f} fps)",
        f"vel  vx {vx:6.1f}  vy {vy:6.1f} mm/s  turn {math.degrees(wz):5.1f} deg/s"
        + ("  (LIMITED)" if GEN.limited else ""),
        f"max straight speed {GEN.max_speed_mm_s():.0f} mm/s  (max_step {p.max_step_mm:.0f} mm)",
        f"feet down {len(stance_pts)}   stability margin "
        + (f"{margin:6.1f} mm" if math.isfinite(margin) else "  none (airborne)")
        + ("  STABLE" if margin > 0 else "  not static-stable"),
        f"unreachable on path: {state.get('path_bad', 0)}", "",
    ]
    if p.gait == "jump":
        ji = GEN.jump_info()
        early = ji["air_s"] < ji["planned_air_s"]
        lines[-1:-1] = [
            f"JUMP takeoff {ji['takeoff_mm_s']:.0f} mm/s  apex {ji['apex_mm']:.1f} mm",
            f"     air time {ji['air_s'] * 1000:.0f} ms vs planned {ji['planned_air_s'] * 1000:.0f} ms"
            + ("  LANDS EARLY" if early else ""),
        ]
    lines += [
        "leg  ph   S/s   horn  thigh   hip   slip",
    ]
    for leg in LEGS:
        r, t = res[leg], targets[leg]
        horn, thigh = solver_to_hw(r["theta1"], r["theta_c"], CFG, leg)
        lines.append(f"{leg:3s} {t.phase:4.2f}  {'S' if t.stance else 's'}  {horn:6.1f} {thigh:6.1f} "
                     f"{r['hip']:5.1f} {state['slip'][leg]:5.1f}" + ("  UNREACH" if not r["valid"] else ""))
    lines += ["", ard, "", "space play  arrows drive  r reset  [ ] period  l reload"]
    size = 8.5
    if checks.get_status()[3]:
        lines, size = calibration_lines(res, ard), 7.5
    axtxt.clear(); axtxt.axis("off")
    axtxt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=size)
    return []


def calibration_lines(res, ard):
    """
    Per leg and servo: the calibration points exactly as read from the config
    (deg0 -> us0, deg1 -> us1), then the current hardware angle and the pulse
    SerialManager.compute_leg_us() gives for it -- the same function that
    builds the frame sent to the Arduino.
    """
    stale = os.path.getmtime(CONFIG_PATH) != CFG_LOADED["mtime"]
    lines = [f"CALIBRATION  ({CONFIG_PATH} read {CFG_LOADED['at']}"
             + (", FILE CHANGED SINCE - press l)" if stale else ")"),
             f"frame: {CFG.get('calibration_frame', 'solver')}   "
             "angle -> us is what would be sent", "",
             "leg servo  ch  cal: deg0 -> us0   deg1 -> us1   now: deg  ->   us"]
    for leg in LEGS:
        lc, r = CFG["legs"][leg], res[leg]
        horn, thigh = solver_to_hw(r["theta1"], r["theta_c"], CFG, leg)
        us = LINK.mgr.compute_leg_us(CFG, leg, r["theta1"], r["theta_c"], r["hip"])
        off = lc.get("offsets", {})
        now = {"theta1": (horn, "horn", off.get("theta1_deg", 0.0)),
               "theta_c": (thigh, "thigh", off.get("thetac_deg", 0.0)),
               "hip": (r["hip"], "hip", off.get("hip_deg", 0.0))}
        for i, joint in enumerate(("theta1", "theta_c", "hip")):
            c = lc["calibration"][joint]
            deg, name, o = now[joint]
            lines.append(f"{leg if i == 0 else '':3s} {name:5s} {lc['channels'][joint]:3d}  "
                         f"{c['deg0']:6.1f} -> {c['us0']:4d}  {c['deg1']:6.1f} -> {c['us1']:4d}"
                         f"   {deg + o:6.1f} -> {us[joint]:4d}"
                         + (f"  (incl. offset {o:+.1f})" if o else ""))
        extra = [f"{k} {lc[k]:+g}" for k in ("x_forward_sign", "hip_dir", "horn_dir") if k in lc]
        lines.append(f"    {'  '.join(extra)}" + ("   UNREACHABLE" if not r["valid"] else ""))
    return lines + ["", ard + "      l = reload config from disk"]


def reload_config():
    """Re-read linkage_config.yaml in place (GEN, LINK and the solver share CFG).
    Calibration, offsets and directions take effect at once; gait_gen params,
    body size and the workspace plot keep their startup values."""
    try:
        new = yaml.safe_load(open(CONFIG_PATH))
    except Exception as e:
        print(f"Config reload failed, keeping the old one: {e}")
        return
    CFG.clear(); CFG.update(new)
    CFG_LOADED.update(at=time.strftime("%H:%M:%S"), mtime=os.path.getmtime(CONFIG_PATH))
    state["path_key"] = None
    print(f"Reloaded {CONFIG_PATH} at {CFG_LOADED['at']}")


def ground_z(targets):
    """
    Ground height in the body frame. Normally the stand height. In the jump gait
    it follows the feet while they're down (crouching lowers the body) and, in the
    air, drops away by the ballistic rise of the body after takeoff.
    """
    p = GEN.p
    if p.gait != "jump":
        return -p.stand_height_mm
    if any(t.stance for t in targets.values()):
        return min(t.y for t in targets.values() if t.stance)
    takeoff_y = -p.stand_height_mm - p.jump_extend_mm * GEN.activity
    return takeoff_y - GEN.body_rise()


def _hull(pts):
    return _convex_hull(pts)


def on_gait(label):
    GEN.set_gait(label)
    s_df.set_val(0.0)                     # back to that gait's default duty
    strafe = GAITS[label].get("strafe")
    s_vx.label.set_text({1: "strafe L mm/s", -1: "strafe R mm/s"}.get(strafe, "vx fwd mm/s"))
    if label == "jump" and not checks.get_status()[1]:
        checks.set_active(1)              # jumping happens in place: start it


def on_key(ev):
    if ev.key == " ":
        checks.set_active(0)
    elif ev.key == "up":
        s_vx.set_val(min(vmax, s_vx.val + 10))
    elif ev.key == "down":
        s_vx.set_val(max(-vmax, s_vx.val - 10))
    elif ev.key == "left":
        s_wz.set_val(min(60, s_wz.val + 5))
    elif ev.key == "right":
        s_wz.set_val(max(-60, s_wz.val - 5))
    elif ev.key == "r":
        reset()
    elif ev.key == "l":
        reload_config()
    elif ev.key == "[":
        s_ct.set_val(max(s_ct.valmin, round(s_ct.val - 0.05, 2)))
    elif ev.key == "]":
        s_ct.set_val(min(s_ct.valmax, round(s_ct.val + 0.05, 2)))


radio.on_clicked(on_gait)
fig.canvas.mpl_connect("key_press_event", on_key)
anim = FuncAnimation(fig, tick, interval=int(FRAME_S * 1000), blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    LINK.close()
