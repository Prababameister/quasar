"""
run.py - drive the quadruped from the keyboard, no window or web server needed.

Uses the same parametric gaits as gait_sim.py (gait_generator.GaitGenerator):
the gait is stepped, solved (IK) and sent to the Arduino OUT_HZ times a
second, so the servos get every point along the foot paths, not just a few
waypoints.

    w : forward               s : backward
    a : strafe left           d : strafe right
    q : turn in place left    e : turn in place right
    j : jump (repeated jumps in place)
    i : march in place (step without moving)
    space / x : stop (ramp down, then stand)

    1-5 : gait used for w / s / q / e:  1 walk  2 trot  3 pace  4 bound  5 pronk
    + / - : faster / slower (25% steps)
    [ / ] : gait period (one full stride) -/+ 0.05 s
    Esc : stand, then quit        Ctrl-C : quit now (robot holds its pose)

HOLD TO MOVE: w a s d q e j i only act while the key is held down; let go
and the robot ramps down and stands. A terminal never reports key RELEASES,
only presses, but a held key auto-repeats, so a motion counts as held while
the repeats keep coming. Release is noticed --release-s after the last repeat
(default 0.15 s). Before the first repeat the terminal waits a moment (its
"repeat delay", usually 0.25-0.6 s), so a single tap moves for up to
--first-repeat-s (default 0.7 s); raise that if a held key stutters at the
start. Number keys, + - [ ] are one-shot settings. Changing gait (e.g. walking
-> strafing, or -> jump) first ramps the current gait down to standing, then
starts the new one, so the feet never jump between phase patterns.

Same safety as gait_sim.py's "Send to Arduino":
  * soft start - at startup the servos glide to the standing pose at
    SOFT_START_DEG_S instead of snapping there all at once
  * speed cap  - no servo is told to move faster than --max-deg-s
  * a frame with any unreachable leg is never sent (the robot holds its pose)

Usage:
    python run.py                        # auto-detect the serial port
    python run.py --port /dev/ttyACM0 [--baud 115200]
    python run.py --no-serial            # no robot: just run the keys + status line
"""

import argparse
import logging
import math
import os
import select
import sys
import termios
import time
import tty

import yaml

from arduino_link import ArduinoLink, add_cli_args
from gait_generator import GAITS, GaitGenerator

CONFIG_PATH = "linkage_config.yaml"
OUT_HZ = 50                     # gait steps + servo frames per second
HOLD_KEYS = set("wasdqeji")     # motion only while held (see HOLD TO MOVE)
US_PER_DEG = 2000 / 270         # servo pulse mapping: 0-270 deg -> 500-2500 us
SOFT_START_DEG_S = 45           # glide speed at startup
SETTLED_US = 10                 # soft start ends once every servo is this close

WALK_GAITS = {"1": "walk", "2": "trot", "3": "pace", "4": "bound", "5": "pronk"}

# Base speeds at speed factor 1.0 (the generator scales these down further if a
# stride would be longer than gait_gen.max_step_mm).
FWD_MM_S = 60.0
STRAFE_MM_S = 40.0
TURN_DEG_S = 30.0

# key -> (label, gait or None = the chosen walking gait, vx, vy, wz sign, step in place)
# vx / vy / wz here are -1..1 and get multiplied by the base speeds and speed factor.
MOTIONS = {
    "w": ("forward",           None,           1, 0, 0, False),
    "s": ("backward",          None,          -1, 0, 0, False),
    "a": ("strafing left",     "strafe_left",  1, 0, 0, False),   # strafe gaits turn vx sideways
    "d": ("strafing right",    "strafe_right", 1, 0, 0, False),
    "q": ("turning left",      None,           0, 0, 1, False),   # +wz = CCW = left
    "e": ("turning right",     None,           0, 0, -1, False),
    "i": ("marching in place", None,           0, 0, 0, True),
    "j": ("jumping",           "jump",         0, 0, 0, True),
    " ": ("stopped",           None,           0, 0, 0, False),
}
MOTIONS["x"] = MOTIONS[" "]


class Robot:
    """GaitGenerator + IK + rate-limited streaming to the Arduino."""

    def __init__(self, cfg, link, max_deg_s):
        self.cfg = cfg
        self.link = link
        self.gen = GaitGenerator(cfg)
        self.gen.set_gait("trot")
        self.walk_gait = "trot"
        self.max_deg_s = max_deg_s
        self.speed = 1.0
        self.motion = MOTIONS[" "]
        self.pending = None           # (gait, command) waiting for the old gait to ramp down
        self.soft_start = True
        self.slew_limited = False
        self.unreachable = []

    # ── commands ──────────────────────────────────────────────────────
    def set_motion(self, key):
        self.motion = MOTIONS[key]
        self._apply()

    def set_walk_gait(self, name):
        self.walk_gait = name
        self._apply()

    def change_speed(self, factor):
        self.speed = min(2.0, max(0.25, self.speed * factor))
        self._apply()

    def change_period(self, delta):
        p = self.gen.p
        p.cycle_time_s = min(5.0, max(0.3, round(p.cycle_time_s + delta, 2)))

    def _apply(self):
        _, gait, vx, vy, wz, in_place = self.motion
        gait = gait or self.walk_gait
        cmd = (vx * FWD_MM_S * self.speed if gait not in ("strafe_left", "strafe_right")
               else vx * STRAFE_MM_S * self.speed,
               vy * STRAFE_MM_S * self.speed,
               math.radians(wz * TURN_DEG_S * self.speed),
               in_place)
        if gait == self.gen.p.gait:
            self.pending = None
            self.gen.set_command(*cmd[:3], step_in_place=cmd[3])
        else:
            # different phase pattern: ramp the current gait down to standing first
            self.pending = (gait, cmd)
            self.gen.set_command(0, 0, 0, step_in_place=False)

    @property
    def standing(self):
        return self.gen.activity <= 0.0 and all(abs(v) < 1e-3 for v in self.gen.vel)

    # ── one control frame ─────────────────────────────────────────────
    def step(self, dt):
        if self.pending and self.standing:
            gait, cmd = self.pending
            self.pending = None
            self.gen.set_gait(gait)
            self.gen.phase = 0.0
            self.gen.set_command(*cmd[:3], step_in_place=cmd[3])
        res = self.gen.solve(self.gen.step(dt))
        self.unreachable = [leg for leg, r in res.items() if not r["valid"]]
        if self.unreachable or not self.link.connected:
            return
        self._send(res, dt)

    def _send(self, res, dt):
        mgr = self.link.mgr
        target = list(mgr._last_us)
        for leg, r in res.items():
            us = mgr.compute_leg_us(self.cfg, leg, r["theta1"], r["theta_c"], r["hip"])
            for joint, ch in self.cfg["legs"][leg]["channels"].items():
                target[ch] = us[joint]
        deg_s = SOFT_START_DEG_S if self.soft_start else self.max_deg_s
        step = max(1, int(deg_s * US_PER_DEG * dt))
        held, out = False, []
        for cur, want in zip(mgr._last_us, target):
            d = max(-step, min(step, want - cur))
            held |= d != want - cur
            out.append(cur + d)
        mgr._last_us[:] = out
        mgr._transmit("run")
        if self.soft_start and max(abs(a - b) for a, b in zip(out, target)) <= SETTLED_US:
            self.soft_start = False
        self.slew_limited = held and not self.soft_start

    def status(self):
        g, p = self.gen, self.gen.p
        if self.soft_start and self.link.connected:
            what = "standing up..."
        elif self.pending:
            what = f"{self.motion[0]} (stopping {p.gait} first)"
        else:
            what = self.motion[0]
        flags = []
        if g.limited:
            flags.append("SPEED LIMITED")
        if self.slew_limited:
            flags.append("SLEW-LIMITED")
        if self.unreachable:
            flags.append(f"UNREACHABLE {','.join(self.unreachable)} - not sent")
        if not self.link.connected:
            flags.append("no robot")
        return (f"{what:30s} gait {p.gait:12s} walk-gait {self.walk_gait:5s} "
                f"speed x{self.speed:.2f}  period {p.cycle_time_s:.2f}s  "
                f"phase {g.phase:.2f}  " + "  ".join(flags))


def status_line(text: str) -> None:
    # cbreak mode leaves output processing alone, but use \r to redraw in place
    sys.stdout.write(f"\r\x1b[K{text}")
    sys.stdout.flush()


def read_keys(fd, timeout):
    """All keys typed within `timeout` s. Arrow keys etc. (escape sequences) are
    dropped; a lone Esc comes back as "\\x1b"."""
    keys = []
    if not select.select([fd], [], [], max(0.0, timeout))[0]:
        return keys
    data = os.read(fd, 64).decode(errors="ignore")   # raw read: sys.stdin would buffer
    if data.endswith("\x1b"):                       # half an escape sequence? wait briefly
        if select.select([fd], [], [], 0.03)[0]:
            data += os.read(fd, 64).decode(errors="ignore")
    i = 0
    while i < len(data):
        c = data[i]
        if c == "\x1b" and i + 1 < len(data) and data[i + 1] in "[O":
            i += 3                                   # skip ESC [ X
            continue
        keys.append(c)
        i += 1
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_cli_args(ap)
    ap.add_argument("--max-deg-s", type=float, default=200.0,
                    help="servo speed cap, deg/s (default 200)")
    ap.add_argument("--release-s", type=float, default=0.15,
                    help="key counts as released this long after its last auto-repeat")
    ap.add_argument("--first-repeat-s", type=float, default=0.7,
                    help="wait this long for the FIRST auto-repeat (terminal repeat delay)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)  # serial_manager logs every frame at INFO

    if not sys.stdin.isatty():
        sys.exit("stdin is not a terminal - run this from an interactive shell")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    link = ArduinoLink(cfg)
    link.connect_from_args(args)          # also syncs channel values with the firmware's boot pose
    robot = Robot(cfg, link, args.max_deg_s)

    print(__doc__.split("Same safety")[0].split("\n\n", 2)[2].rstrip())
    print()

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    period = 1.0 / OUT_HZ
    quitting_at = None
    held = {"key": None, "seen": 0.0, "repeating": False}
    try:
        tty.setcbreak(fd)
        last = time.monotonic()
        while True:
            for ch in read_keys(fd, period - (time.monotonic() - last)):
                c = ch.lower()
                if ch == "\x1b" and quitting_at is None:
                    robot.set_motion(" ")
                    quitting_at = time.monotonic()
                elif quitting_at is not None:
                    continue
                elif c in HOLD_KEYS:
                    if c == held["key"]:              # auto-repeat: still held
                        held["repeating"] = True
                    else:
                        robot.set_motion(c)
                        held.update(key=c, repeating=False)
                    held["seen"] = time.monotonic()
                elif c in MOTIONS:                    # space / x: stop now
                    robot.set_motion(c)
                    held["key"] = None
                elif c in WALK_GAITS:
                    robot.set_walk_gait(WALK_GAITS[c])
                elif c in ("+", "="):
                    robot.change_speed(1.25)
                elif c in ("-", "_"):
                    robot.change_speed(1 / 1.25)
                elif c == "[":
                    robot.change_period(-0.05)
                elif c == "]":
                    robot.change_period(+0.05)
            now = time.monotonic()
            if held["key"] and now - held["seen"] > (
                    args.release_s if held["repeating"] else args.first_repeat_s):
                robot.set_motion(" ")                 # key released: ramp down and stand
                held["key"] = None
            if now - last < period:
                continue
            dt, last = min(now - last, 0.1), now
            try:
                robot.step(dt)
            except Exception as e:
                link.mgr.disconnect()
                status_line(f"send failed, disconnected: {e}")
                print()
                break
            if quitting_at is not None and (
                    robot.standing or now - quitting_at > 2 * robot.gen.p.cycle_time_s + 1):
                break
            status_line(robot.status())
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        print("\nStopped.")
        link.close()


if __name__ == "__main__":
    main()
