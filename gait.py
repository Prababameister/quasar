"""
gait.py — diagonal-trot walking for the quadruped, driven by a hand-placed
foot path.

The foot path is a closed loop of waypoints, defined once in the canonical
left-leg frame (see leg_kinematics.py) and used for every leg. Walking one
cycle traces the loop waypoints[0] -> waypoints[1] -> ... -> waypoints[-1] ->
waypoints[0], spending an equal fraction of the cycle on each segment and
linearly interpolating the foot position within it. The two diagonal pairs
are driven a half-cycle out of phase:

    phase 0.0 : FL, BR
    phase 0.5 : FR, BL

`direction` is +1 to walk the loop forward, -1 to walk it in reverse, 0 to
hold at waypoints[0]. `turn` (+1 clockwise / right, -1 counter-clockwise /
left) rotates in place using the same gait: the two sides are kept in the same
phase but the loop is walked forward on one side and in reverse on the other.

Reversing a loop reflects the phase about the centre of the stance (ground)
segment, so a reversed leg spends exactly the same part of the cycle on the
ground as a forward one. That keeps diagonal pairs landing together and the
two sides' stance/swing timing aligned when they run in opposite directions.

WalkController runs a background thread that, while walking, advances the
phase at `update_rate_hz`, solves IK for all four legs, and streams one
12-channel frame per tick through the serial manager.
"""

import logging
import math
import threading
import time

from leg_kinematics import solve_leg, hip_fixed_deg, leg_side

log = logging.getLogger("gait")

# Half-cycle phase offset between the two diagonal pairs of a trot.
_PHASE_OFFSET = {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}


def _stance_center(waypoints: list) -> float:
    """
    Phase at the middle of the stance segment: the loop segment lying lowest
    (smallest mean y), which is where the foot is on the ground.
    """
    n = len(waypoints)
    if n < 2:
        return 0.0
    lowest = min(range(n), key=lambda i: waypoints[i][1] + waypoints[(i + 1) % n][1])
    return (lowest + 0.5) / n


def _gait_params(cfg: dict) -> dict:
    g = cfg.get("gait", {})
    waypoints = [(float(p[0]), float(p[1])) for p in g.get("waypoints", [])]
    return {
        "stance_center":     _stance_center(waypoints),
        "cycle_time_s":      float(g.get("cycle_time_s", 1.2)),
        "update_rate_hz":    float(g.get("update_rate_hz", 30.0)),
        "forward_axis_sign": float(g.get("forward_axis_sign", 1.0)),
        "waypoints":         waypoints,
    }


def leg_direction(side: str, direction: int, turn: int) -> int:
    """Per-leg walk direction: a turn drives left and right sides oppositely."""
    if turn == 0:
        return direction
    return turn if side == "left" else -turn


def foot_offset(phase: float, gp: dict, direction: int) -> tuple[float, float]:
    """
    Foot target (x, y) in the canonical left-leg frame for a given cycle phase.

    phase     : 0.0 .. 1.0 (wraps)
    gp        : dict from _gait_params()
    direction : +1 walk the loop forward, -1 in reverse, 0 -> hold at waypoints[0]

    Continuous across every segment boundary and the phase wrap.
    """
    wp = gp["waypoints"]
    n = len(wp)
    if n == 0:
        return 0.0, 0.0
    if n == 1 or direction == 0:
        return wp[0]

    eff_forward = direction > 0
    if gp["forward_axis_sign"] < 0:
        eff_forward = not eff_forward

    p = phase % 1.0
    if not eff_forward:
        p = (2.0 * gp.get("stance_center", 0.5) - p) % 1.0

    seg_f = p * n
    i = int(seg_f) % n
    u = seg_f - math.floor(seg_f)
    ax, ay = wp[i]
    bx, by = wp[(i + 1) % n]
    return ax + (bx - ax) * u, ay + (by - ay) * u


class WalkController:
    """Owns a worker thread that streams trot commands to the serial manager."""

    def __init__(self, cfg: dict, serial_mgr):
        self._cfg = cfg
        self._mgr = serial_mgr
        self._gp = _gait_params(cfg)

        self._lock = threading.Lock()
        self._walking = False
        self._direction = 0
        self._turn = 0
        self._phase = 0.0

        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    # ── config ────────────────────────────────────────────────────────
    def refresh_config(self, cfg: dict) -> None:
        """Pick up gait/leg changes after POST /config/reload."""
        with self._lock:
            self._cfg = cfg
            self._gp = _gait_params(cfg)

    def set_params(self, **kw) -> dict:
        """
        Live-tune gait params without a config reload (not persisted to YAML).
        Accepts any of the _gait_params scalar keys (not waypoints).
        """
        with self._lock:
            for k, v in kw.items():
                if k in self._gp and k != "waypoints" and v is not None:
                    self._gp[k] = float(v)
            return dict(self._gp)

    def params(self) -> dict:
        with self._lock:
            return dict(self._gp)

    # ── control ───────────────────────────────────────────────────────
    def start(self, direction: int, turn: int = 0) -> None:
        """direction: +1 fwd / -1 back; turn: +1 right / -1 left (overrides direction)."""
        direction = max(-1, min(1, int(direction)))
        turn = max(-1, min(1, int(turn)))
        with self._lock:
            n_waypoints = len(self._gp["waypoints"])
        moving = direction != 0 or turn != 0
        if moving and n_waypoints < 2:
            raise ValueError("Need at least 2 gait waypoints — record and save a path first")
        with self._lock:
            self._direction = 0 if turn else direction
            self._turn = turn
            self._walking = moving
        self._ensure_thread()

    def set_direction(self, direction: int, turn: int = 0) -> None:
        self.start(direction, turn)

    def stop(self) -> None:
        with self._lock:
            self._walking = False
            self._direction = 0
            self._turn = 0
        # Park the feet at waypoints[0] once, if we can.
        self._send_once(direction=0, turn=0)

    def shutdown(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t:
            t.join(timeout=1.0)

    def status(self) -> dict:
        with self._lock:
            return {
                "walking": self._walking,
                "direction": self._direction,
                "turn": self._turn,
                "phase": round(self._phase, 4),
            }

    # ── worker ────────────────────────────────────────────────────────
    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="walk", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop_evt.is_set():
            with self._lock:
                gp = self._gp
                walking = self._walking
                direction = self._direction
                turn = self._turn

            dt = 1.0 / max(1.0, gp["update_rate_hz"])
            now = time.monotonic()
            elapsed = now - last
            last = now

            if not walking:
                # Nothing to do; let manual control have the bus.
                time.sleep(dt)
                continue

            with self._lock:
                self._phase = (self._phase + elapsed / gp["cycle_time_s"]) % 1.0
                phase = self._phase

            self._send_frame(phase, direction, turn, gp)
            time.sleep(dt)

    # ── send helpers ──────────────────────────────────────────────────
    def _leg_angles(self, phase: float, direction: int, turn: int, gp: dict) -> dict:
        cfg = self._cfg
        hip = hip_fixed_deg(cfg)
        angles = {}
        for leg in cfg.get("legs", {}):
            ph = (phase + _PHASE_OFFSET.get(leg, 0.0)) % 1.0
            ldir = leg_direction(leg_side(cfg, leg), direction, turn)
            x, y = foot_offset(ph, gp, ldir)
            res = solve_leg(cfg, leg, x, y)
            if not res.get("valid", False):
                log.warning("gait: %s target (%.1f, %.1f) unreachable "
                            "(err %.1f mm)", leg, x, y, res.get("error_mm", -1))
            angles[leg] = (res["theta1"], res["theta_c"], hip)
        return angles

    def _send_frame(self, phase: float, direction: int, turn: int, gp: dict) -> None:
        if not self._mgr.connected:
            return
        try:
            self._mgr.send_legs(self._cfg, self._leg_angles(phase, direction, turn, gp))
        except Exception as e:  # pragma: no cover - hardware faults
            log.error("gait: send failed: %s", e)

    def _send_once(self, direction: int, turn: int) -> None:
        with self._lock:
            gp, phase = self._gp, self._phase
        self._send_frame(phase, direction, turn, gp)
