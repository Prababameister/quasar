"""
gait.py — dead-simple diagonal-trot walking for the quadruped.

The foot trajectory is defined once in the canonical left-leg frame (see
leg_kinematics.py) and used for every leg. The two diagonal pairs are driven a
half-cycle out of phase:

    phase 0.0 : FL, BR
    phase 0.5 : FR, BL

Within a cycle each foot spends `duty` of the time in stance (planted, moving
backward relative to the body to push it forward) and the rest in swing (lifted,
returning to the front). `direction` is +1 forward, -1 backward, 0 hold.

WalkController runs a background thread that, while walking, advances the phase
at `update_rate_hz`, solves IK for all four legs, and streams one 12-channel
frame per tick through the serial manager.
"""

import logging
import math
import threading
import time

from leg_kinematics import solve_leg, nominal_target, hip_fixed_deg

log = logging.getLogger("gait")

# Half-cycle phase offset between the two diagonal pairs of a trot.
_PHASE_OFFSET = {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}


def _gait_params(cfg: dict) -> dict:
    g = cfg.get("gait", {})
    return {
        "cycle_time_s":      float(g.get("cycle_time_s", 1.2)),
        "duty":              float(g.get("duty", 0.5)),
        "step_length_mm":    float(g.get("step_length_mm", 40.0)),
        "step_height_mm":    float(g.get("step_height_mm", 25.0)),
        "stance_x_mm":       float(g.get("stance_x_mm", 150.0)),
        "stance_y_mm":       float(g.get("stance_y_mm", -120.0)),
        "update_rate_hz":    float(g.get("update_rate_hz", 30.0)),
        "forward_axis_sign": float(g.get("forward_axis_sign", 1.0)),
    }


def foot_offset(phase: float, gp: dict, direction: int) -> tuple[float, float]:
    """
    Foot target (x, y) in the canonical left-leg frame for a given cycle phase.

    phase     : 0.0 .. 1.0 (wraps)
    gp        : dict from _gait_params()
    direction : +1 forward, -1 backward, 0 -> mid-stance hold

    Continuous across the stance<->swing boundary and the phase wrap.
    """
    phase %= 1.0
    duty = gp["duty"]
    L    = gp["step_length_mm"]
    H    = gp["step_height_mm"]

    if direction == 0:
        p, lift = 0.0, 0.0
    elif phase < duty:
        # Stance: foot travels front (+L/2) -> back (-L/2), on the ground.
        p    = L / 2.0 - (phase / duty) * L
        lift = 0.0
    else:
        # Swing: foot returns back (-L/2) -> front (+L/2), lifted.
        u    = (phase - duty) / (1.0 - duty)
        p    = -L / 2.0 + u * L
        lift = H * math.sin(math.pi * u)

    x = gp["stance_x_mm"] + gp["forward_axis_sign"] * direction * p
    y = gp["stance_y_mm"] + lift
    return x, y


class WalkController:
    """Owns a worker thread that streams trot commands to the serial manager."""

    def __init__(self, cfg: dict, serial_mgr):
        self._cfg = cfg
        self._mgr = serial_mgr
        self._gp = _gait_params(cfg)

        self._lock = threading.Lock()
        self._walking = False
        self._direction = 0
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
        Accepts any of the _gait_params keys.
        """
        with self._lock:
            for k, v in kw.items():
                if k in self._gp and v is not None:
                    self._gp[k] = float(v)
            return dict(self._gp)

    def params(self) -> dict:
        with self._lock:
            return dict(self._gp)

    # ── control ───────────────────────────────────────────────────────
    def start(self, direction: int) -> None:
        with self._lock:
            self._direction = int(direction)
            self._walking = direction != 0
        self._ensure_thread()

    def set_direction(self, direction: int) -> None:
        self.start(direction)

    def stop(self) -> None:
        with self._lock:
            self._walking = False
            self._direction = 0
        # Park the feet at mid-stance once, if we can.
        self._send_once(direction=0)

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

            self._send_frame(phase, direction, gp)
            time.sleep(dt)

    # ── send helpers ──────────────────────────────────────────────────
    def _leg_angles(self, phase: float, direction: int, gp: dict) -> dict:
        cfg = self._cfg
        hip = hip_fixed_deg(cfg)
        angles = {}
        for leg in cfg.get("legs", {}):
            ph = (phase + _PHASE_OFFSET.get(leg, 0.0)) % 1.0
            x, y = foot_offset(ph, gp, direction)
            res = solve_leg(cfg, leg, x, y)
            if not res.get("valid", False):
                log.warning("gait: %s target (%.1f, %.1f) unreachable "
                            "(err %.1f mm)", leg, x, y, res.get("error_mm", -1))
            angles[leg] = (res["theta1"], res["theta_c"], hip)
        return angles

    def _send_frame(self, phase: float, direction: int, gp: dict) -> None:
        if not self._mgr.connected:
            return
        try:
            self._mgr.send_legs(self._cfg, self._leg_angles(phase, direction, gp))
        except Exception as e:  # pragma: no cover - hardware faults
            log.error("gait: send failed: %s", e)

    def _send_once(self, direction: int) -> None:
        with self._lock:
            gp, phase = self._gp, self._phase
        self._send_frame(phase, direction, gp)
