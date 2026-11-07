"""
gait_generator.py - parametric gait generation for the quadruped.

Unlike gait.py (which replays a hand-clicked waypoint loop with equal time per
segment), this module builds every foot trajectory from a few numbers:

    gait          : walk | trot | pace | bound | pronk | strafe_left | strafe_right | jump
                    (sets the leg phase offsets; strafe modes walk sideways at speed vx;
                    jump also swaps in a crouch -> push -> flight height profile)
    cycle_time_s  : one full stride
    duty_factor   : fraction of the cycle each foot is ON THE GROUND
    step_height   : how high a swinging foot lifts
    stand height  : how far below the hips the feet are in stance
    body velocity : vx (forward), vy (left) in mm/s, wz (turn, CCW +) in rad/s

Per leg, per cycle (leg phase p in [0, 1)):

    STANCE  (p < duty)   foot moves in a straight line, at constant speed,
                         from +D/2 to -D/2 at stand height. D is chosen so the
                         foot's speed relative to the body exactly cancels the
                         body's motion -> the foot does not slip on the ground.
    SWING   (p >= duty)  foot returns from -D/2 to +D/2 along a cycloid:
                         zero velocity at lift-off and touchdown, peak height
                         step_height at mid-swing.

The stance displacement D for a leg is its velocity relative to the body
(body velocity + yaw rate x foot position) times the stance time, so walking,
strafing and turning are all the same code path, and they can be combined.

Frames
------
Body frame : origin at the body centre, +X = robot front, +Y = robot left,
             +Z = up. Hip positions come from gait_gen.body_length_mm /
             body_width_mm in linkage_config.yaml.
Solver frame (per leg) : what solve_leg_3d takes. x = canonical linkage x,
             converted to robot-forward with gait_gen.x_forward_sign;
             y = height (negative = below hip); lateral = mm OUTWARD of the
             leg's neutral plane.

Use it as a library (see GaitGenerator, used by gait_sim.py) or run it on the
robot directly:

    python gait_generator.py --gait trot --vx 40 --seconds 6
    python gait_generator.py --gait walk --vx 30 --dry-run      # print, don't send
"""

import argparse
import math
import time
from dataclasses import dataclass, fields

from leg_kinematics import solve_leg_3d, leg_side, hip_offset_mm


def x_forward_sign(cfg: dict, leg: str) -> float:
    """
    +1 if the linkage's +x (knee side) faces the robot's front for this leg, -1 if
    the back. Read from gait_gen.x_forward_sign (a number, or a per-leg dict) so the
    gait tools can differ from legs.<leg>.x_forward_sign, which ik_interactive.py uses.
    Falls back to the per-leg value if gait_gen doesn't set it.
    """
    v = (cfg.get("gait_gen") or {}).get("x_forward_sign")
    if isinstance(v, dict):
        v = v.get(leg)
    if v is None:
        v = cfg["legs"][leg].get("x_forward_sign", 1.0)
    v = float(v)
    if v not in (1.0, -1.0):
        raise ValueError(f"gait_gen.x_forward_sign for {leg} must be 1 or -1, got {v}")
    return v

# Phase offset of each leg (added to the global phase) and default duty factor.
# A leg swings when its phase is in [duty, 1).
GAITS = {
    # static crawl, swing order LH -> LF -> RH -> RF, one foot in the air at a time
    "walk":  {"offsets": {"BL": 0.0, "FL": 0.75, "BR": 0.5, "FR": 0.25}, "duty": 0.8},
    # diagonal pairs together
    "trot":  {"offsets": {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}, "duty": 0.6},
    # same-side pairs together
    "pace":  {"offsets": {"FL": 0.0, "BL": 0.0, "FR": 0.5, "BR": 0.5}, "duty": 0.6},
    # front pair / back pair together
    "bound": {"offsets": {"FL": 0.0, "FR": 0.0, "BL": 0.5, "BR": 0.5}, "duty": 0.6},
    # all four together (hopping)
    "pronk": {"offsets": {"FL": 0.0, "FR": 0.0, "BL": 0.0, "BR": 0.0}, "duty": 0.5},
    # sideways walking on trot timing. The forward-speed command (vx) is turned
    # into sideways motion: vy = strafe * vx (vy +left), and any vy command is
    # ignored. Turning (wz) still applies. The feet move sideways via the hip.
    "strafe_left":  {"offsets": {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}, "duty": 0.6,
                     "strafe": +1},
    "strafe_right": {"offsets": {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}, "duty": 0.6,
                     "strafe": -1},
    # repeated vertical jumps, all four together: crouch -> fast push -> tuck in
    # the air -> land. "Stance" = on the ground (crouch + push), "swing" = airborne.
    "jump":  {"offsets": {"FL": 0.0, "FR": 0.0, "BL": 0.0, "BR": 0.0}, "duty": 0.8},
}

G_MM_S2 = 9810.0


def _smooth(u: float) -> float:
    """Smoothstep 0..1 with zero slope at both ends."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


@dataclass
class GaitParams:
    gait: str = "trot"
    cycle_time_s: float = 0.8
    duty_factor: float = 0.0          # 0 -> use the gait's default
    step_height_mm: float = 25.0
    neutral_x_mm: float = -5.0        # canonical solver-frame x of the foot at mid-stance
    stand_height_mm: float = 155.0    # feet this far below the hips in stance
    stance_width_mm: float = 0.0      # extra outward foot offset (mm)
    max_step_mm: float = 60.0         # longest allowed stance stroke; speeds are scaled down to fit
    body_length_mm: float = 200.0     # front hip axis to back hip axis
    body_width_mm: float = 100.0      # left hip axis to right hip axis
    max_accel_mm_s2: float = 300.0    # velocity ramp limit
    max_yaw_accel_rad_s2: float = 3.0
    update_rate_hz: float = 30.0
    # jump gait only (defaults come from the jump: block if these aren't set)
    jump_crouch_mm: float = 40.0      # feet pulled UP this far toward the body to load
    jump_extend_mm: float = 20.0      # feet pushed DOWN this far past stand height at takeoff
    jump_crouch_frac: float = 0.88    # fraction of ground time spent crouching (rest = a short, fast push)

    @property
    def duty(self) -> float:
        d = self.duty_factor if self.duty_factor > 0 else GAITS[self.gait]["duty"]
        return min(max(d, 0.05), 0.95)

    @property
    def stance_time_s(self) -> float:
        return self.duty * self.cycle_time_s


def load_params(cfg: dict) -> GaitParams:
    """GaitParams from the gait_gen: block of linkage_config.yaml (missing keys -> defaults)."""
    block = cfg.get("gait_gen") or {}
    kw = {}
    j = cfg.get("jump") or {}
    try:                                   # jump: block points are absolute [x, y]
        ny = float(j["neutral"][1])
        kw["jump_crouch_mm"] = float(j["crouch"][1]) - ny
        kw["jump_extend_mm"] = ny - float(j["extend"][1])
    except (KeyError, TypeError, IndexError):
        pass
    for f in fields(GaitParams):
        if f.name in block and block[f.name] is not None:
            kw[f.name] = str(block[f.name]) if f.name == "gait" else float(block[f.name])
    p = GaitParams(**kw)
    if p.gait not in GAITS:
        raise ValueError(f"gait_gen.gait must be one of {list(GAITS)}, got {p.gait!r}")
    return p


# ── geometry ──────────────────────────────────────────────────────────────
def side_sign(cfg: dict, leg: str) -> float:
    return 1.0 if leg_side(cfg, leg) == "left" else -1.0


def hip_position(cfg: dict, leg: str, p: GaitParams) -> tuple[float, float]:
    """Hip axis position in the body frame (X front, Y left)."""
    fx = 1.0 if leg.upper().startswith("F") else -1.0
    return fx * p.body_length_mm / 2, side_sign(cfg, leg) * p.body_width_mm / 2


def nominal_foot(cfg: dict, leg: str, p: GaitParams) -> tuple[float, float]:
    """Body-frame XY of the foot at mid-stance."""
    hx, hy = hip_position(cfg, leg, p)
    s = side_sign(cfg, leg)
    fwd = x_forward_sign(cfg, leg) * p.neutral_x_mm
    return hx + fwd, hy + s * (hip_offset_mm(cfg) + p.stance_width_mm)


def _cycloid(s: float) -> tuple[float, float]:
    """Swing profile, s in [0,1] -> (horizontal 0..1, vertical 0..1..0)."""
    a = 2.0 * math.pi * s
    return s - math.sin(a) / (2.0 * math.pi), (1.0 - math.cos(a)) / 2.0


@dataclass
class FootTarget:
    leg: str
    phase: float       # this leg's phase, 0..1
    stance: bool
    fwd: float         # body-frame displacement from the nominal foot point (mm)
    left: float
    lift: float        # mm above stand height
    x: float           # solver-frame target for solve_leg_3d
    y: float
    lateral: float


class GaitGenerator:
    """
    Stateful gait generator: call set_command() whenever the desired body motion
    changes, then step(dt) every tick to get the four foot targets, and solve()
    to turn them into joint angles.
    """

    def __init__(self, cfg: dict, params: GaitParams | None = None):
        self.cfg = cfg
        self.p = params or load_params(cfg)
        self.legs = list(cfg["legs"])
        self.phase = 0.0
        self.cmd = (0.0, 0.0, 0.0)     # requested (vx, vy, wz)
        self.vel = [0.0, 0.0, 0.0]     # ramped, speed-limited (vx, vy, wz) actually used
        self.limited = False           # True if the command was scaled down to fit max_step
        self.step_in_place = False
        self.activity = 0.0            # 0 = standing still, 1 = full step height

    # ── commands ──────────────────────────────────────────────────────
    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0,
                    step_in_place: bool | None = None) -> None:
        self.cmd = (float(vx), float(vy), float(wz))
        if step_in_place is not None:
            self.step_in_place = bool(step_in_place)

    def set_gait(self, name: str) -> None:
        if name not in GAITS:
            raise ValueError(f"unknown gait {name!r}, pick from {list(GAITS)}")
        self.p.gait = name

    def effective_command(self) -> tuple[float, float, float]:
        """The (vx, vy, wz) the gait actually follows: strafe modes turn vx sideways."""
        vx, vy, wz = self.cmd
        strafe = GAITS[self.p.gait].get("strafe")
        if strafe:
            return 0.0, strafe * vx, wz
        return vx, vy, wz

    def offsets(self) -> dict:
        return {leg: GAITS[self.p.gait]["offsets"].get(leg, 0.0) for leg in self.legs}

    def _leg_velocity(self, leg: str, vel) -> tuple[float, float]:
        """Foot velocity the body needs at this leg: v + w x r (body frame)."""
        vx, vy, wz = vel
        rx, ry = nominal_foot(self.cfg, leg, self.p)
        return vx - wz * ry, vy + wz * rx

    def _limit(self, vel) -> tuple[tuple, bool]:
        """Scale (vx, vy, wz) uniformly so no leg's stance stroke exceeds max_step_mm."""
        t = self.p.stance_time_s
        worst = max(math.hypot(*self._leg_velocity(leg, vel)) * t for leg in self.legs)
        if worst <= self.p.max_step_mm or worst == 0:
            return tuple(vel), False
        k = self.p.max_step_mm / worst
        return tuple(v * k for v in vel), True

    def max_speed_mm_s(self) -> float:
        """Top straight-line speed with the current parameters."""
        return self.p.max_step_mm / self.p.stance_time_s

    # ── time ──────────────────────────────────────────────────────────
    def step(self, dt: float) -> dict[str, FootTarget]:
        cmd = self.effective_command()
        target, self.limited = self._limit(cmd)
        lim = (self.p.max_accel_mm_s2 * dt, self.p.max_accel_mm_s2 * dt,
               self.p.max_yaw_accel_rad_s2 * dt)
        for i in range(3):
            d = target[i] - self.vel[i]
            self.vel[i] += max(-lim[i], min(lim[i], d))
        moving = any(abs(v) > 1e-6 for v in cmd) or any(abs(v) > 1e-3 for v in self.vel)
        want = 1.0 if (moving or self.step_in_place) else 0.0
        rate = dt / max(self.p.cycle_time_s, 1e-3)          # ramp over one cycle
        self.activity += max(-rate, min(rate, want - self.activity))
        if self.activity > 0:
            self.phase = (self.phase + dt / self.p.cycle_time_s) % 1.0
        return self.targets()

    # ── trajectory ────────────────────────────────────────────────────
    def foot_target(self, leg: str, phase: float | None = None, vel=None) -> FootTarget:
        """Foot target for one leg at a global phase (default: current state)."""
        p = self.p
        g = self.phase if phase is None else phase
        vel = self.vel if vel is None else vel
        lp = (g + GAITS[p.gait]["offsets"].get(leg, 0.0)) % 1.0
        ux, uy = self._leg_velocity(leg, vel)
        # Stance stroke D: the foot moves at -u relative to the body for the whole
        # stance, so it stays still on the ground. Position = D * k, with
        # k = -0.5 at touchdown (foot ahead, since D < 0 walking forward) and
        # k = +0.5 at lift-off (foot behind). Swing runs k back from +0.5 to -0.5.
        dx, dy = -ux * p.stance_time_s, -uy * p.stance_time_s
        duty = p.duty
        if p.gait == "jump":
            k, lift, stance = self._jump_profile(lp)
        elif lp < duty:
            k, lift, stance = lp / duty - 0.5, 0.0, True
        else:
            h, v = _cycloid((lp - duty) / (1.0 - duty))
            k, lift, stance = 0.5 - h, p.step_height_mm * self.activity * v, False
        fwd, left = k * dx, k * dy
        s_side = side_sign(self.cfg, leg)
        return FootTarget(
            leg=leg, phase=lp, stance=stance, fwd=fwd, left=left, lift=lift,
            x=p.neutral_x_mm + x_forward_sign(self.cfg, leg) * fwd,
            y=-p.stand_height_mm + lift,
            lateral=p.stance_width_mm + s_side * left,
        )

    # ── jump ──────────────────────────────────────────────────────────
    def _jump_profile(self, lp: float) -> tuple[float, float, bool]:
        """
        (k, lift, on_ground) for the jump gait at leg phase lp. lift is mm ABOVE
        stand height (negative = foot pushed below it). Horizontal k works like the
        other gaits, so a forward command makes it hop forward.

          ground, crouch : stand -> crouch (eased)
          ground, push   : crouch -> extend at constant (= top) speed -> takeoff
          air            : extend -> stand (eased) with a step_height tuck mid-air
        """
        p, a = self.p, self.activity
        duty, cf = p.duty, min(max(p.jump_crouch_frac, 0.05), 0.95)
        crouch, extend = p.jump_crouch_mm * a, -p.jump_extend_mm * a
        if lp < duty:
            s = lp / duty
            if s < cf:
                lift = crouch * _smooth(s / cf)
            else:
                lift = crouch + (extend - crouch) * (s - cf) / (1.0 - cf)
            return s - 0.5, lift, True
        s = (lp - duty) / (1.0 - duty)
        h, v = _cycloid(s)
        lift = extend * (1.0 - _smooth(s)) + p.step_height_mm * a * v
        return 0.5 - h, lift, False

    def jump_info(self) -> dict:
        """
        Takeoff physics for the jump gait, assuming the body leaves the ground
        with the feet's push speed (rigid legs, no servo lag):
          takeoff_mm_s : push speed at takeoff
          apex_mm      : how high the body rises above its takeoff height
          air_s        : time until the feet would touch the ground again, if the
                         legs reach the landing pose (lift 0) at once
          planned_air_s: the time the gait spends in the air
        """
        p, a = self.p, self.activity
        push_t = p.duty * p.cycle_time_s * (1.0 - min(max(p.jump_crouch_frac, 0.05), 0.95))
        v0 = (p.jump_crouch_mm + p.jump_extend_mm) * a / max(push_t, 1e-6)
        drop = p.jump_extend_mm * a          # body lands this much lower than it took off
        air = (v0 + math.sqrt(v0 * v0 + 2 * G_MM_S2 * drop)) / G_MM_S2
        return {"takeoff_mm_s": v0, "apex_mm": v0 * v0 / (2 * G_MM_S2),
                "air_s": air, "planned_air_s": (1.0 - p.duty) * p.cycle_time_s}

    def body_rise(self, phase: float | None = None) -> float:
        """
        Jump gait: mm the body is above its TAKEOFF height while airborne
        (ballistic; can go negative as it falls). 0 on the ground / other gaits.
        """
        p = self.p
        lp = self.phase if phase is None else phase
        if p.gait != "jump" or lp < p.duty:
            return 0.0
        t = (lp - p.duty) * p.cycle_time_s
        v0 = self.jump_info()["takeoff_mm_s"]
        return v0 * t - 0.5 * G_MM_S2 * t * t

    def targets(self, phase: float | None = None, vel=None) -> dict[str, FootTarget]:
        return {leg: self.foot_target(leg, phase, vel) for leg in self.legs}

    def cycle_path(self, leg: str, n: int = 60) -> list[FootTarget]:
        """One full cycle of foot targets for a leg at the current velocity (no IK)."""
        return [self.foot_target(leg, i / n) for i in range(n + 1)]

    # ── IK ────────────────────────────────────────────────────────────
    def solve(self, targets: dict[str, FootTarget]) -> dict[str, dict]:
        """IK for every target -> {leg: solve_leg_3d result}."""
        return {leg: solve_leg_3d(self.cfg, leg, t.x, t.y, t.lateral)
                for leg, t in targets.items()}


def stance_intervals(gen: GaitGenerator, leg: str) -> list[tuple[float, float]]:
    """Global-phase intervals (start, width) where this leg is on the ground."""
    off = gen.offsets()[leg]
    start = (-off) % 1.0
    end = start + gen.p.duty
    if end <= 1.0:
        return [(start, gen.p.duty)]
    return [(start, 1.0 - start), (0.0, end - 1.0)]


def support_margin(points: list[tuple[float, float]], com=(0.0, 0.0)) -> float:
    """
    Static stability margin (mm) of `com` over the support polygon of the given
    ground-contact points: distance to the nearest polygon edge, positive if the
    COM is inside, negative if outside. With fewer than 3 contacts the robot is
    never statically stable; returns -distance to the support line / point.
    """
    if not points:
        return -math.inf
    if len(points) == 1:
        return -math.dist(points[0], com)
    hull = _convex_hull(points)
    if len(hull) < 3:
        (ax, ay), (bx, by) = hull[0], hull[-1]
        return -_seg_dist(com, (ax, ay), (bx, by))
    inside = all(_cross(hull[i], hull[(i + 1) % len(hull)], com) >= 0 for i in range(len(hull)))
    d = min(_seg_dist(com, hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull)))
    return d if inside else -d


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(pts):
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and _cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and _cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]            # counter-clockwise


def _seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


# ── run on the robot ──────────────────────────────────────────────────────
def main():
    import yaml
    from arduino_link import ArduinoLink, add_cli_args

    ap = argparse.ArgumentParser(description="Stream a parametric gait to the robot")
    ap.add_argument("--gait", choices=list(GAITS))
    ap.add_argument("--vx", type=float, default=0.0, help="forward speed, mm/s")
    ap.add_argument("--vy", type=float, default=0.0, help="sideways speed (+left), mm/s")
    ap.add_argument("--wz", type=float, default=0.0, help="turn rate (+CCW/left), deg/s")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--in-place", action="store_true", help="march in place")
    ap.add_argument("--dry-run", action="store_true", help="print targets, don't send")
    add_cli_args(ap)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("linkage_config.yaml"))
    gen = GaitGenerator(cfg)
    if args.gait:
        gen.set_gait(args.gait)
    gen.set_command(args.vx, args.vy, math.radians(args.wz),
                    step_in_place=args.in_place or gen.p.gait == "jump")  # jump runs in place too
    print(f"gait {gen.p.gait}  duty {gen.p.duty:.2f}  cycle {gen.p.cycle_time_s}s  "
          f"max speed {gen.max_speed_mm_s():.0f} mm/s")

    link = None
    if not args.dry_run:
        link = ArduinoLink(cfg)
        if not link.connect_from_args(args):
            print("No Arduino - use --dry-run to just print.")
            return

    dt = 1.0 / gen.p.update_rate_hz
    t_end = time.monotonic() + args.seconds
    stop_at = None
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if now >= t_end and stop_at is None:
                gen.set_command(0, 0, 0, step_in_place=False)   # ramp down, then park
                stop_at = now + gen.p.cycle_time_s * 1.5
            if stop_at is not None and now >= stop_at:
                break
            tg = gen.step(now - last)
            last = now
            res = gen.solve(tg)
            bad = [leg for leg, r in res.items() if not r["valid"]]
            if args.dry_run:
                print(f"phase {gen.phase:.2f} " + "  ".join(
                    f"{leg}:{'S' if t.stance else 's'}({t.x:6.1f},{t.y:6.1f},{t.lateral:5.1f})"
                    for leg, t in tg.items()) + (f"  UNREACHABLE {bad}" if bad else ""))
            elif bad:
                print(f"unreachable {bad} - frame skipped")
            else:
                link.mgr.send_legs(cfg, {leg: (r["theta1"], r["theta_c"], r["hip"])
                                         for leg, r in res.items()})
            time.sleep(max(0.0, dt - (time.monotonic() - now)))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if link:
            link.close()


if __name__ == "__main__":
    main()
