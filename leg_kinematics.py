"""
leg_kinematics.py — per-leg wrapper around the single-leg FK/IK solvers.

The linkage geometry in linkage_config.yaml describes one canonical LEFT leg.
The four quadruped legs are:

    FL, BL : side "left"   — use the canonical geometry directly
    FR, BR : side "right"  — the leg is the canonical linkage mirrored across
                             the config y-axis

Mirroring rule
--------------
Foot targets and joint angles are ALWAYS expressed in the canonical (left) frame,
for every leg. `ik_solve` runs identically for all four legs, so a given foot
target yields the same theta1 / theta_c on the left and right.

The mirror is handled in exactly two places, neither of which is the solver:

  * physically — a right leg is built mirror-image and its servos mounted
    reversed, so feeding them the canonical angle produces the mirrored motion;
    residual direction/zero error is taken up by that leg's `offsets` in the
    config.
  * visually — the renderer flips the x-axis for right-side legs (see
    fk_leg()'s `side` field) so the drawing shows a real right leg, and a click
    in a right quadrant is un-flipped back to a canonical target.
"""

import math

from linkage_solver import fk_solve
from ik_solver import ik_solve


def leg_names(cfg: dict) -> list[str]:
    return list(cfg.get("legs", {}).keys())


def leg_side(cfg: dict, leg: str) -> str:
    try:
        return cfg["legs"][leg]["side"]
    except KeyError:
        raise ValueError(f"Unknown leg '{leg}'")


def hip_fixed_deg(cfg: dict) -> float:
    return float(cfg.get("hip_fixed_deg", 90.0))


def nominal_target(cfg: dict) -> tuple[float, float]:
    """Parked foot position (canonical left-leg frame): the first gait waypoint."""
    g = cfg.get("gait", {})
    wp = g.get("waypoints") or []
    if wp:
        return float(wp[0][0]), float(wp[0][1])
    return 150.0, -120.0


def solve_leg(cfg: dict, leg: str, x: float, y: float) -> dict:
    """
    Inverse kinematics for one named leg.

    Returns the ik_solve() dict (valid, theta1, theta_c, d_actual, error_mm,
    iterations) plus:
        leg  : the leg name
        hip  : the fixed hip angle in degrees
        side : "left" | "right"

    theta1 / theta_c are in the canonical frame for every leg (see module docstring).
    """
    side = leg_side(cfg, leg)
    res = ik_solve(cfg, x, y)
    res["leg"] = leg
    res["side"] = side
    res["hip"] = hip_fixed_deg(cfg)
    return res


def fk_leg(cfg: dict, leg: str, theta1: float, theta_c: float) -> dict:
    """
    Forward kinematics for one named leg.

    Returns {"side": "left"|"right", "points": {name: {"x", "y"}}}.

    Points are in the canonical frame for every leg. The renderer flips the
    x-axis for right-side legs when drawing.
    """
    side = leg_side(cfg, leg)
    pts = fk_solve(cfg, {"theta1": theta1, "theta_c": theta_c})
    return {
        "side": side,
        "points": {name: {"x": round(px, 4), "y": round(py, 4)}
                   for name, (px, py) in pts.items()},
    }

def calibration_frame(cfg: dict) -> str:
    """Which angle frame legs.<leg>.calibration is written in: "solver" or "hardware"."""
    return str(cfg.get("calibration_frame", "solver"))


def horn_dir(cfg: dict | None, leg: str | None) -> float:
    """
    legs.<leg>.horn_dir: which way the horn's HARDWARE angle counts (default +1).
      +1 : 0 = level on the solver's -x, 90 = pointing UP   (-x -> +y)
      -1 : 0 = level on the solver's -x, 90 = pointing DOWN (-x -> -y)
    """
    if cfg is None or leg is None:
        return 1.0
    v = float(cfg["legs"][leg].get("horn_dir", 1.0))
    if v not in (1.0, -1.0):
        raise ValueError(f"legs.{leg}.horn_dir must be 1 or -1, got {v}")
    return v


def solver_to_hw(theta1: float, theta_c: float,
                 cfg: dict | None = None, leg: str | None = None) -> tuple[float, float]:
    """
    Solver-frame joint angles -> the HARDWARE frame the servos are calibrated in.

    Solver frame: standard math angles in the leg plane (+x forward, +y up).
      theta1  = horn servo  (O -> A)
      theta_c = thigh servo (Q -> P1)
    Hardware frame (the 'Quadruped V1 Documentation' convention):
      horn  (config key theta1):  0 = level, pointing back (solver -x);
                                  90 = pointing up, or DOWN if the leg's
                                  horn_dir is -1 (pass cfg + leg for that)
      thigh (config key theta_c): 0 = level, pointing forward; 90 = straight down
    Hip is the same in both frames (90 = leg hanging straight down).
    """
    return _wrap180(horn_dir(cfg, leg) * (180.0 - theta1)), _wrap180(-theta_c)


def hw_to_solver(horn_hw: float, thigh_hw: float,
                 cfg: dict | None = None, leg: str | None = None) -> tuple[float, float]:
    """Inverse of solver_to_hw, returned inside the solver's input limits (theta1 in 90..360)."""
    t1 = (180.0 - horn_dir(cfg, leg) * horn_hw) % 360.0
    return (t1 + 360.0 if t1 < 90.0 else t1), -thigh_hw


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def x_forward_sign(cfg: dict, leg: str) -> float:
    """+1 if the solver's +x is the robot's FRONT for this leg, -1 if it is the BACK."""
    v = float(cfg["legs"][leg].get("x_forward_sign", cfg.get("x_forward_sign", 1.0)))
    if v not in (1.0, -1.0):
        raise ValueError(f"legs.{leg}.x_forward_sign must be 1 or -1, got {v}")
    return v


def hip_dir(cfg: dict, leg: str) -> float:
    """+1/-1: which way the hip servo angle moves for an OUTWARD foot shift."""
    return float(cfg["legs"][leg].get("hip_dir", 1.0))


def hip_offset_mm(cfg: dict) -> float:
    """Sideways distance from the hip servo's rotation axis to the leg plane."""
    return float(cfg.get("hip_offset_mm", 0.0))


def leg_plane_to_3d(cfg: dict, leg: str, px: float, py: float, hip_deg: float):
    """
    A point (px, py) in the leg's own plane -> leg-local 3D (X fwd, Y, Z up),
    hip axis at the origin. Y is +left, so right legs go toward -Y.

    The hip offset is a rigid bar fixed at a CONSTANT 90 deg to the leg plane:
    it runs from the hip axis (along X) to the leg-plane origin. Swinging the
    hip rotates the bar and the leg together about X, so the bar always stays
    perpendicular to the leg. A positive swing moves the foot outward.
    """
    s = 1.0 if leg_side(cfg, leg) == "left" else -1.0
    phi = math.radians((hip_deg - hip_fixed_deg(cfg)) / hip_dir(cfg, leg))
    d = hip_offset_mm(cfg)
    # bar direction (cos, sin) and leg-down direction (sin, -cos) are perpendicular
    out = d * math.cos(phi) - py * math.sin(phi)
    z = d * math.sin(phi) + py * math.cos(phi)
    return (px, s * out, z)


def solve_leg_3d(cfg: dict, leg: str, x: float, y: float, lateral: float = 0.0) -> dict:
    """
    x, y    : canonical sagittal target (same as solve_leg; y negative = below hip)
    lateral : mm the foot sits OUTWARD of the leg's neutral (hip = hip_fixed_deg)
              vertical plane (>0 = away from body). With a hip offset d, the
              foot is therefore d + lateral outward of the hip axis.
    Returns the solve_leg dict, but with "hip" computed instead of fixed, plus
    "swing_deg" (unclamped swing) and "reach" (in-plane depth that was solved).
    """
    if lateral == 0.0:
        res = solve_leg(cfg, leg, x, y)  # identical to current behaviour
        res["swing_deg"] = 0.0
        res["reach"] = -y
        return res

    # The offset bar stays at 90 deg to the leg, so in the hip's cross-section
    # the foot is the corner of a right triangle: legs d (bar) and reach (leg).
    d = hip_offset_mm(cfg)
    out = d + lateral                     # foot distance outward of the hip axis
    depth = -y
    reach = math.sqrt(max(out * out + depth * depth - d * d, 0.0))
    # foot angle about the hip axis, minus the angle the bar itself takes up
    phi = math.atan2(out, depth) - math.atan2(d, reach)

    res = solve_leg(cfg, leg, x, -reach)  # planar IK, unchanged
    limit = float(cfg.get("hip_swing_max_deg", 30.0))
    swing = max(-limit, min(limit, math.degrees(phi)))
    res["hip"] = hip_fixed_deg(cfg) + hip_dir(cfg, leg) * swing
    res["lateral"] = lateral
    res["swing_deg"] = math.degrees(phi)
    res["reach"] = reach
    return res


def lateral_for_swing(cfg: dict, y: float, swing_deg: float) -> float:
    """Inverse of solve_leg_3d's hip: the lateral that gives this swing at height y."""
    d = hip_offset_mm(cfg)
    phi = math.radians(swing_deg)
    reach = (d * math.sin(phi) - y) / math.cos(phi)
    return d * math.cos(phi) + reach * math.sin(phi) - d
