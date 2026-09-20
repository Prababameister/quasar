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
