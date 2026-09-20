"""
ik_solver.py — Gradient-descent inverse kinematics.

Minimises  loss = |D_fk(theta1, theta_c) - D_target|² + limit_penalty
by iterating on (theta1, theta_c) with numerical gradients.

Tuning knobs (all overridable via ik_config block in YAML):
    tolerance_mm   : stop when |error| < this              (default 1.0)
    max_iterations : hard cap on gradient steps per start  (default 2000)
    learning_rate  : initial step size in degrees          (default 1.0)
    finite_diff_h  : perturbation for numerical gradient   (default 0.01 deg)
    multi_start    : number of starting points (grid)      (default 9)
    limit_margin   : degrees from limit where penalty kicks in (default 10.0)
    limit_weight   : penalty scale factor                  (default 500.0)
"""

import math
from linkage_solver import fk_solve, _find_edge_length


# ── Defaults ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "tolerance_mm":   1.0,
    "max_iterations": 2000,
    "learning_rate":  1.0,
    "finite_diff_h":  0.01,
    "multi_start":    9,
    "limit_margin":   10.0,
    "limit_weight":   500.0,
}


def _get_ik_cfg(cfg: dict) -> dict:
    return {**_DEFAULTS, **cfg.get("ik_config", {})}


_INF = float("inf")


def _angle_limits(cfg: dict) -> dict:
    """Return {input_name: (min_deg, max_deg)} hard limits; unbounded if not set."""
    return {inp["name"]: (inp.get("min_deg", -_INF), inp.get("max_deg", _INF))
            for inp in cfg.get("inputs", [])}


def _seed_ranges(cfg: dict) -> dict:
    """
    Return {input_name: (lo, hi)} spanning the multi-start grid. These only pick
    where the search begins; they never constrain the result. Falls back to the
    hard limits if set, else to -180..180.
    """
    ranges = {}
    for inp in cfg.get("inputs", []):
        lo = inp.get("seed_min_deg", inp.get("min_deg", -180.0))
        hi = inp.get("seed_max_deg", inp.get("max_deg",  180.0))
        ranges[inp["name"]] = (lo, hi)
    return ranges


def _default_angle(cfg: dict, name: str) -> float:
    for inp in cfg.get("inputs", []):
        if inp["name"] == name:
            return float(inp["angle_deg"])
    return 0.0


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _limit_penalty(val, lo, hi, margin, weight):
    """
    Smooth quadratic penalty that rises in the last `margin` degrees
    before each joint limit. Keeps gradients informative near walls.
    """
    penalty = 0.0
    if lo != -_INF and val < lo + margin:
        penalty += ((lo + margin - val) / margin) ** 2
    if hi != _INF and val > hi - margin:
        penalty += ((val - hi + margin) / margin) ** 2
    return penalty * weight


def _d_from_angles(cfg, theta1, theta_c):
    """Run FK and return D position as (x, y)."""
    pts = fk_solve(cfg, {"theta1": theta1, "theta_c": theta_c})
    return pts["D"]


def _ik_single(cfg, target_x, target_y,
               t1_init, tc_init, ikcfg,
               t1_lo, t1_hi, tc_lo, tc_hi) -> dict:
    """
    Run one gradient descent attempt from a given starting point.
    Returns the same result dict as ik_solve.
    """
    tol    = float(ikcfg["tolerance_mm"])
    max_it = int(ikcfg["max_iterations"])
    lr     = float(ikcfg["learning_rate"])
    h      = float(ikcfg["finite_diff_h"])
    margin = float(ikcfg["limit_margin"])
    weight = float(ikcfg["limit_weight"])
    max_grad = 10.0

    t1 = _clamp(t1_init, t1_lo, t1_hi)
    tc = _clamp(tc_init, tc_lo, tc_hi)

    def loss(a1, ac):
        try:
            dx, dy = _d_from_angles(cfg, a1, ac)
            pos_loss = (dx - target_x) ** 2 + (dy - target_y) ** 2
            lim_loss = (
                _limit_penalty(a1, t1_lo, t1_hi, margin, weight) +
                _limit_penalty(ac, tc_lo, tc_hi, margin, weight)
            )
            return pos_loss + lim_loss
        except Exception:
            return 1e12

    prev_loss = loss(t1, tc)

    for i in range(max_it):
        # Check convergence on raw position error (not penalised loss)
        try:
            dx, dy = _d_from_angles(cfg, t1, tc)
            err = math.hypot(dx - target_x, dy - target_y)
        except Exception:
            err = float("inf")

        if err < tol:
            return {
                "valid":      True,
                "theta1":     round(t1, 4),
                "theta_c":    round(tc, 4),
                "d_actual":   {"x": round(dx, 4), "y": round(dy, 4)},
                "error_mm":   round(err, 4),
                "iterations": i,
            }

        # Numerical gradient
        g1 = (loss(t1 + h, tc) - prev_loss) / h
        gc = (loss(t1, tc + h) - prev_loss) / h

        # Gradient clipping — prevents runaway steps near limits
        grad_norm = math.hypot(g1, gc)
        if grad_norm > max_grad:
            g1 *= max_grad / grad_norm
            gc *= max_grad / grad_norm

        # Gradient step
        t1_new = _clamp(t1 - lr * g1, t1_lo, t1_hi)
        tc_new = _clamp(tc - lr * gc, tc_lo, tc_hi)

        new_loss = loss(t1_new, tc_new)

        if new_loss < prev_loss:
            t1, tc = t1_new, tc_new
            prev_loss = new_loss
            lr = min(lr * 1.05, 5.0)   # grow lr slightly on success
        else:
            lr *= 0.5                   # shrink on failure
            if lr < 1e-6:
                break

    # Did not converge — return best found
    try:
        dx, dy = _d_from_angles(cfg, t1, tc)
        err = math.hypot(dx - target_x, dy - target_y)
    except Exception:
        dx, dy, err = target_x, target_y, float("inf")

    return {
        "valid":      False,
        "theta1":     round(t1, 4),
        "theta_c":    round(tc, 4),
        "d_actual":   {"x": round(dx, 4), "y": round(dy, 4)},
        "error_mm":   round(err, 4),
        "iterations": max_it,
    }


def ik_solve(cfg: dict, target_x: float, target_y: float) -> dict:
    """
    Multi-start gradient-descent IK.

    Runs a grid of starting points across the joint angle space,
    keeps the best result. Stops early if a valid solution is found.

    Parameters
    ----------
    cfg      : parsed linkage config
    target_x : desired D.x in mm
    target_y : desired D.y in mm

    Returns
    -------
    {
        "valid":      bool,
        "theta1":     float (degrees),
        "theta_c":    float (degrees),
        "d_actual":   {"x": float, "y": float},
        "error_mm":   float,
        "iterations": int,
    }
    """
    ikcfg   = _get_ik_cfg(cfg)
    limits  = _angle_limits(cfg)
    t1_lo, t1_hi = limits.get("theta1",  (-_INF, _INF))
    tc_lo, tc_hi = limits.get("theta_c", (-_INF, _INF))
    seeds   = _seed_ranges(cfg)
    s1_lo, s1_hi = seeds.get("theta1",  (-180.0, 180.0))
    sc_lo, sc_hi = seeds.get("theta_c", (-180.0, 180.0))

    # Build grid of starting points
    n_starts = int(ikcfg["multi_start"])
    n        = max(1, round(n_starts ** 0.5))

    starts = [
        # Always try the config default first
        (_default_angle(cfg, "theta1"), _default_angle(cfg, "theta_c")),
    ]
    for i in range(n):
        for j in range(n):
            t1 = s1_lo + (s1_hi - s1_lo) * i / max(n - 1, 1)
            tc = sc_lo + (sc_hi - sc_lo) * j / max(n - 1, 1)
            candidate = (t1, tc)
            if candidate not in starts:
                starts.append(candidate)

    best = None
    for (t1_init, tc_init) in starts:
        result = _ik_single(
            cfg, target_x, target_y,
            t1_init, tc_init, ikcfg,
            t1_lo, t1_hi, tc_lo, tc_hi,
        )
        if best is None or result["error_mm"] < best["error_mm"]:
            best = result
        if best["valid"]:
            break   # found a good solution, no need to try more starts

    return best
