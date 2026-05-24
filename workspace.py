"""
workspace.py — Precomputes the reachable workspace of the end effector (D).

Sweeps theta1 and theta_c across their allowed ranges in configurable steps,
runs FK for every combination, and collects valid D positions.

Config knobs (under workspace_config in YAML):
    step_deg : angular step size for the sweep (default 2.0 degrees)
"""

import math
import logging
from linkage_solver import fk_solve

log = logging.getLogger("workspace")

_DEFAULT_STEP = 2.0


def _angle_range(inp: dict):
    lo   = float(inp.get("min_deg",   -180.0))
    hi   = float(inp.get("max_deg",    180.0))
    return lo, hi


def compute_workspace(cfg: dict) -> list[dict]:
    """
    Returns a list of {"x": float, "y": float} dicts representing all
    reachable D positions given the joint angle limits in the config.
    """
    ws_cfg  = cfg.get("workspace_config", {})
    step    = float(ws_cfg.get("step_deg", _DEFAULT_STEP))

    # Find theta1 and theta_c input configs
    inputs  = {inp["name"]: inp for inp in cfg.get("inputs", [])}
    inp1    = inputs.get("theta1",  {})
    inpc    = inputs.get("theta_c", {})

    t1_lo, t1_hi = _angle_range(inp1)
    tc_lo, tc_hi = _angle_range(inpc)

    points = []
    steps1 = max(1, round((t1_hi - t1_lo) / step))
    stepsc = max(1, round((tc_hi - tc_lo) / step))

    for i in range(steps1 + 1):
        t1 = t1_lo + i * (t1_hi - t1_lo) / steps1
        for j in range(stepsc + 1):
            tc = tc_lo + j * (tc_hi - tc_lo) / stepsc
            try:
                pts = fk_solve(cfg, {"theta1": t1, "theta_c": tc})
                d   = pts["D"]
                points.append({"x": round(d[0], 3), "y": round(d[1], 3)})
            except Exception:
                pass   # degenerate configuration — skip

    log.info(f"Workspace sweep: {len(points)} valid points "
             f"({steps1+1}×{stepsc+1} grid, step={step}°)")
    return points
