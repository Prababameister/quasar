"""
validate_ik.py - software validation of the leg IK, no hardware needed.

Run from the quasar-1 folder:   python validate_ik.py

Checks
  1. IK -> FK round trip over the reachable workspace (and gait/jump points)
  2. Gait loop continuity (no sudden joint-angle jumps = no branch flipping)
  3. Left/right mirror: servo pulse slope must flip sign between sides
  4. 3D lateral (hip) test: foot lands where requested
Plots
  - sagittal workspace + gait loop + linkage drawn at the gait waypoints
  - 3D view of all four legs standing at the nominal pose

ASSUMPTIONS: hip axis passes through linkage origin O; body mount offsets
below are made up (only for the picture). Change them to your robot.
"""

import math
import random

import numpy as np
import yaml
import matplotlib.pyplot as plt

from leg_kinematics import (solve_leg, solve_leg_3d, fk_leg, leg_names,
                            leg_side, nominal_target, leg_plane_to_3d,
                            hip_offset_mm)
from workspace import compute_workspace

CFG = yaml.safe_load(open("linkage_config.yaml"))
TOL_MM = 1.5
MOUNT = {"FL": (100, 60), "BL": (-100, 60), "BR": (-100, -60), "FR": (100, -60)}

_results = []


def check(name, ok, detail=""):
    _results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def roundtrip(x, y):
    res = solve_leg(CFG, "FL", x, y)
    d = fk_leg(CFG, "FL", res["theta1"], res["theta_c"])["points"]["D"]
    return res, math.hypot(d["x"] - x, d["y"] - y)


# 1. IK -> FK round trip ------------------------------------------------------
print("\n[1] IK -> FK round trip")
ws = compute_workspace(CFG)
print(f"  workspace has {len(ws)} reachable points")
random.seed(0)
sample = random.sample(ws, min(80, len(ws)))
worst, fails = 0.0, 0
for p in sample:
    res, err = roundtrip(p["x"], p["y"])
    worst = max(worst, err)
    fails += err > TOL_MM
check("random reachable targets", fails == 0,
      f"({fails}/{len(sample)} failed, worst {worst:.2f} mm)")

wps = CFG["gait"]["waypoints"]
worst = max(roundtrip(x, y)[1] for x, y in wps)
check("gait waypoints", worst < TOL_MM, f"(worst {worst:.2f} mm)")

jp = CFG.get("jump", {})
for k in ("neutral", "crouch", "extend"):
    if k in jp:
        _, err = roundtrip(*jp[k])
        check(f"jump point '{k}'", err < TOL_MM, f"({err:.2f} mm)")

# 2. Gait continuity ----------------------------------------------------------
print("\n[2] Gait loop continuity")
loop = []
n = len(wps)
for i in range(n):
    a, b = np.array(wps[i]), np.array(wps[(i + 1) % n])
    for t in np.linspace(0, 1, 25, endpoint=False):
        loop.append(a + (b - a) * t)
angs = np.array([[r["theta1"], r["theta_c"]]
                 for r in (solve_leg(CFG, "FL", x, y) for x, y in loop)])
jumps = np.abs(np.diff(np.vstack([angs, angs[:1]]), axis=0)).max(axis=0)
check("no angle jumps along loop", (jumps < 10).all(),
      f"(max step theta1 {jumps[0]:.1f} deg, theta_c {jumps[1]:.1f} deg)")

# 3. Left/right mirror via servo calibration ---------------------------------
print("\n[3] Mirror check: pulse-vs-angle slope sign per servo")
for servo in ("theta1", "theta_c", "hip"):
    slopes = {}
    for leg in leg_names(CFG):
        c = CFG["legs"][leg]["calibration"][servo]
        slopes[leg] = (c["us1"] - c["us0"]) / (c["deg1"] - c["deg0"])
    left = {math.copysign(1, s) for l, s in slopes.items() if leg_side(CFG, l) == "left"}
    right = {math.copysign(1, s) for l, s in slopes.items() if leg_side(CFG, l) == "right"}
    ok = len(left) == 1 and len(right) == 1 and left != right
    check(f"{servo}: sides opposite", ok,
          "  ".join(f"{l}:{s:+.2f}us/deg" for l, s in slopes.items()))

# 4. 3D / lateral hip test ----------------------------------------------------
def to_3d(leg, pts, hip_deg):
    """canonical (x,y) points -> body frame (X fwd, Y left, Z up)."""
    mx, my = MOUNT[leg]
    return {k: np.array(leg_plane_to_3d(CFG, leg, p["x"], p["y"], hip_deg)) + [mx, my, 0.0]
            for k, p in pts.items()}


print("\n[4] 3D lateral test (foot 20 mm outward)")
nx, ny = nominal_target(CFG)
for leg in leg_names(CFG):
    r = solve_leg_3d(CFG, leg, nx, ny, lateral=20.0)
    pts = fk_leg(CFG, leg, r["theta1"], r["theta_c"])["points"]
    foot = to_3d(leg, pts, r["hip"])["D"]
    mx, my = MOUNT[leg]
    s = 1.0 if leg_side(CFG, leg) == "left" else -1.0
    want = np.array([mx + nx, my + s * (hip_offset_mm(CFG) + 20.0), ny])
    err = np.linalg.norm(foot - want)
    check(f"{leg} foot at requested spot", err < 2.5,
          f"(err {err:.2f} mm, hip {r['hip']:.1f} deg)")

# Plots ------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 6))
ax2 = fig.add_subplot(1, 2, 1)
ax2.scatter([p["x"] for p in ws], [p["y"] for p in ws], s=1, c="lightgray", label="workspace")
lx, ly = zip(*(list(loop) + [loop[0]]))
ax2.plot(lx, ly, "r-", label="gait loop")
for x, y in wps:
    r = solve_leg(CFG, "FL", x, y)
    pts = fk_leg(CFG, "FL", r["theta1"], r["theta_c"])["points"]
    for a, b in [("O", "A"), ("A", "B"), ("Q", "B"), ("Q", "C"), ("Q", "P1"),
                 ("C", "P2"), ("P1", "P2"), ("P1", "D")]:
        ax2.plot([pts[a]["x"], pts[b]["x"]], [pts[a]["y"], pts[b]["y"]], "b-", lw=0.6, alpha=0.5)
ax2.set_aspect("equal"); ax2.legend(); ax2.set_title("Sagittal plane (canonical left leg)")

ax3 = fig.add_subplot(1, 2, 2, projection="3d")
for leg in leg_names(CFG):
    r = solve_leg_3d(CFG, leg, nx, ny, lateral=0.0)
    pts = fk_leg(CFG, leg, r["theta1"], r["theta_c"])["points"]
    p3 = to_3d(leg, pts, r["hip"])
    for a, b in [("O", "A"), ("A", "B"), ("Q", "B"), ("Q", "C"), ("Q", "P1"),
                 ("C", "P2"), ("P1", "P2"), ("P1", "D")]:
        ax3.plot(*zip(p3[a], p3[b]), "b-")
    ax3.scatter(*p3["D"], c="r")
    ax3.text(*p3["O"], leg)
ax3.set_xlabel("X fwd"); ax3.set_ylabel("Y left"); ax3.set_zlabel("Z up")
ax3.set_title("All four legs, nominal pose")
plt.tight_layout()

print(f"\n{sum(_results)}/{len(_results)} checks passed")
plt.show()