"""
one_leg.py - smallest verifiable IK check: ONE leg, one pose at a time.

Run from the quasar-1 folder:   python one_leg.py
Change LEG to "FR" (then BL, BR) once FL looks right.
"""

import yaml
import matplotlib.pyplot as plt

from leg_kinematics import solve_leg, fk_leg
from serial_manager import SerialManager

LEG = "FL"       # FL, FR, BL or BR
PLOT = True

CFG = yaml.safe_load(open("linkage_config.yaml"))
leg_cfg = CFG["legs"][LEG]


def leg_us(r):
    """Exactly the pulses send_leg() would use (offsets + calibration frame + calibration)."""
    return SerialManager().compute_leg_us(CFG, LEG, r["theta1"], r["theta_c"], r["hip"])


print(f"Leg {LEG} ({leg_cfg['side']}), channels {leg_cfg['channels']}\n")
print("STEP 1 - one pose (first gait waypoint):")
x, y = CFG["gait"]["waypoints"][0]
r = solve_leg(CFG, LEG, x, y)
d = fk_leg(CFG, LEG, r["theta1"], r["theta_c"])["points"]["D"]
print(f"  target   ({x:.2f}, {y:.2f})")
print(f"  IK       theta1 {r['theta1']:.2f}  theta_c {r['theta_c']:.2f}  hip {r['hip']:.1f}")
print(f"  FK foot  ({d['x']:.2f}, {d['y']:.2f})   error {r['error_mm']:.2f} mm   valid={r['valid']}")
us = leg_us(r)
print(f"  pulses   theta1 {us['theta1']} us   theta_c {us['theta_c']} us   hip {us['hip']} us")

print("\nSTEP 2 - whole gait loop on this leg:")
for x, y in CFG["gait"]["waypoints"]:
    r = solve_leg(CFG, LEG, x, y)
    print(f"  ({x:7.1f},{y:7.1f}) -> theta1 {r['theta1']:7.2f}  theta_c {r['theta_c']:7.2f}  "
          f"err {r['error_mm']:.2f} mm")

if PLOT:
    r = solve_leg(CFG, LEG, *CFG["gait"]["waypoints"][0])
    p = fk_leg(CFG, LEG, r["theta1"], r["theta_c"])["points"]
    for a, b in [("O", "A"), ("A", "B"), ("Q", "B"), ("Q", "C"), ("Q", "P1"),
                 ("C", "P2"), ("P1", "P2"), ("P1", "D")]:
        plt.plot([p[a]["x"], p[b]["x"]], [p[a]["y"], p[b]["y"]], "b-")
    plt.scatter(p["D"]["x"], p["D"]["y"], c="r")
    plt.axis("equal"); plt.title(f"{LEG}: linkage at first waypoint")
    plt.show()