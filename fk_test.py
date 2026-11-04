"""
fk_test.py - STEP 1: forward kinematics on ONE leg.

Give the leg two motor angles, get the foot position, then check the linkage
is built correctly (every bar keeps its real length, B-Q-C stays at 90 deg).

Run from the quasar-1 folder:   python fk_test.py
"""

import math
import yaml
import matplotlib.pyplot as plt

from leg_kinematics import fk_leg

LEG = "FL"
THETA1 = 150.0      # degrees, allowed 90..360
THETA_C = -60.0     # degrees, allowed -90..180

CFG = yaml.safe_load(open("linkage_config.yaml"))

try:
    p = fk_leg(CFG, LEG, THETA1, THETA_C)["points"]
except ValueError as e:
    raise SystemExit(f"These angles can't assemble the linkage: {e}\n"
                     "Try different THETA1 / THETA_C values.")


def dist(a, b):
    return math.hypot(p[a]["x"] - p[b]["x"], p[a]["y"] - p[b]["y"])


print(f"Leg {LEG}   theta1 = {THETA1}   theta_c = {THETA_C}\n")
print("Point positions (mm):")
for name, pt in p.items():
    print(f"  {name:3s} ({pt['x']:8.2f}, {pt['y']:8.2f})")

print(f"\nFOOT (D) = ({p['D']['x']:.2f}, {p['D']['y']:.2f})\n")

print("Check 1 - every bar keeps its real length:")
all_ok = True
for e in CFG["edges"]:
    got, want = dist(e["from"], e["to"]), e["length"]
    ok = abs(got - want) < 0.01
    all_ok &= ok
    print(f"  {e['from']}-{e['to']:3s} want {want:8.3f}  got {got:8.3f}  {'OK' if ok else 'BAD'}")

bq = (p["B"]["x"] - p["Q"]["x"], p["B"]["y"] - p["Q"]["y"])
qc = (p["C"]["x"] - p["Q"]["x"], p["C"]["y"] - p["Q"]["y"])
ang = math.degrees(math.acos((bq[0]*qc[0] + bq[1]*qc[1]) / (math.hypot(*bq) * math.hypot(*qc))))
print(f"\nCheck 2 - angle B-Q-C should be 90: got {ang:.2f}  {'OK' if abs(ang - 90) < 0.01 else 'BAD'}")
print("\nRESULT:", "PASS" if all_ok and abs(ang - 90) < 0.01 else "FAIL")

for a, b in [("O", "A"), ("A", "B"), ("Q", "B"), ("Q", "C"), ("Q", "P1"),
             ("C", "P2"), ("P1", "P2"), ("P1", "D")]:
    plt.plot([p[a]["x"], p[b]["x"]], [p[a]["y"], p[b]["y"]], "b-")
plt.scatter(p["D"]["x"], p["D"]["y"], c="r", label="foot (D)")
for n, pt in p.items():
    plt.text(pt["x"], pt["y"], n)
plt.axis("equal"); plt.legend(); plt.title(f"{LEG}: theta1={THETA1}, theta_c={THETA_C}")
plt.show()