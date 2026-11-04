import random, yaml
from workspace import compute_workspace
from leg_kinematics import solve_leg

CFG = yaml.safe_load(open("linkage_config.yaml"))
ws = compute_workspace(CFG)
random.seed(0)
for p in random.sample(ws, 80):
    r = solve_leg(CFG, "FL", p["x"], p["y"])
    if r["error_mm"] > 1.5:
        print(f"target ({p['x']:.0f}, {p['y']:.0f})  err {r['error_mm']:.1f}  "
              f"theta1 {r['theta1']:.0f}  theta_c {r['theta_c']:.0f}")