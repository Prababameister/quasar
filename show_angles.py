import yaml
from leg_kinematics import solve_leg, solve_leg_3d

CFG = yaml.safe_load(open("linkage_config.yaml"))

print("2D (walking) angles, hip fixed at 90:")
for x, y in CFG["gait"]["waypoints"]:
    r = solve_leg(CFG, "FL", x, y)
    print(f"  foot ({x:7.1f}, {y:7.1f})  theta1 {r['theta1']:7.2f}  theta_c {r['theta_c']:7.2f}  "
          f"hip {r['hip']:.1f}  error {r['error_mm']:.2f} mm")

print("\n3D angles, foot 20 mm outward:")
for leg in ["FL", "FR"]:
    r = solve_leg_3d(CFG, leg, -5.0, -150.0, lateral=20.0)
    print(f"  {leg}  theta1 {r['theta1']:7.2f}  theta_c {r['theta_c']:7.2f}  hip {r['hip']:.1f}")