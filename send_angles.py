"""
send_angles.py - send raw joint angles to ONE leg from the command line.

    uv run send_angles.py FL 150 -60 90            # leg theta1 theta_c hip (deg)
    uv run send_angles.py BR 140 -50 90 --port COM4

Angles are the same IK-facing joint angles the solvers use; per-leg offsets
and calibration from linkage_config.yaml are applied before sending.
NOTE: opening the port resets the UNO, so the other legs go back to the
firmware's boot pose (chan[] in cobs_motor_prog.ino) while this leg moves.
For live adjusting without resets, use fk_interactive.py instead.
"""

import argparse
import time

import yaml

from arduino_link import ArduinoLink, add_cli_args
from leg_kinematics import fk_leg, leg_names

CFG = yaml.safe_load(open("linkage_config.yaml"))
LIMITS = {i["name"]: (i.get("min_deg", -999), i.get("max_deg", 999)) for i in CFG["inputs"]}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("leg", choices=leg_names(CFG))
ap.add_argument("theta1", type=float)
ap.add_argument("theta_c", type=float)
ap.add_argument("hip", type=float)
ap.add_argument("--force", action="store_true",
                help="send even if outside limits / linkage can't assemble")
add_cli_args(ap)
args = ap.parse_args()

problems = []
for name in ("theta1", "theta_c"):
    lo, hi = LIMITS[name]
    if not lo <= getattr(args, name) <= hi:
        problems.append(f"{name} {getattr(args, name)} outside limits {lo}..{hi}")
try:
    d = fk_leg(CFG, args.leg, args.theta1, args.theta_c)["points"]["D"]
    print(f"FK foot (leg plane): x {d['x']:.1f}  y {d['y']:.1f}")
except Exception as e:
    problems.append(f"linkage can't assemble: {e}")

us = ArduinoLink(CFG).mgr.compute_leg_us(CFG, args.leg, args.theta1, args.theta_c, args.hip)
print(f"pulses (us): theta1 {us['theta1']}  theta_c {us['theta_c']}  hip {us['hip']}")

if problems and not args.force:
    print("NOT SENT:\n  " + "\n  ".join(problems) + "\n(use --force to send anyway)")
    raise SystemExit(1)

link = ArduinoLink(CFG)
if not link.connect_from_args(args):
    raise SystemExit("No Arduino connection - nothing sent.")
print(link.send(args.leg, args.theta1, args.theta_c, args.hip))
time.sleep(0.1)   # let the frame drain before closing the port
link.close()
