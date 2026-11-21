"""
sweep_bl_offset.py - stand the robot, then step ONE leg's horn (theta1) offset
through a range so you can watch for the value where its calf is level.

    uv run sweep_bl_offset.py                         # BL, 0..30 deg, 3 deg steps, 3 s each
    uv run sweep_bl_offset.py --start 10 --stop 40 --step 2 --hold 4
    uv run sweep_bl_offset.py --leg BR --joint theta_c

The offset is applied in memory only; linkage_config.yaml is NOT changed. Note
the value that looks level and put it in legs.<leg>.offsets yourself (or ask).
Moves between poses are ramped, never jumped. Ctrl+C stops (servos hold).
"""

import argparse
import time

import yaml

from arduino_link import ArduinoLink, add_cli_args
from gait_generator import GaitGenerator

CONFIG_PATH = "linkage_config.yaml"
RATE_HZ = 30
OFFSET_KEY = {"theta1": "theta1_deg", "theta_c": "thetac_deg"}


def stand_us(cfg, mgr, gen):
    """Full 12-channel pulse frame for the standing pose under the current cfg."""
    res = gen.solve(gen.step(0.0))
    bad = [leg for leg, r in res.items() if not r["valid"]]
    if bad:
        raise SystemExit(f"stand pose unreachable for {bad}")
    out = list(mgr._last_us)
    for leg, r in res.items():
        us = mgr.compute_leg_us(cfg, leg, r["theta1"], r["theta_c"], r["hip"])
        for joint, ch in cfg["legs"][leg]["channels"].items():
            out[ch] = us[joint]
    return out


def ramp(mgr, target, seconds):
    start = list(mgr._last_us)
    n = max(1, int(seconds * RATE_HZ))
    for i in range(1, n + 1):
        mgr._last_us[:] = [round(a + (b - a) * i / n) for a, b in zip(start, target)]
        mgr._transmit("sweep")
        time.sleep(1.0 / RATE_HZ)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", default="BL")
    ap.add_argument("--joint", default="theta1", choices=list(OFFSET_KEY))
    ap.add_argument("--start", type=float, default=0.0, help="first offset, deg")
    ap.add_argument("--stop", type=float, default=30.0, help="last offset, deg")
    ap.add_argument("--step", type=float, default=3.0, help="offset step, deg")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold each value")
    add_cli_args(ap)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    key = OFFSET_KEY[args.joint]
    offs = cfg["legs"][args.leg].setdefault("offsets", {})
    original = float(offs.get(key, 0.0))

    link = ArduinoLink(cfg)
    if not link.connect_from_args(args):
        raise SystemExit("No Arduino connection (is run.py / gait_sim.py holding the port?)")
    mgr, gen = link.mgr, GaitGenerator(cfg)
    gen.set_command(0, 0, 0, step_in_place=False)

    ch = cfg["legs"][args.leg]["channels"][args.joint]
    print(f"\nconfig value for {args.leg} {key}: {original:+.1f} deg")
    print("standing up...")
    offs[key] = args.start
    ramp(mgr, stand_us(cfg, mgr, gen), 2.0)

    vals, v = [], args.start
    sgn = 1 if args.stop >= args.start else -1
    while sgn * (args.stop - v) >= -1e-9:
        vals.append(round(v, 3))
        v += sgn * abs(args.step)
    try:
        for v in vals:
            offs[key] = v
            target = stand_us(cfg, mgr, gen)
            ramp(mgr, target, 0.5)
            print(f"\n   >>>  {args.leg} {key} = {v:+6.1f} deg   (ch{ch} = {target[ch]} us)  <<<", flush=True)
            time.sleep(args.hold)
        print("\nsweep done - holding the last value. Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopped (servos hold their last pulse).")


if __name__ == "__main__":
    main()
