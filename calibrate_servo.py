"""
calibrate_servo.py - jog ONE servo by raw pulse width to find its calibration.

    uv run calibrate_servo.py BL theta1            # leg + joint (theta1 | theta_c | hip)
    uv run calibrate_servo.py BL theta1 --port COM4

Bypasses all angle/offset math: you type pulse widths, the servo moves.
Starts at the pulse the CURRENT calibration gives for deg0, so you can see
how far off it is. Find the pulse that puts the joint exactly at deg0, then
at deg1 (angles in the hardware frame described in linkage_config.yaml), and
copy those into that leg's `calibration:` block.

Commands at the prompt:
  1250      go to 1250 us             +10 / -10   nudge by that many us
  0 / 90    go to where the CURRENT calibration says that angle is
  a / b     record the current pulse as us0 / us1
  q         quit (prints the yaml line to paste)
"""

import argparse
import time

import yaml

from arduino_link import ArduinoLink, add_cli_args
from serial_manager import _deg_to_us, _clamp_us

CFG = yaml.safe_load(open("linkage_config.yaml", encoding="utf-8"))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("leg", choices=list(CFG["legs"]))
ap.add_argument("joint", choices=["theta1", "theta_c", "hip"])
add_cli_args(ap)
args = ap.parse_args()

lc = CFG["legs"][args.leg]
ch = lc["channels"][args.joint]
cal = dict(lc["calibration"][args.joint])
print(f"{args.leg}.{args.joint} -> channel {ch}")
print(f"current calibration: deg0 {cal['deg0']} = {cal['us0']} us,  deg1 {cal['deg1']} = {cal['us1']} us")

link = ArduinoLink(CFG)
if not link.connect_from_args(args):
    raise SystemExit("No Arduino connection.")
time.sleep(2.0)   # opening the port resets the UNO; wait for it to boot

us = int(cal["us0"])
rec = {"us0": None, "us1": None}


def go(v):
    global us
    us = _clamp_us(int(v))
    link.mgr.send_raw(ch, us)
    print(f"  -> {us} us")


go(us)
try:
    while True:
        s = input(f"[{us} us] > ").strip().lower()
        if not s:
            continue
        if s == "q":
            break
        if s in ("a", "b"):
            key = "us0" if s == "a" else "us1"
            rec[key] = us
            print(f"  recorded {key} = {us}  (deg{'0' if s == 'a' else '1'} = {cal['deg' + ('0' if s == 'a' else '1')]})")
            continue
        try:
            if s[0] in "+-":
                go(us + int(s))
            else:
                v = float(s)
                if abs(v) <= 360:   # small number = angle via current calibration
                    go(_deg_to_us(v, cal["deg0"], cal["us0"], cal["deg1"], cal["us1"]))
                else:
                    go(v)
        except ValueError:
            print("  bad input (try 1250, +10, -10, 0, 90, a, b, q)")
except (KeyboardInterrupt, EOFError):
    pass
finally:
    time.sleep(0.1)
    link.close()

us0 = rec["us0"] if rec["us0"] is not None else cal["us0"]
us1 = rec["us1"] if rec["us1"] is not None else cal["us1"]
print(f"\nPaste into legs.{args.leg}.calibration in linkage_config.yaml:")
print(f"      {args.joint}: {{deg0: {cal['deg0']}, us0: {us0}, deg1: {cal['deg1']}, us1: {us1}}}")
