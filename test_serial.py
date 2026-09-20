"""
test_serial.py — interactively drive ONE leg of the quadruped over serial.

Usage:
    .venv/bin/python test_serial.py --port /dev/ttyACM0
    .venv/bin/python test_serial.py --port /dev/ttyACM0 --leg FR

Goes through the same path as the web app: per-leg servo offsets and PCA9685
channel map come from linkage_config.yaml, and every send transmits a full
12-channel COBS frame. Only the selected leg's three channels change per send;
the other nine hold their last value (neutral on a fresh connection).

At the prompt:
    pwm <channel> <us>  send a RAW pulse width (500-2500 us) directly to a
                        PCA9685 channel (0-11), bypassing all angle/offset
                        math. Use this to check a servo's wiring/PWM response
                        before trusting the angle -> us conversion at all.
    <t1> <tc> [hip]     send joint angles in degrees
                        (hip defaults to hip_fixed_deg from the config)
    calc <t1> <tc> [hip]  compute the microseconds the active leg's offsets +
                        calibration would produce for these angles — prints
                        the result WITHOUT sending anything or needing a
                        connection. Use this to sanity-check a calibration
                        before trusting it on the real servo.
    ik <x> <y>          solve IK for the active leg at foot target (x, y) mm,
                        print the angles, and send if the solution is valid
    leg <FL|FR|BL|BR>   switch the active leg
    legs               list legs with their channels / offsets
    n | neutral        re-send the active leg at hip_fixed_deg and mid-range joints
    r | reload         re-read linkage_config.yaml (picks up offset edits)
    q | quit           exit
"""

import argparse
import logging
import pathlib
import threading

import yaml

from serial_manager import SerialManager
from leg_kinematics import solve_leg, hip_fixed_deg, leg_names, nominal_target

CONFIG_PATH = pathlib.Path(__file__).parent / "linkage_config.yaml"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("test_serial")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def listener(mgr: SerialManager, stop_event: threading.Event) -> None:
    """Print anything the Arduino sends back (0xFF = malformed packet)."""
    while not stop_event.is_set():
        port = mgr._port
        if not (port and port.is_open):
            stop_event.wait(0.2)
            continue
        try:
            b = port.read(1)
        except Exception:
            break
        if b == b"\xff":
            print("\n[arduino] ERROR: malformed packet", flush=True)
        elif b:
            print(f"\n[arduino] unexpected byte: 0x{b.hex()}", flush=True)


def print_legs(cfg: dict) -> None:
    for name, lc in cfg.get("legs", {}).items():
        ch = lc["channels"]
        off = lc.get("offsets", {})
        print(f"  {name:3s}  side={lc['side']:5s}  "
              f"ch(t1/tc/hip)={ch['theta1']}/{ch['theta_c']}/{ch['hip']}  "
              f"offsets(t1/tc/hip)="
              f"{off.get('theta1_deg', 0.0)}/{off.get('thetac_deg', 0.0)}/"
              f"{off.get('hip_deg', 0.0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-leg serial tester.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--leg", default=None, help="Leg to start on (default: first)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't open a port; just print the frames that would be sent")
    args = parser.parse_args()

    cfg = load_config()
    legs = leg_names(cfg)
    if not legs:
        raise SystemExit("No legs defined in linkage_config.yaml")

    leg = args.leg or legs[0]
    if leg not in legs:
        raise SystemExit(f"Unknown leg {leg!r}; choose from {legs}")

    mgr = SerialManager()
    if args.dry_run:
        class _DryPort:
            is_open, port = True, "dry-run"
            def write(self, b): pass          # SerialManager logs us + frame hex
            def read(self, n=1): return b""
            def close(self): pass
        mgr._port = _DryPort()
        print(f"Dry run — no port opened. Active leg: {leg}")
    else:
        print(f"Connecting to {args.port} @ {args.baud}…")
        mgr.connect(args.port, args.baud)
        print(f"Connected. Active leg: {leg}")
    print("Type 'pwm ch us', 't1 tc [hip]', 'calc t1 tc [hip]', 'ik x y', "
          "'leg <name>', 'legs', 'n', 'r', or 'q'.\n")

    stop_event = threading.Event()
    t = threading.Thread(target=listener, args=(mgr, stop_event), daemon=True)
    t.start()

    hip = hip_fixed_deg(cfg)

    def send(t1: float, tc: float, h: float) -> None:
        mgr.send_leg(cfg, leg, t1, tc, h)
        print(f"  sent {leg}: t1={t1:.1f}  tc={tc:.1f}  hip={h:.1f}")

    try:
        while True:
            try:
                raw = input(f"[{leg}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()

            if cmd in ("q", "quit", "exit"):
                print("Exiting.")
                break

            if cmd == "legs":
                print_legs(cfg)
                continue

            if cmd in ("r", "reload"):
                cfg = load_config()
                hip = hip_fixed_deg(cfg)
                print("  config reloaded")
                continue

            if cmd == "pwm":
                if len(parts) != 3:
                    print("  usage: pwm <channel 0-11> <us 500-2500>")
                    continue
                try:
                    channel, us = int(parts[1]), int(parts[2])
                except ValueError:
                    print("  channel and us must be integers")
                    continue
                try:
                    mgr.send_raw(channel, us)
                    print(f"  sent raw ch={channel}: {us} us "
                          f"(clamped to {max(500, min(2500, us))})")
                except (ValueError, RuntimeError) as e:
                    print(f"  {e}")
                continue

            if cmd == "calc":
                if len(parts) not in (3, 4):
                    print("  usage: calc <t1> <tc> [hip]")
                    continue
                try:
                    nums = [float(p) for p in parts[1:]]
                except ValueError:
                    print("  t1/tc/hip must be numbers")
                    continue
                t1, tc = nums[0], nums[1]
                h = nums[2] if len(nums) == 3 else hip
                try:
                    us = mgr.compute_leg_us(cfg, leg, t1, tc, h)
                except ValueError as e:
                    print(f"  {e}")
                    continue
                print(f"  {leg}: t1={t1:.1f}->{us['theta1']}us  "
                      f"tc={tc:.1f}->{us['theta_c']}us  "
                      f"hip={h:.1f}->{us['hip']}us  (not sent)")
                continue

            if cmd == "leg":
                if len(parts) != 2 or parts[1] not in leg_names(cfg):
                    print(f"  usage: leg <{'|'.join(leg_names(cfg))}>")
                    continue
                leg = parts[1]
                print(f"  active leg: {leg}")
                continue

            if cmd in ("n", "neutral"):
                res = solve_leg(cfg, leg, *nominal_target(cfg))
                if not res["valid"]:
                    print("  nominal gait stance unreachable — not sent")
                    continue
                send(res["theta1"], res["theta_c"], res["hip"])
                continue

            if cmd == "ik":
                if len(parts) != 3:
                    print("  usage: ik <x> <y>   (mm, canonical left-leg frame)")
                    continue
                try:
                    x, y = float(parts[1]), float(parts[2])
                except ValueError:
                    print("  x and y must be numbers")
                    continue
                res = solve_leg(cfg, leg, x, y)
                print(f"  IK: t1={res['theta1']:.2f}  tc={res['theta_c']:.2f}  "
                      f"err={res['error_mm']:.2f} mm  valid={res['valid']}")
                if res["valid"]:
                    send(res["theta1"], res["theta_c"], res["hip"])
                else:
                    print("  not sent (unreachable) — send explicit angles to override")
                continue

            # Otherwise: bare angles "t1 tc [hip]"
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                print("  unrecognised command")
                continue
            if len(nums) == 2:
                send(nums[0], nums[1], hip)
            elif len(nums) == 3:
                send(nums[0], nums[1], nums[2])
            else:
                print("  enter 2 or 3 numbers: t1 tc [hip]")

    finally:
        stop_event.set()
        mgr.disconnect()


if __name__ == "__main__":
    main()
