"""
run.py — drive the quadruped from the keyboard, no web server needed.

    w : walk forward         a : rotate left (counter-clockwise)
    s : walk backward        d : rotate right (clockwise)
    j : jump straight up (only while stopped)
    space / x : stop (feet parked at the start of the gait path)
    q / Ctrl-C : stop and quit

Motion latches: a key keeps the robot moving until you press another key, so
there is no need to hold anything down (a terminal only reports key presses,
not releases). Pressing the key for the current motion again does nothing.

Usage:
    python run.py                     # auto-detect the serial port
    python run.py --port /dev/ttyACM0 [--baud 115200]
"""

import argparse
import logging
import select
import sys
import termios
import tty

import yaml

from gait import WalkController
from serial_manager import get_manager, CONFIG_PATH

# key -> (direction, turn), matching WalkController.start()
KEYS = {
    "w": (1, 0),
    "s": (-1, 0),
    "a": (0, -1),
    "d": (0, 1),
}
LABELS = {
    (1, 0): "forward",
    (-1, 0): "backward",
    (0, -1): "rotating left",
    (0, 1): "rotating right",
    (0, 0): "stopped",
}


def find_port(mgr) -> str:
    ports = mgr.list_ports()
    if not ports:
        sys.exit("No serial ports found — pass one with --port")
    # Prefer USB-serial adapters (Arduino shows up as ttyACM*/ttyUSB*).
    for p in ports:
        if "ACM" in p["port"] or "USB" in p["port"]:
            return p["port"]
    return ports[0]["port"]


def status_line(text: str) -> None:
    # Raw mode disables newline translation, so use \r explicitly.
    sys.stdout.write(f"\r\x1b[K{text}")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)  # serial_manager logs every frame at INFO

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    mgr = get_manager()
    port = args.port or find_port(mgr)
    mgr.connect(port, args.baud)
    print(f"Connected to {port} @ {args.baud}")

    walker = WalkController(cfg, mgr)
    print(__doc__.split("Usage:")[0].strip())
    print()

    if not sys.stdin.isatty():
        sys.exit("stdin is not a terminal — run this from an interactive shell")

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    current = (0, 0)
    try:
        tty.setcbreak(fd)
        status_line(LABELS[current])
        while True:
            if not select.select([sys.stdin], [], [], 0.5)[0]:
                continue
            ch = sys.stdin.read(1).lower()
            if ch in ("q", "\x03"):
                break
            if ch == "j":
                if current != (0, 0):
                    continue  # stop before jumping
                try:
                    walker.jump()
                except (ValueError, RuntimeError) as e:
                    status_line(f"jump: {e}")
                else:
                    status_line("jumping")
                continue
            if ch in (" ", "x"):
                cmd = (0, 0)
            elif ch in KEYS:
                cmd = KEYS[ch]
            else:
                continue
            if cmd == current:
                continue
            if cmd == (0, 0):
                walker.stop()
            else:
                walker.start(*cmd)
            current = cmd
            status_line(LABELS[current])
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        print("\nStopping…")
        walker.stop()
        walker.shutdown()
        mgr.disconnect()


if __name__ == "__main__":
    main()
