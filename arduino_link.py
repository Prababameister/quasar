"""
arduino_link.py - shared "send this leg's joint angles to the Arduino" helper
used by ik_interactive.py, fk_interactive.py and send_angles.py.

Wraps serial_manager.SerialManager (angle -> us calibration + COBS framing)
and adds: port auto-detect, syncing the starting channel values with the
firmware's boot values, an on/off switch, skipping duplicate sends, and a
one-line status string for display.
"""

import re

from serial_manager import SerialManager

INO_PATH = "cobs_motor_prog.ino"


def add_cli_args(ap):
    ap.add_argument("--port", help="serial port, e.g. COM4 (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-serial", action="store_true", help="don't talk to the Arduino")


def firmware_boot_us():
    """The channel values cobs_motor_prog.ino holds at boot (its chan[] array).
    Every frame carries all 12 channels, so the manager has to start from the
    same values or the first send snaps the OTHER legs somewhere else."""
    try:
        src = open(INO_PATH, encoding="utf-8").read()
        body = re.search(r"chan\[NUM_CHANNELS\]\s*=\s*\{([^}]*)\}", src).group(1)
        vals = [int(v) for v in re.findall(r"\d+", body)]
        return vals if len(vals) == 12 else None
    except Exception:
        return None


def pick_port(port=None):
    if port:
        return port
    ports = SerialManager.list_ports()
    likely = [p for p in ports if re.search(r"arduino|ch340|usb.?serial|wch",
                                            p["description"], re.I)]
    if len(likely) == 1:
        return likely[0]["port"]
    if len(ports) == 1:
        return ports[0]["port"]
    if ports:
        print("Several serial ports found, pass one with --port:")
        for p in ports:
            print(f"  {p['port']:8s} {p['description']}")
    else:
        print("No serial ports found - running without the Arduino.")
    return None


class ArduinoLink:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mgr = SerialManager()
        self.enabled = False
        self._last_sent = None

    @property
    def connected(self):
        return self.mgr.connected

    def connect_from_args(self, args):
        if args.no_serial:
            return False
        port = pick_port(args.port)
        if not port:
            return False
        try:
            print(f"Connecting to Arduino on {port} ...")
            self.mgr.connect(port, args.baud)   # opening the port resets the UNO
        except Exception as e:
            print(f"Could not open {port}: {e}")
            print("Running without the Arduino.")
            return False
        boot = firmware_boot_us()
        if boot:
            self.mgr._last_us = boot
        else:
            print(f"WARNING: couldn't read chan[] from {INO_PATH}; other legs "
                  "may jump on the first send.")
        self.enabled = True
        print("Connected.")
        return True

    def set_enabled(self, on):
        self.enabled = bool(on)
        self._last_sent = None   # resend the current pose when re-enabled

    def send(self, leg, theta1, theta_c, hip, blockers=()):
        """Send one leg's joint angles (deg). Returns a status line."""
        if not self.connected:
            return "ARDUINO: not connected"
        name = self.mgr.port_name
        if not self.enabled:
            return f"ARDUINO: {name}  (sending off)"
        if blockers:
            return f"ARDUINO: {name}  NOT SENT ({blockers[0]})"
        key = (leg, round(theta1, 2), round(theta_c, 2), round(hip, 2))
        if key == self._last_sent:
            return f"ARDUINO: {name}  sent"
        try:
            self.mgr.send_leg(self.cfg, leg, theta1, theta_c, hip)
        except Exception as e:
            self.mgr.disconnect()
            return f"ARDUINO: send failed, disconnected ({str(e)[:40]})"
        self._last_sent = key
        return f"ARDUINO: {name}  sent"

    def close(self):
        self.mgr.disconnect()
