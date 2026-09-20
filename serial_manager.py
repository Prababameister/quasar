"""
serial_manager.py — UART interface for the quadruped leg test utility.

Handles:
  - Listing available serial ports
  - Opening / closing a connection
  - Encoding all 12 servo angles as a COBS-framed 24-byte packet and sending

Hardware
--------
  Arduino UNO + one Adafruit PCA9685 (16-channel PWM). The quadruped uses
  channels 0-11: three servos (theta1 / theta_c / hip) per leg. The channel each
  servo sits on is defined per leg in linkage_config.yaml under `legs`.

Angle mapping
-------------
  Each of the 12 physical servos has its own angle -> microsecond calibration
  in linkage_config.yaml (legs.<leg>.calibration.<theta1|theta_c|hip>), given
  as two reference points (deg0, us0) / (deg1, us1). deg0/deg1 are two IK-facing
  joint angles; us0/us1 are whatever pulse width actually drives that specific
  servo to those two angles. The
  conversion is a straight-line fit between those two points, per servo.

  Per-leg offsets (degrees) from linkage_config.yaml are ADDED to each angle
  before conversion. They zero each servo and take up the rotation-direction
  difference on the mirrored right legs.

COBS framing
------------
  payload = 12x uint16 big-endian, ordered by PCA9685 channel 0..11   (24 bytes)
  frame   = cobs_encode(payload) + b'\\x00'
  Matches cobs_motor_prog.ino (PAYLOAD_SIZE 24).
"""

import logging
import pathlib
import struct
import threading
from typing import Optional

import serial
import serial.tools.list_ports
from cobs import cobs

log = logging.getLogger("serial_manager")

CONFIG_PATH = pathlib.Path(__file__).parent / "linkage_config.yaml"

# ── Pulse limits ─────────────────────────────────────────────────────────────
SERVO_MIN_US   =  500
SERVO_MAX_US   = 2500
# Hard safety clamp applied to every commanded pulse regardless of a servo's
# calibration. cobs_motor_prog.ino does no clamping of its own -- it forwards
# whatever microsecond value it receives straight to pwm.writeMicroseconds --
# so this is the only thing standing between a bad calibration value and an
# over-driven servo.
# Must match cobs_motor_prog.ino's own boot-time chan[] default (currently
# 1500 for all 12 channels). This is the value Python assumes every channel
# already holds before it has sent anything; every send transmits a FULL
# 12-channel frame (see _transmit()), so any mismatch here vs. the Arduino's
# real default makes the very first command snap every OTHER channel to
# whatever's wrong here, even though you only meant to touch one.
SERVO_MID_US   = 1166

N_CHANNELS     = 12


def _deg_to_us(value: float, deg0: float, us0: float, deg1: float, us1: float) -> int:
    """
    Linearly map an angle to microseconds using one servo's two calibration
    points (deg0, us0) and (deg1, us1). deg0/deg1 need not be sorted, and the
    angle is NOT clamped to the span between them — a value outside the two
    calibration points is extrapolated along the same line. The final result
    still passes through _clamp_us(), so the hard 500-2500us safety limit
    still applies regardless of the calibration or the input angle.
    """
    slope = (us1 - us0) / (deg1 - deg0)
    return int(round(us0 + slope * (value - deg0)))


def _clamp_us(us: int) -> int:
    return max(SERVO_MIN_US, min(SERVO_MAX_US, us))


# ── Serial manager ────────────────────────────────────────────────────────────

class SerialManager:
    def __init__(self):
        self._port: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._last_us = [500, 500, 1200, 580, 575, 1166, 2420, 2020, 1180, 2310, 1960, 1166]


    # ── Port discovery ────────────────────────────────────────────────────
    @staticmethod
    def list_ports() -> list[dict]:
        ports = []
        for info in sorted(serial.tools.list_ports.comports(),
                           key=lambda p: p.device):
            ports.append({
                "port":        info.device,
                "description": info.description or info.device,
            })
        return ports

    # ── Connection management ─────────────────────────────────────────────
    def connect(self, port: str, baud: int = 115200) -> None:
        with self._lock:
            if self._port and self._port.is_open:
                self._port.close()
                log.info("Closed existing connection before opening new one")
            self._port = serial.Serial(port, baud, timeout=1)
            log.info(f"Connected to {port} @ {baud}")

    def disconnect(self) -> None:
        with self._lock:
            if self._port and self._port.is_open:
                self._port.close()
                log.info("Serial port closed")
            self._port = None

    @property
    def connected(self) -> bool:
        return bool(self._port and self._port.is_open)

    @property
    def port_name(self) -> Optional[str]:
        return self._port.port if self.connected else None

    # ── Send ──────────────────────────────────────────────────────────────
    def compute_leg_us(self, cfg: dict, leg: str,
                        theta1: float, theta_c: float, hip: float) -> dict[str, int]:
        """
        Run one leg's angles through offsets + calibration and return the
        resulting {"theta1": us, "theta_c": us, "hip": us} — the same math
        send_legs() uses, but pure: no serial connection needed, and nothing
        is transmitted or stored. Use this to sanity-check a calibration or
        an angle before sending it to real hardware.

        Raises ValueError for an unknown leg or missing calibration.
        """
        lc = cfg.get("legs", {}).get(leg)
        if lc is None:
            raise ValueError(f"Unknown leg '{leg}' in config")
        off = lc.get("offsets", {})
        cal = lc.get("calibration", {})
        return {
            "theta1":  self._joint_us(cal, "theta1", theta1 + float(off.get("theta1_deg", 0.0))),
            "theta_c": self._joint_us(cal, "theta_c", theta_c + float(off.get("thetac_deg", 0.0))),
            "hip":     self._joint_us(cal, "hip", hip + float(off.get("hip_deg", 0.0))),
        }

    def send_legs(self, cfg: dict,
                  angles: dict[str, tuple[float, float, float]]) -> None:
        """
        Update the given legs' servo channels and transmit a full 12-channel frame.

        angles : {leg_name: (theta1_deg, theta_c_deg, hip_deg)}   — canonical frame

        Channels not present in `angles` keep their last commanded value (the
        Arduino rewrites every channel continuously, so a full frame is always sent).

        Raises RuntimeError if not connected.
        """
        if not self.connected:
            raise RuntimeError("Not connected to a serial port")

        legs = cfg.get("legs", {})
        for leg, (t1, tc, hip) in angles.items():
            us = self.compute_leg_us(cfg, leg, t1, tc, hip)
            ch = legs[leg]["channels"]
            self._last_us[ch["theta1"]]  = us["theta1"]
            self._last_us[ch["theta_c"]] = us["theta_c"]
            self._last_us[ch["hip"]]     = us["hip"]

        self._transmit(f"send_legs {list(angles)}")

    @staticmethod
    def _joint_us(calibration: dict, joint: str, angle_deg: float) -> int:
        """Convert one joint's angle to microseconds using its servo's calibration."""
        c = calibration.get(joint)
        if c is None:
            raise ValueError(
                f"Missing calibration.{joint} in leg config — add "
                f"{{deg0, us0, deg1, us1}} under that leg's 'calibration' block")
        return _clamp_us(_deg_to_us(angle_deg, c["deg0"], c["us0"], c["deg1"], c["us1"]))

    def send_leg(self, cfg: dict, leg: str,
                 theta1: float, theta_c: float, hip: float) -> None:
        """Convenience: command a single leg (still transmits the full frame)."""
        self.send_legs(cfg, {leg: (theta1, theta_c, hip)})

    def send_raw(self, channel: int, us: int) -> None:
        """
        Set one PCA9685 channel directly to a pulse width in microseconds,
        bypassing all angle/offset math. For testing PWM/servo wiring
        independently of IK and leg config — e.g. before trusting send_leg().

        Raises RuntimeError if not connected, ValueError for a bad channel.
        """
        if not self.connected:
            raise RuntimeError("Not connected to a serial port")
        if not (0 <= channel < N_CHANNELS):
            raise ValueError(f"channel must be 0..{N_CHANNELS - 1}")

        self._last_us[channel] = _clamp_us(int(us))
        self._transmit(f"send_raw ch={channel}")

    # ── Transmit ──────────────────────────────────────────────────────────
    def _transmit(self, tag: str) -> None:
        """Pack the current per-channel state and write one COBS frame."""
        payload = struct.pack(">" + "H" * N_CHANNELS, *self._last_us)
        frame   = cobs.encode(payload) + b"\x00"

        log.info("%s -> us=%s frame=%s", tag, self._last_us, frame.hex())

        with self._lock:
            self._port.write(frame)


# ── Module-level singleton ────────────────────────────────────────────────────
_manager = SerialManager()


def get_manager() -> SerialManager:
    return _manager
