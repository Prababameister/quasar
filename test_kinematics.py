"""
test_kinematics.py — sanity checks for 4-leg kinematics, gait, and serial framing.

Run:  .venv/bin/python test_kinematics.py
No test framework — asserts + a PASS/FAIL summary.
"""

import math
import struct

import yaml
from cobs import cobs

from leg_kinematics import solve_leg, fk_leg, leg_names
from gait import foot_offset, _gait_params, _PHASE_OFFSET
from serial_manager import SerialManager, _deg_to_us, N_CHANNELS

CFG = yaml.safe_load(open("linkage_config.yaml"))

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── 1. Mirror correctness ─────────────────────────────────────────────
print("\n[mirror]")
tx, ty = 150.0, -120.0
left = solve_leg(CFG, "FL", tx, ty)
right = solve_leg(CFG, "FR", tx, ty)    # same canonical target
check("FL solves", left["valid"], left)
check("FR solves", right["valid"], right)
check("FL and FR yield identical canonical angles",
      abs(left["theta1"] - right["theta1"]) < 1e-6 and
      abs(left["theta_c"] - right["theta_c"]) < 1e-6,
      (left["theta1"], left["theta_c"], right["theta1"], right["theta_c"]))

# fk_leg round-trips angles back to the canonical foot point for both sides
fk_l = fk_leg(CFG, "FL", left["theta1"], left["theta_c"])
dl = fk_l["points"]["D"]
check("fk_leg(FL) foot near target",
      math.hypot(dl["x"] - tx, dl["y"] - ty) < 1.5, dl)
fk_r = fk_leg(CFG, "FR", right["theta1"], right["theta_c"])
dr = fk_r["points"]["D"]
check("fk_leg(FR) foot near target",
      math.hypot(dr["x"] - tx, dr["y"] - ty) < 1.5, dr)
check("fk_leg reports side", fk_r["side"] == "right" and fk_l["side"] == "left")

# ── 2. Gait continuity ───────────────────────────────────────────────
print("\n[gait]")
gp = _gait_params(CFG)
duty = gp["duty"]

def foot(ph, direction=1):
    return foot_offset(ph, gp, direction)

eps = 1e-4
# stance->swing boundary
a = foot(duty - eps); b = foot(duty + eps)
check("continuous at stance->swing",
      math.hypot(a[0] - b[0], a[1] - b[1]) < 0.05, (a, b))
# swing->stance wrap (phase ~1 -> ~0)
a = foot(1.0 - eps); b = foot(eps)
check("continuous across phase wrap",
      math.hypot(a[0] - b[0], a[1] - b[1]) < 0.05, (a, b))
# lift is zero at both ends of swing, positive in the middle
check("no lift at swing start", abs(foot(duty + eps)[1] - gp["stance_y_mm"]) < 0.05)
check("no lift at swing end",   abs(foot(1.0 - eps)[1] - gp["stance_y_mm"]) < 0.05)
check("lift positive mid-swing", foot((duty + 1.0) / 2)[1] > gp["stance_y_mm"] + 1.0)
# direction 0 -> parked at mid-stance
check("direction 0 parks at stance point",
      foot(0.37, 0) == (gp["stance_x_mm"], gp["stance_y_mm"]))
# reversing direction mirrors the fore-aft position
check("backward mirrors forward",
      abs(foot(0.2, 1)[0] + foot(0.2, -1)[0] - 2 * gp["stance_x_mm"]) < 1e-9)

# trot phase groups: diagonal pairs share phase, adjacent differ by 0.5
check("trot phasing", _PHASE_OFFSET["FL"] == _PHASE_OFFSET["BR"]
      and _PHASE_OFFSET["FR"] == _PHASE_OFFSET["BL"]
      and abs(_PHASE_OFFSET["FL"] - _PHASE_OFFSET["FR"]) == 0.5)

# every gait pose is reachable for every leg
print("\n[gait reachability]")
worst = 0.0
for leg in leg_names(CFG):
    for i in range(24):
        ph = (i / 24 + _PHASE_OFFSET.get(leg, 0.0)) % 1.0
        x, y = foot(ph)
        res = solve_leg(CFG, leg, x, y)
        worst = max(worst, res["error_mm"])
check(f"all trot poses reachable (worst err {worst:.2f} mm)", worst < 3.0)

# ── 3. Serial framing ────────────────────────────────────────────────
print("\n[serial]")
mgr = SerialManager()

class FakePort:
    is_open = True
    port = "fake"
    def __init__(self): self.written = b""
    def write(self, b): self.written += b
    def close(self): self.is_open = False

fp = FakePort()
mgr._port = fp

angles = {leg: (150.0, -10.0, 90.0) for leg in leg_names(CFG)}
mgr.send_legs(CFG, angles)

frame = fp.written
check("frame ends with 0x00 delimiter", frame.endswith(b"\x00"))
check("no interior 0x00", 0 not in frame[:-1])

payload = cobs.decode(frame[:-1])
check("payload is 24 bytes", len(payload) == N_CHANNELS * 2, len(payload))

us = struct.unpack(">" + "H" * N_CHANNELS, payload)
# check the FL channels carry the expected microseconds
fl = CFG["legs"]["FL"]["channels"]
fl_cal = CFG["legs"]["FL"]["calibration"]
exp_t1 = _deg_to_us(150.0, fl_cal["theta1"]["deg0"], fl_cal["theta1"]["us0"],
                    fl_cal["theta1"]["deg1"], fl_cal["theta1"]["us1"])
exp_tc = _deg_to_us(-10.0, fl_cal["theta_c"]["deg0"], fl_cal["theta_c"]["us0"],
                    fl_cal["theta_c"]["deg1"], fl_cal["theta_c"]["us1"])
exp_hip = _deg_to_us(90.0, fl_cal["hip"]["deg0"], fl_cal["hip"]["us0"],
                     fl_cal["hip"]["deg1"], fl_cal["hip"]["us1"])
check("FL theta1 channel us", us[fl["theta1"]] == exp_t1, (us[fl["theta1"]], exp_t1))
check("FL theta_c channel us", us[fl["theta_c"]] == exp_tc, (us[fl["theta_c"]], exp_tc))
check("FL hip channel us", us[fl["hip"]] == exp_hip, (us[fl["hip"]], exp_hip))
check("all 12 channels populated", all(500 <= v <= 2500 for v in us), us)

# offset is actually applied
mgr2 = SerialManager()
fp2 = FakePort(); mgr2._port = fp2
cfg2 = yaml.safe_load(open("linkage_config.yaml"))
cfg2["legs"]["FL"]["offsets"]["theta1_deg"] = 12.0
mgr2.send_legs(cfg2, {"FL": (150.0, -10.0, 90.0)})
us2 = struct.unpack(">" + "H" * N_CHANNELS, cobs.decode(fp2.written[:-1]))
check("theta1 offset shifts the pulse",
      us2[fl["theta1"]] == _deg_to_us(162.0, fl_cal["theta1"]["deg0"], fl_cal["theta1"]["us0"],
                                       fl_cal["theta1"]["deg1"], fl_cal["theta1"]["us1"]),
      (us2[fl["theta1"]],))

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*40}\n{_passed} passed, {_failed} failed\n{'='*40}")
raise SystemExit(1 if _failed else 0)
