"""
linkage_solver.py — FK solver (analytical) + SolveSpace constraint solver.

Public API
----------
fk_solve(cfg, input_angles)  -> {point_name: (x, y)}
solve(cfg, input_angles)     -> {point_name: (x, y)}   (SolveSpace backend)
load_config(path)            -> dict
"""

import math
import yaml
from python_solvespace import SolverSystem, ResultFlag


# ── Config ────────────────────────────────────────────────────────────

def load_config(path="linkage_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Shared helpers ────────────────────────────────────────────────────

def _find_edge_length(cfg, a, b):
    for edge in cfg["edges"]:
        if (edge["from"] == a and edge["to"] == b) or \
           (edge["from"] == b and edge["to"] == a):
            return edge["length"]
    return None


def _get_line(sys_, wp, line_handles, handles, a, b):
    if (a, b) in line_handles:
        return line_handles[(a, b)]
    if (b, a) in line_handles:
        return line_handles[(b, a)]
    return sys_.add_line_2d(handles[a], handles[b], wp)


# ── Analytical FK ─────────────────────────────────────────────────────

def _circle_intersect(cx1, cy1, r1, cx2, cy2, r2, want_upper=True):
    """
    Return one intersection point of two circles.
    Raises ValueError if circles don't intersect.
    """
    dx, dy = cx2 - cx1, cy2 - cy1
    d = math.hypot(dx, dy)
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9 or d < 1e-9:
        raise ValueError(
            f"Circles don't intersect: "
            f"c1=({cx1:.2f},{cy1:.2f}) r={r1:.3f}  "
            f"c2=({cx2:.2f},{cy2:.2f}) r={r2:.3f}  d={d:.3f}"
        )
    a  = (r1**2 - r2**2 + d**2) / (2 * d)
    h  = math.sqrt(max(r1**2 - a**2, 0.0))
    mx = cx1 + a * dx / d
    my = cy1 + a * dy / d
    px =  h * dy / d
    py = -h * dx / d
    if want_upper:
        if py < 0 or (py == 0 and px < 0):
            px, py = -px, -py
    else:
        if py > 0 or (py == 0 and px > 0):
            px, py = -px, -py
    return (mx + px, my + py)


def fk_solve(cfg: dict, input_angles: dict) -> dict:
    """
    Closed-form forward kinematics.

    Chain
    -----
    O  → A   : driven by theta1
    A, Q → B : circle-circle intersection
    B, Q → C : Q→C is Q→B rotated 90° CCW (enforces B→Q→C = 90°)
    Q  → P1  : driven by theta_c
    C, P1→P2 : circle-circle intersection
    P1, P2→D : extend P2→P1 direction by |P1D|
    """
    pts = {}

    # Fixed points
    for name, p in cfg["points"].items():
        if p.get("fixed", False):
            pts[name] = (float(p["x"]), float(p["y"]))

    ox, oy = pts["O"]
    qx, qy = pts["Q"]

    def L(a, b):
        v = _find_edge_length(cfg, a, b)
        if v is None:
            raise ValueError(f"No edge {a}↔{b} in config")
        return v

    # A — driven by theta1
    theta1 = math.radians(input_angles.get("theta1", 45.0))
    pts["A"] = (ox + L("O", "A") * math.cos(theta1),
                oy + L("O", "A") * math.sin(theta1))
    ax, ay = pts["A"]

    # B — circle-circle intersection (A, Q)
    pts["B"] = _circle_intersect(ax, ay, L("A", "B"),
                                  qx, qy, L("Q", "B"),
                                  want_upper=True)
    bx, by = pts["B"]

    # C — Q→B rotated 90° CW, scaled to |QC|
    qb_dx, qb_dy = bx - qx, by - qy
    qb_len = math.hypot(qb_dx, qb_dy)
    l_qc   = L("Q", "C")
    ux, uy = qb_dy / qb_len, -qb_dx / qb_len
    pts["C"] = (qx + ux * l_qc, qy + uy * l_qc)
    cx, cy = pts["C"]

    # P1 — driven by theta_c
    theta_c = math.radians(input_angles.get("theta_c", -30.0))
    pts["P1"] = (qx + L("Q", "P1") * math.cos(theta_c),
                 qy + L("Q", "P1") * math.sin(theta_c))
    p1x, p1y = pts["P1"]

    # Seed P2 via parallelogram identity to select correct assembly branch
    cfg["points"]["P2"]["x"] = p1x + (cx - qx)
    cfg["points"]["P2"]["y"] = p1y + (cy - qy)

    # P2 — circle-circle intersection (C, P1)
    pts["P2"] = _circle_intersect(cx, cy, L("C", "P2"),
                                   p1x, p1y, L("P1", "P2"),
                                   want_upper=True)
    p2x, p2y = pts["P2"]

    # D — extend P2→P1 direction by |P1D|
    ddx, ddy = p1x - p2x, p1y - p2y
    dmag = math.hypot(ddx, ddy)
    l_p1d = L("P1", "D")
    pts["D"] = (p1x + ddx / dmag * l_p1d,
                p1y + ddy / dmag * l_p1d)

    return pts


# ── SolveSpace constraint solver ──────────────────────────────────────

def solve(cfg: dict, input_angles: dict) -> dict:
    """
    Build a fresh SolverSystem from cfg, apply input angles, solve.
    Returns {point_name: (x, y)} for every point in the config.
    """
    sys_ = SolverSystem()
    wp   = sys_.create_2d_base()

    fixed_names = {n for n, p in cfg["points"].items() if p.get("fixed", False)}
    driven_map  = {inp["point"]: inp for inp in cfg.get("inputs", [])}

    # Seed P1 and P2 for correct assembly branch
    theta_c_inp = next(i for i in cfg["inputs"] if i["name"] == "theta_c")
    theta_c_deg = input_angles.get("theta_c", theta_c_inp["angle_deg"])
    theta_c_rad = math.radians(theta_c_deg)

    qx = cfg["points"]["Q"]["x"]
    qy = cfg["points"]["Q"]["y"]
    L_qp1 = _find_edge_length(cfg, "Q", "P1")

    p1x = qx + L_qp1 * math.cos(theta_c_rad)
    p1y = qy + L_qp1 * math.sin(theta_c_rad)
    cfg["points"]["P1"]["x"] = p1x
    cfg["points"]["P1"]["y"] = p1y

    cx = cfg["points"]["C"].get("x", 0.0)
    cy = cfg["points"]["C"].get("y", 0.0)
    cfg["points"]["P2"]["x"] = p1x + (cx - qx)
    cfg["points"]["P2"]["y"] = p1y + (cy - qy)

    # Add all points
    handles = {}
    for name, p in cfg["points"].items():
        x = p.get("x", 0.0)
        y = p.get("y", 0.0)
        h = sys_.add_point_2d(float(x), float(y), wp)
        handles[name] = h
        if name in fixed_names:
            sys_.dragged(h, wp)

    # Reference line for angle constraints
    _ref_origin = sys_.add_point_2d(0.0, 0.0, wp)
    _ref_tip    = sys_.add_point_2d(1.0, 0.0, wp)
    sys_.dragged(_ref_origin, wp)
    sys_.dragged(_ref_tip, wp)
    sys_.add_line_2d(_ref_origin, _ref_tip, wp)

    # Driven points
    for name, inp in driven_map.items():
        pivot_name = inp["pivot"]
        angle_deg  = input_angles.get(inp["name"], inp["angle_deg"])
        angle_rad  = math.radians(angle_deg)
        length = _find_edge_length(cfg, pivot_name, name)
        if length is None:
            raise ValueError(f"No edge between '{pivot_name}' and '{name}'")
        px = cfg["points"][pivot_name].get("x", 0.0)
        py = cfg["points"][pivot_name].get("y", 0.0)
        sys_.set_params(handles[name].params,
                        (px + length * math.cos(angle_rad),
                         py + length * math.sin(angle_rad)))
        sys_.dragged(handles[name], wp)

    # Distance constraints — skip P1→D (computed analytically)
    line_handles = {}
    for edge in cfg["edges"]:
        a, b = edge["from"], edge["to"]
        if {a, b} == {"P1", "D"}:
            continue
        a_pinned = a in fixed_names or a in driven_map
        b_pinned = b in fixed_names or b in driven_map
        line_handles[(a, b)] = sys_.add_line_2d(handles[a], handles[b], wp)
        if not (a_pinned and b_pinned):
            sys_.distance(handles[a], handles[b], edge["length"], wp)

    # Angle constraints
    for ac in cfg.get("angle_constraints", []):
        angle_deg = float(ac["angle_deg"])
        if "edge" in ac:
            a, b  = ac["edge"]
            line  = _get_line(sys_, wp, line_handles, handles, a, b)
            pivot_h = handles[a]
            px, py  = sys_.params(pivot_h.params)
            tip_h   = sys_.add_point_2d(px + 1.0, py, wp)
            sys_.dragged(tip_h, wp)
            local_ref = sys_.add_line_2d(pivot_h, tip_h, wp)
            sys_.angle(local_ref, line, angle_deg, wp)
        elif "edge1" in ac and "edge2" in ac:
            a1, b1 = ac["edge1"]
            a2, b2 = ac["edge2"]
            line1  = _get_line(sys_, wp, line_handles, handles, a1, b1)
            line2  = _get_line(sys_, wp, line_handles, handles, a2, b2)
            sys_.angle(line1, line2, angle_deg, wp)
        else:
            raise ValueError(f"Bad angle_constraint: {ac}")

    result = sys_.solve()
    if result != ResultFlag.OKAY:
        raise RuntimeError(
            f"Solver returned {result.name}. Failed: {sys_.failures()}"
        )

    pts = {name: tuple(sys_.params(h.params)) for name, h in handles.items()}

    # D: extend P2→P1 direction
    p1  = pts["P1"]
    p2  = pts["P2"]
    ddx, ddy = p1[0] - p2[0], p1[1] - p2[1]
    dmag = math.hypot(ddx, ddy)
    l_p1d = _find_edge_length(cfg, "P1", "D")
    pts["D"] = (p1[0] + ddx / dmag * l_p1d,
                p1[1] + ddy / dmag * l_p1d)

    return pts
