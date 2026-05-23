"""
Linkage Solver — python-solvespace backend
===========================================
Reads linkage_config.yaml and solves all free joint positions using
the SolveSpace geometric constraint kernel.

Usage
-----
    python linkage_solver.py                       # uses linkage_config.yaml
    python linkage_solver.py my_config.yaml

Importing
---------
    from linkage_solver import load_config, solve

    cfg    = load_config("linkage_config.yaml")
    points = solve(cfg, {"theta1": 60.0, "theta_c": -45.0})
    print(points["E"])   # -> (x, y) tuple
"""

import sys
import math
import yaml
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from python_solvespace import SolverSystem, ResultFlag

# ── Forward Kinematics ────────────────────────────────────────────────

def _circle_intersect(cx1, cy1, r1, cx2, cy2, r2, want_upper=True):
    """
    Return one of the two intersection points of two circles.
    want_upper=True  picks the point with the larger y (or whichever
    side the correct assembly is on — flip as needed per joint).
    Raises ValueError if circles don't intersect.
    """
    dx, dy = cx2 - cx1, cy2 - cy1
    d = math.hypot(dx, dy)
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9 or d < 1e-9:
        raise ValueError(
            f"Circles don't intersect: c1=({cx1:.2f},{cy1:.2f}) r1={r1:.3f} "
            f"c2=({cx2:.2f},{cy2:.2f}) r2={r2:.3f} d={d:.3f}"
        )
    a  = (r1**2 - r2**2 + d**2) / (2 * d)
    h  = math.sqrt(max(r1**2 - a**2, 0.0))
    mx = cx1 + a * dx / d
    my = cy1 + a * dy / d
    px =  h * dy / d
    py = -h * dx / d
    if want_upper:
        # pick the point with larger y; if tied, larger x
        if py < 0 or (py == 0 and px < 0):
            px, py = -px, -py
    else:
        if py > 0 or (py == 0 and px > 0):
            px, py = -px, -py
    return (mx + px, my + py)


def fk_solve(cfg, input_angles: dict) -> dict:
    """
    Closed-form forward kinematics.

    Chain
    -----
    O  → A   : driven by theta1 (length O→A)
    A,Q → B  : circle-circle intersection (lengths A→B, Q→B)
    B,Q → C  : Q→C rotated 90° CCW from Q→B direction (length Q→C),
                enforcing the B→Q→C = 90° constraint
    Q  → P1  : driven by theta_c (length Q→P1)
    C,P1→ P2 : circle-circle intersection (lengths C→P2, P1→P2)
    P1,P2→ D : extend P2→P1 by length P1→D
    """
    pts = {}

    # ── Fixed points ──────────────────────────────────────────────────
    for name, p in cfg["points"].items():
        if p.get("fixed", False):
            pts[name] = (float(p["x"]), float(p["y"]))

    ox, oy = pts["O"]
    qx, qy = pts["Q"]

    # ── Helper: look up edge length ───────────────────────────────────
    def L(a, b):
        v = _find_edge_length(cfg, a, b)
        if v is None:
            raise ValueError(f"No edge {a}↔{b} in config")
        return v

    # ── A : driven by theta1 ──────────────────────────────────────────
    theta1 = math.radians(input_angles.get("theta1", 45.0))
    l_oa   = L("O", "A")
    pts["A"] = (ox + l_oa * math.cos(theta1),
                oy + l_oa * math.sin(theta1))
    ax, ay = pts["A"]

    # ── B : intersection of circle(A, |AB|) and circle(Q, |QB|) ──────
    # The correct assembly has B above the O-Q line → want_upper=True
    pts["B"] = _circle_intersect(ax, ay, L("A", "B"),
                                  qx, qy, L("Q", "B"),
                                  want_upper=True)
    bx, by = pts["B"]

    # ── C : Q→C is Q→B rotated 90° CCW, scaled to |QC| ──────────────
    # Enforces the B→Q→C = 90° angle constraint analytically
    qb_dx, qb_dy = bx - qx, by - qy
    qb_len = math.hypot(qb_dx, qb_dy)
    l_qc   = L("Q", "C")
    # Rotate 90° CW: (dx, dy) → (-dy, dx)
    ux, uy = qb_dy / qb_len, -qb_dx / qb_len
    pts["C"] = (qx + ux * l_qc, qy + uy * l_qc)
    cx, cy = pts["C"]

    # ── P1 : driven by theta_c ────────────────────────────────────────
    theta_c = math.radians(input_angles.get("theta_c", -30.0))
    l_qp1   = L("Q", "P1")
    pts["P1"] = (qx + l_qp1 * math.cos(theta_c),
                 qy + l_qp1 * math.sin(theta_c))
    p1x, p1y = pts["P1"]

    # ── P2 : intersection of circle(C, |CP2|) and circle(P1, |P1P2|) ─
    # Correct assembly: P2 on the same side as the non-crossing branch.
    # The parallelogram normal (C-Q cross P1-Q) tells us which side.
    # We want P2 such that Q,P1,P2,C form a proper (non-butterfly) quad.
    # Empirically: want_upper=False matches the seeded solver assembly.
    pts["P2"] = _circle_intersect(cx, cy, L("C", "P2"),
                                   p1x, p1y, L("P1", "P2"),
                                   want_upper=True)
    p2x, p2y = pts["P2"]

    # ── D : extend P2→P1 by |P1D| ────────────────────────────────────
    ddx, ddy = p1x - p2x, p1y - p2y
    dmag = math.hypot(ddx, ddy)
    l_p1d = L("P1", "D")
    pts["D"] = (p1x + ddx / dmag * l_p1d,
                p1y + ddy / dmag * l_p1d)

    return pts

# ── Config ────────────────────────────────────────────────────────────

def load_config(path="linkage_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Solver ────────────────────────────────────────────────────────────

def _segments_intersect(p1, p2, p3, p4):
    def _cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return (((d1>0 and d2<0) or (d1<0 and d2>0)) and
            ((d3>0 and d4<0) or (d3<0 and d4>0)))

def solve(cfg, input_angles: dict) -> dict:
    sys_ = SolverSystem()
    wp   = sys_.create_2d_base()

    fixed_names = {n for n, p in cfg["points"].items() if p.get("fixed", False)}
    driven_map  = {inp["point"]: inp for inp in cfg.get("inputs", [])}

    # ── Seed P1 and P2 to the correct assembly branch ────────────────
    theta_c_inp = next(i for i in cfg["inputs"] if i["name"] == "theta_c")
    theta_c_deg = input_angles.get("theta_c", theta_c_inp["angle_deg"])
    theta_c_rad = math.radians(theta_c_deg)

    qx = cfg["points"]["Q"]["x"]
    qy = cfg["points"]["Q"]["y"]
    L_qp1 = _find_edge_length(cfg, "Q", "P1")

    # P1 is fully driven — compute exactly where it is
    p1x = qx + L_qp1 * math.cos(theta_c_rad)
    p1y = qy + L_qp1 * math.sin(theta_c_rad)
    cfg["points"]["P1"]["x"] = p1x
    cfg["points"]["P1"]["y"] = p1y

    # P2 seeded via parallelogram identity: P2 = P1 + (C - Q)
    # This picks the non-crossed assembly every time
    cx = cfg["points"]["C"].get("x", 0.0)
    cy = cfg["points"]["C"].get("y", 0.0)
    cfg["points"]["P2"]["x"] = p1x + (cx - qx)
    cfg["points"]["P2"]["y"] = p1y + (cy - qy)

    # ── 1. Add all points ─────────────────────────────────────────────
    handles = {}
    for name, p in cfg["points"].items():
        x = p.get("x", 0.0)
        y = p.get("y", 0.0)
        h = sys_.add_point_2d(float(x), float(y), wp)
        handles[name] = h
        if name in fixed_names:
            sys_.dragged(h, wp)

    # ── 2. Reference line ─────────────────────────────────────────────
    _ref_origin = sys_.add_point_2d(0.0, 0.0, wp)
    _ref_tip    = sys_.add_point_2d(1.0, 0.0, wp)
    sys_.dragged(_ref_origin, wp)
    sys_.dragged(_ref_tip, wp)
    _ref_line = sys_.add_line_2d(_ref_origin, _ref_tip, wp)

    # ── 3. Driven points ──────────────────────────────────────────────
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

    # ── 4. Distance constraints ───────────────────────────────────────
    line_handles = {}
    for edge in cfg["edges"]:
        a, b = edge["from"], edge["to"]
        # Skip P1→D edge — D is computed analytically, not by solver
        if {a, b} == {"P1", "D"}:
            continue
        a_pinned = a in fixed_names or a in driven_map
        b_pinned = b in fixed_names or b in driven_map
        line_handles[(a, b)] = sys_.add_line_2d(handles[a], handles[b], wp)
        if not (a_pinned and b_pinned):
            sys_.distance(handles[a], handles[b], edge["length"], wp)

    # ── 5. Angle constraints ──────────────────────────────────────────
    for ac in cfg.get("angle_constraints", []):
        angle_deg = float(ac["angle_deg"])
        if "edge" in ac:
            a, b = ac["edge"]
            line = _get_line(sys_, wp, line_handles, handles, a, b)
            pivot_h = handles[a]
            px, py  = sys_.params(pivot_h.params)
            tip_h   = sys_.add_point_2d(px + 1.0, py, wp)
            sys_.dragged(tip_h, wp)
            local_ref = sys_.add_line_2d(pivot_h, tip_h, wp)
            sys_.angle(local_ref, line, angle_deg, wp)
        elif "edge1" in ac and "edge2" in ac:
            a1, b1 = ac["edge1"]
            a2, b2 = ac["edge2"]
            line1 = _get_line(sys_, wp, line_handles, handles, a1, b1)
            line2 = _get_line(sys_, wp, line_handles, handles, a2, b2)
            sys_.angle(line1, line2, angle_deg, wp)
        else:
            raise ValueError(
                f"angle_constraint must have either 'edge' or 'edge1'+'edge2': {ac}"
            )

    # ── 6. Solve ──────────────────────────────────────────────────────
    result = sys_.solve()
    if result != ResultFlag.OKAY:
        raise RuntimeError(
            f"Solver returned {result.name}. Failed: {sys_.failures()}"
        )

    pts = {name: tuple(sys_.params(h.params)) for name, h in handles.items()}

    # ── 7. Compute D analytically — away from Q through P1 ───────────
    p1  = pts["P1"]
    p2  = pts["P2"]
    length_p1d = _find_edge_length(cfg, "P1", "D")
    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    mag    = math.hypot(dx, dy)
    pts["D"] = (p1[0] + dx/mag * length_p1d,
                p1[1] + dy/mag * length_p1d)

    return pts

def _find_edge_length(cfg, a, b):
    for edge in cfg["edges"]:
        if (edge["from"] == a and edge["to"] == b) or \
           (edge["from"] == b and edge["to"] == a):
            return edge["length"]
    return None


def _get_line(sys_, wp, line_handles, handles, a, b):
    """Return an existing line handle or create one on the fly."""
    if (a, b) in line_handles:
        return line_handles[(a, b)]
    if (b, a) in line_handles:
        return line_handles[(b, a)]
    # Create a temporary line entity for this pair
    return sys_.add_line_2d(handles[a], handles[b], wp)


# ── Plotting ──────────────────────────────────────────────────────────

EDGE_COLORS = [
    "#3B8BD4", "#D85A30", "#1D9E75", "#7F77DD",
    "#BA7517", "#A32D2D", "#0F6E56", "#993556",
]

# Fixed viewport in mm — adjust to fit your mechanism
VIEWPORT = {
    "xmin": -50,
    "xmax": 250,
    "ymin": -200,
    "ymax": 150,
}


def plot_linkage(ax, cfg, pts, title="Linkage — forward kinematics  (SolveSpace)"):
    ax.clear()

    fixed_names  = {n for n, p in cfg["points"].items() if p.get("fixed", False)}
    driven_names = {inp["point"] for inp in cfg.get("inputs", [])}
    angle_cons   = cfg.get("angle_constraints", [])

    # Edges
    for i, edge in enumerate(cfg["edges"]):
        a, b   = edge["from"], edge["to"]
        pa, pb = pts[a], pts[b]
        color  = EDGE_COLORS[i % len(EDGE_COLORS)]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color=color, lw=2.5, solid_capstyle="round", zorder=2)
        mx, my = (pa[0]+pb[0])/2, (pa[1]+pb[1])/2
        ax.text(mx, my, f"{edge['length']:.2f}", fontsize=6.5,
                ha="center", va="bottom", color=color, alpha=0.85,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

    # Angle constraint arcs (visual indicator)
    for ac in angle_cons:
        angle_deg = float(ac["angle_deg"])
        if "edge" in ac:
            a, b  = ac["edge"]
            pa    = pts[a]
            r     = 8.0
            # draw a small arc from horizontal to the constrained edge
            edge_angle = math.degrees(math.atan2(pts[b][1]-pa[1], pts[b][0]-pa[0]))
            theta1, theta2 = (0, edge_angle) if edge_angle >= 0 else (edge_angle, 0)
            arc = mpatches.Arc(pa, 2*r, 2*r, angle=0,
                               theta1=theta1, theta2=theta2,
                               color="#7F77DD", lw=1.2, ls="--", zorder=3)
            ax.add_patch(arc)
            ax.text(pa[0]+r+2, pa[1]+2, f"{angle_deg:.1f}°",
                    fontsize=6, color="#7F77DD")

    # Joints
    for name, pos in pts.items():
        if name in fixed_names:
            marker, size, color, zo = "s", 90, "#333", 5
            ax.plot([pos[0]-6, pos[0]+6], [pos[1]-6, pos[1]-6],
                    color="#888", lw=1, zorder=3)
        elif name in driven_names:
            marker, size, color, zo = "D", 70, "#D85A30", 4
        else:
            marker, size, color, zo = "o", 70, "#3B8BD4", 4

        ax.scatter(pos[0], pos[1], marker=marker, s=size,
                   color=color, zorder=zo,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(name, pos, textcoords="offset points",
                    xytext=(5, 5), fontsize=8.5, fontweight="500",
                    color="#222")

    # Coordinate readout
    lines = [f"{n:>4s} = ({p[0]:8.3f}, {p[1]:8.3f}) mm"
             for n, p in sorted(pts.items())]
    ax.text(0.01, 0.99, "\n".join(lines),
            transform=ax.transAxes, va="top", fontsize=7.5,
            fontfamily="monospace", color="#333",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", alpha=0.88, edgecolor="#ccc"))

    # Legend
    legend = [
        mpatches.Patch(color="#333",    label="Fixed pivot"),
        mpatches.Patch(color="#D85A30", label="Driven joint"),
        mpatches.Patch(color="#3B8BD4", label="Free joint"),
        mpatches.Patch(color="#7F77DD", label="Angle constraint"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.85)

    ax.set_xlim(VIEWPORT["xmin"], VIEWPORT["xmax"])
    ax.set_ylim(VIEWPORT["ymin"], VIEWPORT["ymax"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e8e8e8", lw=0.5)
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_title("Linkage — forward kinematics  (SolveSpace)", fontsize=10)
    ax.figure.canvas.draw_idle()


# ── GUI ───────────────────────────────────────────────────────────────

def run_gui(cfg):
    inputs   = cfg.get("inputs", [])
    n_inputs = len(inputs)

    fig = plt.figure(figsize=(16, 7), dpi=100)   # wider for side-by-side
    fig.patch.set_facecolor("#f8f8f6")

    slider_strip_h = 0.06 + n_inputs * 0.07
    plot_bottom    = slider_strip_h + 0.05

    ax_sv = fig.add_axes([0.05, plot_bottom, 0.43, 0.93 - plot_bottom])  # left: solvespace
    ax_fk = fig.add_axes([0.54, plot_bottom, 0.43, 0.93 - plot_bottom])  # right: FK

    slider_objs = {}
    for i, inp in enumerate(inputs):
        bottom = 0.035 + i * 0.07
        sax = fig.add_axes([0.10, bottom, 0.75, 0.04])
        sax.set_facecolor("#f0f0f0")
        sl = Slider(sax, f"{inp['name']}  (°)", -180, 180,
                    valinit=inp["angle_deg"], valfmt="%.1f", color="#3B8BD4")
        sl.label.set_fontsize(9)
        sl.valtext.set_fontsize(9)
        slider_objs[inp["name"]] = sl

    def get_angles():
        return {name: sl.val for name, sl in slider_objs.items()}

    def update(_):
        angles = get_angles()

        # ── Left: SolveSpace ─────────────────────────────────────────
        try:
            pts_sv = solve(cfg, angles)
            plot_linkage(ax_sv, cfg, pts_sv, title="SolveSpace (constrained)")
        except Exception as exc:
            ax_sv.clear()
            ax_sv.text(0.5, 0.5, f"Solver error:\n{exc}",
                       transform=ax_sv.transAxes, ha="center", va="center",
                       fontsize=9, color="red")
            fig.canvas.draw_idle()

        # ── Right: Forward kinematics ─────────────────────────────────
        try:
            pts_fk = fk_solve(cfg, angles)
            plot_linkage(ax_fk, cfg, pts_fk, title="Forward kinematics (analytical)")
        except Exception as exc:
            ax_fk.clear()
            ax_fk.text(0.5, 0.5, f"FK error:\n{exc}",
                       transform=ax_fk.transAxes, ha="center", va="center",
                       fontsize=9, color="red")
            fig.canvas.draw_idle()

    for sl in slider_objs.values():
        sl.on_changed(update)

    btn_ax = fig.add_axes([0.88, 0.005, 0.10, 0.04])
    btn = Button(btn_ax, "Reset", color="#ebebeb", hovercolor="#ddd")
    def reset(_):
        for sl in slider_objs.values():
            sl.reset()
        update(None)
    btn.on_clicked(reset)

    update(None)

    plt.suptitle("Edit VIEWPORT dict in linkage_solver.py to adjust canvas bounds",
                 fontsize=7, color="#aaa", y=0.998)
    fig.canvas.manager.set_window_title("Linkage Solver")
    try:
        win = fig.canvas.manager.window
        win.setFixedSize(win.size())
    except Exception:
        pass

    plt.show()

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "linkage_config.yaml"
    cfg = load_config(config_path)
    run_gui(cfg)
