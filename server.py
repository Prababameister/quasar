"""
server.py — FastAPI backend for the quadruped leg test utility.

Endpoints
---------
GET  /config            parsed config as JSON
POST /config/reload     re-read YAML, recompute workspace, refresh gait params
GET  /workspace         precomputed reachable D positions (canonical left-leg frame)
GET  /legs              leg names, sides, channels

POST /leg/fk            {leg, theta1, theta_c}     -> points for that leg
POST /leg/ik            {leg, x, y}                -> IK solution for that leg
POST /leg/send          {leg, theta1, theta_c}    -> command one leg over serial

POST /walk/start        {direction, turn}   (direction +1 fwd, -1 back, 0 hold;
                                       turn +1 right, -1 left, overrides direction)
POST /walk/direction    {direction, turn}
POST /walk/stop
GET  /walk/status
POST /walk/params       {cycle_time_s}

GET  /gait/waypoints    current foot-path loop (canonical left-leg frame)
POST /gait/waypoints    {waypoints: [[x, y], ...]} -> replace loop, persist to YAML, reload

GET  /serial/ports | POST /serial/connect | POST /serial/disconnect | GET /serial/status

GET  /                   serves index.html
"""

import logging
import pathlib
import re

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from workspace import compute_workspace
from serial_manager import get_manager
from leg_kinematics import solve_leg, fk_leg, leg_names, hip_fixed_deg
from gait import WalkController

# ── Paths ──────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).parent
CONFIG_PATH = ROOT / "linkage_config.yaml"
STATIC_DIR  = ROOT / "static"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server")

app = FastAPI()


@app.middleware("http")
async def no_cache_static(request, call_next):
    """
    This is a local dev tool whose static/*.js and index.html change often
    across sessions; without an explicit Cache-Control header browsers may
    heuristically cache them past an ordinary reload, silently running a
    stale script against a fresh page (mismatched DOM -> broken JS). Force
    revalidation on every request instead.
    """
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

_WAYPOINTS_KEY_RE = re.compile(r'^(?P<indent>[ \t]*)waypoints:(?P<comment>.*)$')


def _replace_waypoints_in_yaml(text: str, waypoints: list) -> str:
    """
    Rewrite just the gait.waypoints block-sequence in-place, leaving every
    other line (and all comments/formatting) untouched. Raises ValueError if
    the `waypoints:` key isn't found.
    """
    lines = text.splitlines(keepends=True)
    for i, raw in enumerate(lines):
        m = _WAYPOINTS_KEY_RE.match(raw.rstrip("\n"))
        if not m:
            continue
        key_indent = m.group("indent")
        item_indent = key_indent + "  "

        j = i + 1
        while j < len(lines):
            stripped = lines[j].rstrip("\n")
            if stripped.strip() == "":
                break
            indent = len(stripped) - len(stripped.lstrip(" "))
            if indent <= len(key_indent) or not stripped.lstrip().startswith("-"):
                break
            j += 1

        key_line = raw if raw.endswith("\n") else raw + "\n"
        item_lines = [f"{item_indent}- [{x}, {y}]\n" for x, y in waypoints]
        return "".join(lines[:i] + [key_line] + item_lines + lines[j:])

    raise ValueError("gait.waypoints key not found in linkage_config.yaml")


def _save_waypoints(waypoints: list) -> None:
    text = CONFIG_PATH.read_text()
    new_text = _replace_waypoints_in_yaml(text, waypoints)
    yaml.safe_load(new_text)  # validate before committing to disk
    CONFIG_PATH.write_text(new_text)

# ── State ──────────────────────────────────────────────────────────────
_cfg       = {}
_workspace = []
_walker: WalkController | None = None


def _load_config():
    global _cfg, _workspace, _walker
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
    log.info("Config loaded — recomputing workspace…")
    _workspace = compute_workspace(_cfg)
    log.info(f"Workspace: {len(_workspace)} points")
    if _walker is None:
        _walker = WalkController(_cfg, get_manager())
    else:
        _walker.refresh_config(_cfg)


_load_config()


# ── Request models ─────────────────────────────────────────────────────
class LegFKRequest(BaseModel):
    leg:     str
    theta1:  float
    theta_c: float

class LegIKRequest(BaseModel):
    leg: str
    x:   float
    y:   float

class LegSendRequest(BaseModel):
    leg:     str
    theta1:  float
    theta_c: float

class DirectionRequest(BaseModel):
    direction: int = 0
    turn: int = 0

class GaitParamsRequest(BaseModel):
    cycle_time_s: float | None = None

class WaypointsRequest(BaseModel):
    waypoints: list[tuple[float, float]]

class SerialConnectRequest(BaseModel):
    port: str
    baud: int = 115200


# ── Config / workspace ────────────────────────────────────────────────
@app.get("/config")
def get_config():
    return JSONResponse(_cfg)


@app.post("/config/reload")
def reload_config():
    try:
        _load_config()
        return {"status": "ok", "workspace_points": len(_workspace)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workspace")
def get_workspace():
    return _workspace


@app.get("/legs")
def get_legs():
    return [
        {"name": name, "side": lc["side"], "channels": lc["channels"]}
        for name, lc in _cfg.get("legs", {}).items()
    ]


# ── Per-leg kinematics ────────────────────────────────────────────────
@app.post("/leg/fk")
def leg_forward_kinematics(req: LegFKRequest):
    try:
        return fk_leg(_cfg, req.leg, req.theta1, req.theta_c)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/leg/ik")
def leg_inverse_kinematics(req: LegIKRequest):
    try:
        return solve_leg(_cfg, req.leg, req.x, req.y)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/leg/send")
def leg_send(req: LegSendRequest):
    if _walker and _walker.status()["walking"]:
        raise HTTPException(status_code=409, detail="Walking — stop before manual send")
    mgr = get_manager()
    try:
        mgr.send_leg(_cfg, req.leg, req.theta1, req.theta_c, hip_fixed_deg(_cfg))
        return {"status": "sent", "leg": req.leg}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Walking ───────────────────────────────────────────────────────────
@app.post("/walk/start")
def walk_start(req: DirectionRequest):
    try:
        _walker.start(req.direction, req.turn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _walker.status()


@app.post("/walk/direction")
def walk_direction(req: DirectionRequest):
    try:
        _walker.set_direction(req.direction, req.turn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _walker.status()


@app.post("/walk/stop")
def walk_stop():
    _walker.stop()
    return _walker.status()


@app.get("/walk/status")
def walk_status():
    return {**_walker.status(), "params": _walker.params()}


@app.post("/walk/params")
def walk_params(req: GaitParamsRequest):
    return _walker.set_params(**req.model_dump())


# ── Gait path ─────────────────────────────────────────────────────────
@app.get("/gait/waypoints")
def gait_waypoints():
    return _cfg.get("gait", {}).get("waypoints", [])


@app.post("/gait/waypoints")
def set_gait_waypoints(req: WaypointsRequest):
    if _walker and _walker.status()["walking"]:
        raise HTTPException(status_code=409, detail="Walking — stop before editing the gait path")
    waypoints = [[round(x, 3), round(y, 3)] for x, y in req.waypoints]
    try:
        _save_waypoints(waypoints)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save waypoints: {e}")
    _load_config()
    return {"status": "saved", "waypoints": waypoints}


# ── Serial ────────────────────────────────────────────────────────────
@app.get("/serial/ports")
def serial_ports():
    return get_manager().list_ports()


@app.post("/serial/connect")
def serial_connect(req: SerialConnectRequest):
    mgr = get_manager()
    try:
        mgr.connect(req.port, req.baud)
        return {"status": "connected", "port": req.port, "baud": req.baud}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/serial/disconnect")
def serial_disconnect():
    if _walker:
        _walker.stop()
    get_manager().disconnect()
    return {"status": "disconnected"}


@app.get("/serial/status")
def serial_status():
    mgr = get_manager()
    return {"connected": mgr.connected, "port": mgr.port_name}


# ── Static files ──────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8121, reload=False)
