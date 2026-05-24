"""
server.py — FastAPI backend for the linkage test utility.

Endpoints
---------
GET  /config           returns parsed config as JSON
POST /config/reload    re-reads YAML from disk, recomputes workspace
POST /fk               {theta1, theta_c} -> all named points
POST /ik               {x, y}            -> {theta1, theta_c, valid, d_actual}
GET  /workspace        precomputed reachable D positions
GET  /                 serves index.html
"""

import math
import pathlib
import logging

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from linkage_solver import fk_solve, _find_edge_length
from ik_solver import ik_solve
from workspace import compute_workspace

# ── Paths ──────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).parent
CONFIG_PATH = ROOT / "linkage_config.yaml"
STATIC_DIR  = ROOT / "static"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server")

app = FastAPI()

# ── State ──────────────────────────────────────────────────────────────
_cfg       = {}
_workspace = []   # list of {x, y} dicts


def _load_config():
    global _cfg, _workspace
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
    log.info("Config loaded — recomputing workspace…")
    _workspace = compute_workspace(_cfg)
    log.info(f"Workspace: {len(_workspace)} points")


_load_config()


# ── Request / Response models ──────────────────────────────────────────
class FKRequest(BaseModel):
    theta1:  float
    theta_c: float

class IKRequest(BaseModel):
    x: float
    y: float


# ── Routes ────────────────────────────────────────────────────────────

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


@app.post("/fk")
def forward_kinematics(req: FKRequest):
    try:
        pts = fk_solve(_cfg, {"theta1": req.theta1, "theta_c": req.theta_c})
        return {name: {"x": round(xy[0], 4), "y": round(xy[1], 4)}
                for name, xy in pts.items()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ik")
def inverse_kinematics(req: IKRequest):
    try:
        result = ik_solve(_cfg, req.x, req.y)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/workspace")
def get_workspace():
    return _workspace


# ── Static files + SPA fallback ───────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
