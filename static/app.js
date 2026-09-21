/**
 * app.js — quadruped leg test utility.
 *
 *   - Loads config + workspace + legs on startup
 *   - Draws all four legs in a 2x2 canvas (FL FR / BL BR)
 *   - Click a quadrant -> POST /leg/ik for that leg, then /leg/fk to redraw,
 *     and /leg/send if serial connected and not walking
 *   - Hold ArrowUp/Down (or W/S), or use the Walk buttons, to trot fwd/back
 *   - Gait slider live-tunes cycle time
 *   - "Record" toggles gait-path mode: clicks append a canonical-frame
 *     waypoint instead of driving the leg; "Save Path" persists the closed
 *     loop to the gait block and reloads config. WalkController linearly
 *     interpolates between waypoints, equal time per segment, looping.
 */

// ── Layout of the 2x2 grid ─────────────────────────────────────────────
const CELLS = [
  { leg: "FL", col: 0, row: 0 },
  { leg: "FR", col: 1, row: 0 },
  { leg: "BL", col: 0, row: 1 },
  { leg: "BR", col: 1, row: 1 },
];

// ── State ──────────────────────────────────────────────────────────────
const state = {
  workspace: [],
  legs:      {},     // name -> {side, channels}
  legView:   {},     // name -> {pts, target, ikValid, angles}
  selected:  null,
  serialConn: false,
  walking:    false,
  walkDir:    0,
  walkTurn:   0,      // +1 rotating right, -1 rotating left
  pollTimer:  null,
  recording:  false,
  waypoints:  [],     // [{x, y, valid}] canonical-frame gait path, in click order
};

// ── DOM refs ───────────────────────────────────────────────────────────
const canvas     = document.getElementById("canvas");
const readoutEl  = document.getElementById("readout");
const statusMsg  = document.getElementById("status-msg");
const btnReload  = document.getElementById("btn-reload");

const hwPortSelect = document.getElementById("hw-port-select");
const hwRefresh    = document.getElementById("hw-refresh");
const hwConnect    = document.getElementById("hw-connect");
const hwStatus     = document.getElementById("hw-serial-status");

const walkBack   = document.getElementById("walk-back");
const walkStopEl = document.getElementById("walk-stop");
const walkFwd    = document.getElementById("walk-fwd");
const walkLeft   = document.getElementById("walk-left");
const walkRight  = document.getElementById("walk-right");
const walkStatus = document.getElementById("walk-status");

const gCycle = document.getElementById("g-cycle");

const gpRecord = document.getElementById("gp-record");
const gpUndo   = document.getElementById("gp-undo");
const gpClear  = document.getElementById("gp-clear");
const gpSave   = document.getElementById("gp-save");
const gpList   = document.getElementById("gait-path-list");

// ── Canvas sizing ──────────────────────────────────────────────────────
function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width  = Math.max(320, Math.floor(rect.width));
  canvas.height = Math.max(320, Math.floor(rect.height));
}

function cellRect(cell) {
  const w = canvas.width / 2;
  const h = canvas.height / 2;
  return { x: cell.col * w, y: cell.row * h, w, h };
}

function cellAt(cx, cy) {
  const col = cx < canvas.width / 2 ? 0 : 1;
  const row = cy < canvas.height / 2 ? 0 : 1;
  return CELLS.find(c => c.col === col && c.row === row);
}

// ── Status helper ──────────────────────────────────────────────────────
function setStatus(msg, cls = "") {
  statusMsg.textContent = msg;
  statusMsg.className = "status-msg " + cls;
  if (cls === "ok") setTimeout(() => {
    statusMsg.textContent = ""; statusMsg.className = "status-msg";
  }, 2500);
}

// ── Render ─────────────────────────────────────────────────────────────
function render() {
  const cfg = Config.get();
  const vp  = Config.viewport();
  Renderer.clear(canvas);
  for (const cell of CELLS) {
    const v = state.legView[cell.leg] || {};
    const side = (state.legs[cell.leg] || {}).side;
    Renderer.drawLeg(canvas, cellRect(cell), {
      cfg,
      pts:        v.pts || null,
      workspace:  state.workspace,
      target:     v.target || null,
      ikValid:    v.ikValid ?? null,
      endEffector: Config.endEffector(),
      vp,
      mirror:     side === "right",
      label:      `${cell.leg}  (${side || "?"})`,
      selected:   state.selected === cell.leg,
      gaitPath:   state.waypoints,
    });
  }
  renderReadout();
}

function renderWaypointList() {
  gpList.innerHTML = "";
  state.waypoints.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "gp-item" + (p.valid === false ? " invalid" : "");
    const label = document.createElement("span");
    label.textContent = `${i + 1}: (${p.x.toFixed(1)}, ${p.y.toFixed(1)})${p.valid === false ? " ✗" : ""}`;
    const del = document.createElement("button");
    del.textContent = "×";
    del.title = "Remove point";
    del.addEventListener("click", () => {
      state.waypoints.splice(i, 1);
      renderWaypointList();
      render();
    });
    row.appendChild(label);
    row.appendChild(del);
    gpList.appendChild(row);
  });
}

function renderReadout() {
  const lines = CELLS.map(({ leg }) => {
    const v = state.legView[leg];
    if (!v || !v.angles) return `${leg.padEnd(3)}  —`;
    const { theta1, theta_c } = v.angles;
    const flag = v.ikValid === false ? " ✗ unreachable"
               : v.ikValid === true  ? " ✓" : "";
    return `${leg.padEnd(3)}  θ1=${theta1.toFixed(1).padStart(7)}°  ` +
           `θc=${theta_c.toFixed(1).padStart(7)}°${flag}`;
  });
  readoutEl.textContent = lines.join("\n");
}

// ── API ────────────────────────────────────────────────────────────────
async function jpost(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.status);
  return res.json();
}

async function legFk(leg, theta1, theta_c) {
  return jpost("/leg/fk", { leg, theta1, theta_c });
}
async function legIk(leg, x, y) {
  return jpost("/leg/ik", { leg, x, y });
}

// ── Populate one leg's view from angles ────────────────────────────────
async function refreshLeg(leg, theta1, theta_c) {
  const fk = await legFk(leg, theta1, theta_c);
  state.legView[leg] = {
    ...(state.legView[leg] || {}),
    pts: fk.points,
    angles: { theta1, theta_c },
  };
}

// ── Quadrant click -> IK ──────────────────────────────────────────────
canvas.addEventListener("click", async (e) => {
  if (state.walking) { setStatus("Stop walking to set a target", "error"); return; }
  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (canvas.width  / rect.width);
  const cy = (e.clientY - rect.top)  * (canvas.height / rect.height);

  const cell = cellAt(cx, cy);
  if (!cell) return;
  const leg  = cell.leg;
  const side = (state.legs[leg] || {}).side;

  const t = Renderer.makeTransform(Config.viewport(), cellRect(cell), side === "right");
  const [lx, ly] = t.toLinkage(cx, cy);

  if (state.recording) {
    const wp = { x: lx, y: ly, valid: null };
    state.waypoints.push(wp);
    renderWaypointList();
    render();
    try {
      const r = await legIk(leg, lx, ly);
      wp.valid = r.valid;
    } catch (err) {
      wp.valid = false;
    }
    renderWaypointList();
    render();
    return;
  }

  state.selected = leg;
  state.legView[leg] = { ...(state.legView[leg] || {}), target: { x: lx, y: ly }, ikValid: null };
  render();

  try {
    const r = await legIk(leg, lx, ly);
    state.legView[leg].ikValid = r.valid;
    state.legView[leg].angles  = { theta1: r.theta1, theta_c: r.theta_c };
    await refreshLeg(leg, r.theta1, r.theta_c);
    render();

    if (r.valid && state.serialConn) {
      await jpost("/leg/send", { leg, theta1: r.theta1, theta_c: r.theta_c });
      setStatus(`${leg} sent`, "ok");
    } else if (!r.valid) {
      setStatus(`${leg}: unreachable (${r.error_mm.toFixed(1)} mm)`, "error");
    }
  } catch (err) {
    setStatus(`${leg} IK error: ${err.message}`, "error");
  }
});

// ── Reload config ─────────────────────────────────────────────────────
btnReload.addEventListener("click", async () => {
  btnReload.disabled = true;
  try {
    await Config.reload();
    await bootLegs();
    setStatus("Config reloaded", "ok");
  } catch (e) {
    setStatus("Reload failed: " + e.message, "error");
  } finally {
    btnReload.disabled = false;
  }
});

// ── Serial ────────────────────────────────────────────────────────────
function setSerialStatus(connected, portName = "", errorMsg = "") {
  state.serialConn = connected;
  if (errorMsg) {
    hwStatus.textContent = "⬤ " + errorMsg;
    hwStatus.className = "hw-status error";
  } else if (connected) {
    hwStatus.textContent = `⬤ Connected — ${portName}`;
    hwStatus.className = "hw-status connected";
    hwConnect.textContent = "Disconnect";
    hwConnect.classList.add("connected");
  } else {
    hwStatus.textContent = "⬤ Disconnected";
    hwStatus.className = "hw-status disconnected";
    hwConnect.textContent = "Connect";
    hwConnect.classList.remove("connected");
  }
}

async function refreshPorts() {
  hwRefresh.disabled = true;
  try {
    const ports = await fetch("/serial/ports").then(r => r.json());
    const prev = hwPortSelect.value;
    hwPortSelect.innerHTML = "<option value=\"\">— select port —</option>";
    for (const { port, description } of ports) {
      const opt = document.createElement("option");
      opt.value = port;
      opt.textContent = description === port ? port : `${port}  —  ${description}`;
      hwPortSelect.appendChild(opt);
    }
    if (prev && [...hwPortSelect.options].some(o => o.value === prev)) hwPortSelect.value = prev;
  } catch (e) {
    setStatus("Port list error: " + e.message, "error");
  } finally {
    hwRefresh.disabled = false;
  }
}

hwRefresh.addEventListener("click", refreshPorts);

hwConnect.addEventListener("click", async () => {
  hwConnect.disabled = true;
  try {
    if (state.serialConn) {
      await fetch("/serial/disconnect", { method: "POST" });
      setSerialStatus(false);
    } else {
      const port = hwPortSelect.value;
      if (!port) { setStatus("Select a port first", "error"); return; }
      try {
        const data = await jpost("/serial/connect", { port });
        setSerialStatus(true, data.port);
      } catch (err) {
        setSerialStatus(false, "", err.message || "Connection failed");
      }
    }
  } finally {
    hwConnect.disabled = false;
  }
});

// ── Walking ───────────────────────────────────────────────────────────
async function walkStart(dir, turn = 0) {
  if (state.walkDir === dir && state.walkTurn === turn) return;
  try {
    await jpost("/walk/start", { direction: dir, turn });
    state.walking = dir !== 0 || turn !== 0;
    state.walkDir = dir;
    state.walkTurn = turn;
    updateWalkUI();
    startPoll();
  } catch (e) {
    setStatus("Walk error: " + e.message, "error");
  }
}

async function walkStop() {
  try {
    await jpost("/walk/stop", {});
  } catch (e) { /* ignore */ }
  state.walking = false;
  state.walkDir = 0;
  state.walkTurn = 0;
  updateWalkUI();
  stopPoll();
}

function updateWalkUI() {
  walkFwd.classList.toggle("active", state.walkDir === 1);
  walkBack.classList.toggle("active", state.walkDir === -1);
  walkLeft.classList.toggle("active", state.walkTurn === -1);
  walkRight.classList.toggle("active", state.walkTurn === 1);
  walkStatus.textContent = !state.walking ? "idle"
    : state.walkTurn ? (state.walkTurn === 1 ? "turning ⟳" : "⟲ turning")
    : (state.walkDir === 1 ? "walking ▶" : "◀ walking");
}

function startPoll() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(async () => {
    try {
      const s = await fetch("/walk/status").then(r => r.json());
      walkStatus.textContent = s.walking
        ? `${s.turn ? (s.turn === 1 ? "⟳" : "⟲") : (s.direction === 1 ? "▶" : "◀")} phase ${s.phase.toFixed(2)}`
        : "idle";
      if (!s.walking) { state.walking = false; state.walkDir = 0; state.walkTurn = 0; updateWalkUI(); stopPoll(); }
    } catch (e) { /* ignore */ }
  }, 200);
}
function stopPoll() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

walkFwd.addEventListener("click", () => walkStart(1));
walkBack.addEventListener("click", () => walkStart(-1));
walkLeft.addEventListener("click", () => walkStart(0, -1));
walkRight.addEventListener("click", () => walkStart(0, 1));
walkStopEl.addEventListener("click", () => walkStop());

// Hold-to-walk keyboard
window.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.key === "ArrowUp" || e.key === "w") { e.preventDefault(); walkStart(1); }
  else if (e.key === "ArrowDown" || e.key === "s") { e.preventDefault(); walkStart(-1); }
  else if (e.key === "ArrowLeft" || e.key === "a") { e.preventDefault(); walkStart(0, -1); }
  else if (e.key === "ArrowRight" || e.key === "d") { e.preventDefault(); walkStart(0, 1); }
});
window.addEventListener("keyup", (e) => {
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "w", "s", "a", "d"].includes(e.key)) walkStop();
});
window.addEventListener("blur", () => { if (state.walking) walkStop(); });

// ── Gait sliders ──────────────────────────────────────────────────────
function bindGaitSlider(el, valEl, key, fmt) {
  el.addEventListener("input", () => { valEl.textContent = fmt(el.value); });
  el.addEventListener("change", async () => {
    try { await jpost("/walk/params", { [key]: parseFloat(el.value) }); }
    catch (e) { setStatus("Gait param error: " + e.message, "error"); }
  });
}
bindGaitSlider(gCycle, document.getElementById("g-cycle-val"), "cycle_time_s",
               v => parseFloat(v).toFixed(2));

// ── Gait path recording ─────────────────────────────────────────────────
gpRecord.addEventListener("click", () => {
  state.recording = !state.recording;
  gpRecord.classList.toggle("active", state.recording);
  gpRecord.textContent = state.recording ? "● Recording…" : "● Record";
  setStatus(state.recording ? "Recording: click a quadrant to add a gait point" : "");
});

gpUndo.addEventListener("click", () => {
  state.waypoints.pop();
  renderWaypointList();
  render();
});

gpClear.addEventListener("click", () => {
  state.waypoints = [];
  renderWaypointList();
  render();
});

gpSave.addEventListener("click", async () => {
  if (state.waypoints.length < 2) { setStatus("Need at least 2 points to save a path", "error"); return; }
  gpSave.disabled = true;
  try {
    await jpost("/gait/waypoints", { waypoints: state.waypoints.map(p => [p.x, p.y]) });
    setStatus("Gait path saved", "ok");
  } catch (e) {
    setStatus("Save failed: " + e.message, "error");
  } finally {
    gpSave.disabled = false;
  }
});

// ── Boot ──────────────────────────────────────────────────────────────
async function bootLegs() {
  const legs = Config.legs();
  state.legs = Object.fromEntries(legs.map(l => [l.name, l]));
  state.workspace = await fetch("/workspace").then(r => r.json());

  const g = Config.gait();
  gCycle.value = g.cycle_time_s; document.getElementById("g-cycle-val").textContent = g.cycle_time_s.toFixed(2);

  state.waypoints = g.waypoints.map(([x, y]) => ({ x, y, valid: null }));
  renderWaypointList();

  // Seed each leg at the parked pose (first gait waypoint, or a fallback).
  const [seedX, seedY] = g.waypoints.length ? g.waypoints[0] : [150, -120];
  for (const { name } of legs) {
    try {
      const r = await legIk(name, seedX, seedY);
      state.legView[name] = { ikValid: r.valid, angles: { theta1: r.theta1, theta_c: r.theta_c } };
      await refreshLeg(name, r.theta1, r.theta_c);
    } catch (e) {
      state.legView[name] = {};
    }
  }
  render();
}

(async () => {
  try {
    await Config.load();
    resizeCanvas();
    await bootLegs();

    await refreshPorts();
    const s = await fetch("/serial/status").then(r => r.json());
    if (s.connected) setSerialStatus(true, s.port);

    setStatus("Ready", "ok");
  } catch (e) {
    setStatus("Boot error: " + e.message, "error");
    console.error(e);
  }
})();

window.addEventListener("resize", () => { resizeCanvas(); render(); });
