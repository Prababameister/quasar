/**
 * app.js — main application logic.
 *
 * Responsibilities:
 *   - Load config + workspace on startup
 *   - Build sliders from config inputs
 *   - On slider release: POST /fk -> render FK canvas
 *   - On IK canvas click: POST /ik -> render IK canvas
 *   - Reload config button
 */

// ── State ──────────────────────────────────────────────────────────────
const state = {
  fkPts:       null,   // {name: {x,y}} from last /fk call
  ikPts:       null,   // {name: {x,y}} from last /ik call
  workspace:   [],     // [{x,y}]
  angles:      {},     // {theta1: deg, theta_c: deg, ...}
  ikTarget:    null,   // {x,y} in linkage mm coords
  ikValid:     null,   // bool | null
  ikAngles:    null,   // {theta1, theta_c} from last IK solve
};

// ── DOM refs ───────────────────────────────────────────────────────────
const canvasFk   = document.getElementById("canvas-fk");
const canvasIk   = document.getElementById("canvas-ik");
const slidersDiv = document.getElementById("sliders");
const coordsFk   = document.getElementById("coords-fk");
const coordsIk   = document.getElementById("coords-ik");
const ikStatusEl = document.getElementById("ik-status");
const statusMsg  = document.getElementById("status-msg");
const btnReload  = document.getElementById("btn-reload");

// ── Canvas sizing (fill parent flex cell) ─────────────────────────────
function resizeCanvas(canvas) {
  const rect = canvas.parentElement.getBoundingClientRect();
  // Canvas fills panel minus header, sliders, coords (roughly)
  canvas.width  = Math.floor(rect.width - 32);
  canvas.height = Math.floor(rect.height - 32);
}

function resizeAll() {
  resizeCanvas(canvasFk);
  resizeCanvas(canvasIk);
  renderFk();
  renderIk();
}

window.addEventListener("resize", resizeAll);

// ── Status helper ──────────────────────────────────────────────────────
function setStatus(msg, cls = "") {
  statusMsg.textContent  = msg;
  statusMsg.className    = "status-msg " + cls;
  if (cls === "ok") setTimeout(() => { statusMsg.textContent = ""; statusMsg.className = "status-msg"; }, 3000);
}

// ── Coords display ─────────────────────────────────────────────────────
function renderCoords(el, pts) {
  if (!pts) { el.textContent = ""; return; }
  el.textContent = Object.entries(pts)
    .sort(([a],[b]) => a.localeCompare(b))
    .map(([n, p]) => `${n.padEnd(4)} = (${p.x.toFixed(3).padStart(8)}, ${p.y.toFixed(3).padStart(8)}) mm`)
    .join("\n");
}

// ── Render helpers ─────────────────────────────────────────────────────
function renderFk() {
  const cfg = Config.get();
  Renderer.draw(canvasFk, {
    cfg,
    pts:         state.fkPts,
    workspace:   state.workspace,
    target:      null,
    ikValid:     null,
    endEffector: Config.endEffector(),
    vp:          Config.viewport(),
  });
  renderCoords(coordsFk, state.fkPts);
}

function renderIk() {
  const cfg = Config.get();
  Renderer.draw(canvasIk, {
    cfg,
    pts:         state.ikPts,
    workspace:   state.workspace,
    target:      state.ikTarget,
    ikValid:     state.ikValid,
    endEffector: Config.endEffector(),
    vp:          Config.viewport(),
  });
  renderCoords(coordsIk, state.ikPts);
}

// ── API calls ──────────────────────────────────────────────────────────
async function callFk(angles) {
  const res  = await fetch("/fk", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(angles),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function callIk(x, y) {
  const res = await fetch("/ik", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ x, y }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchWorkspace() {
  const res = await fetch("/workspace");
  if (!res.ok) throw new Error("workspace fetch failed");
  return res.json();
}

// ── Sliders ────────────────────────────────────────────────────────────
function buildSliders() {
  slidersDiv.innerHTML = "";
  for (const inp of Config.inputs()) {
    state.angles[inp.name] = inp.angle_deg;

    const row   = document.createElement("div");
    row.className = "slider-row";

    const lbl   = document.createElement("label");
    lbl.textContent = `${inp.name} (°)`;

    const slider = document.createElement("input");
    slider.type  = "range";
    slider.min   = inp.min_deg;
    slider.max   = inp.max_deg;
    slider.step  = "0.1";
    slider.value = inp.angle_deg;

    const val = document.createElement("span");
    val.className   = "slider-val";
    val.textContent = inp.angle_deg.toFixed(1) + "°";

    // Update display live, but only call /fk on release
    slider.addEventListener("input", () => {
      state.angles[inp.name] = parseFloat(slider.value);
      val.textContent = parseFloat(slider.value).toFixed(1) + "°";
    });

    slider.addEventListener("change", async () => {
      try {
        state.fkPts = await callFk(state.angles);
        renderFk();
      } catch (e) {
        setStatus("FK error: " + e.message, "error");
      }
    });

    row.append(lbl, slider, val);
    slidersDiv.appendChild(row);
  }
}

// ── IK canvas click ────────────────────────────────────────────────────
canvasIk.addEventListener("click", async (e) => {
  const rect = canvasIk.getBoundingClientRect();
  const cx   = (e.clientX - rect.left) * (canvasIk.width  / rect.width);
  const cy   = (e.clientY - rect.top)  * (canvasIk.height / rect.height);

  const t      = Renderer.makeTransform(Config.viewport(), canvasIk.width, canvasIk.height);
  const [lx, ly] = t.toLinkage(cx, cy);

  state.ikTarget = { x: lx, y: ly };
  state.ikValid  = null;
  state.ikPts    = null;

  ikStatusEl.textContent = "Solving…";
  ikStatusEl.className   = "ik-status pending";
  renderIk();

  try {
    const result = await callIk(lx, ly);
    state.ikValid  = result.valid;
    state.ikAngles = { theta1: result.theta1, theta_c: result.theta_c };

    // Run FK at the solved angles to get all points
    state.ikPts = await callFk(state.ikAngles);
    renderIk();

    if (result.valid) {
      ikStatusEl.className   = "ik-status ok";
      ikStatusEl.textContent =
        `✓ Solved in ${result.iterations} iters — `+
        `error ${result.error_mm.toFixed(2)} mm — `+
        `θ1=${result.theta1.toFixed(1)}° θc=${result.theta_c.toFixed(1)}°`;
    } else {
      ikStatusEl.className   = "ik-status error";
      ikStatusEl.textContent =
        `✗ Unreachable — closest error ${result.error_mm.toFixed(1)} mm`;
    }
  } catch (e) {
    ikStatusEl.className   = "ik-status error";
    ikStatusEl.textContent = "IK error: " + e.message;
  }
});

// ── Reload config ──────────────────────────────────────────────────────
btnReload.addEventListener("click", async () => {
  btnReload.disabled   = true;
  btnReload.textContent = "Reloading…";
  try {
    await Config.reload();
    state.workspace = await fetchWorkspace();
    buildSliders();

    // Re-run FK with current (reset to default) angles
    for (const inp of Config.inputs()) {
      state.angles[inp.name] = inp.angle_deg;
    }
    state.fkPts    = await callFk(state.angles);
    state.ikPts    = null;
    state.ikTarget = null;
    state.ikValid  = null;

    resizeAll();
    setStatus("Config reloaded", "ok");
  } catch (e) {
    setStatus("Reload failed: " + e.message, "error");
  } finally {
    btnReload.disabled    = false;
    btnReload.textContent = "↺ Reload Config";
  }
});

// ── Boot ───────────────────────────────────────────────────────────────
(async () => {
  try {
    await Config.load();
    state.workspace = await fetchWorkspace();
    buildSliders();
    resizeCanvas(canvasFk);
    resizeCanvas(canvasIk);

    state.fkPts = await callFk(state.angles);
    renderFk();
    renderIk();

    setStatus("Ready", "ok");
  } catch (e) {
    setStatus("Boot error: " + e.message, "error");
    console.error(e);
  }
})();
