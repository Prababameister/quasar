/**
 * renderer.js — draws the linkage on a canvas element.
 *
 * Coordinate system: linkage mm coords, y-up.
 * Canvas coords:     pixels, y-down (flipped).
 */

const Renderer = (() => {

  const COLORS = {
    edge:    ["#3B8BD4","#D85A30","#1D9E75","#7F77DD",
              "#BA7517","#A32D2D","#0F6E56","#993556"],
    fixed:   "#333333",
    driven:  "#D85A30",
    free:    "#3B8BD4",
    target:  "#D85A30",
    workspace: "rgba(59,139,212,0.12)",
    workspaceBorder: "rgba(59,139,212,0.35)",
    endEffector: "#1D9E75",
    unreachable: "rgba(216,90,48,0.18)",
  };

  /**
   * Build a transform object from viewport + canvas size.
   * Exposes toCanvas(x,y) and toLinkage(cx,cy).
   */
  function makeTransform(vp, canvasW, canvasH) {
    const pad  = 20;
    const vpW  = vp.xmax - vp.xmin;
    const vpH  = vp.ymax - vp.ymin;
    const scaleX = (canvasW - 2 * pad) / vpW;
    const scaleY = (canvasH - 2 * pad) / vpH;
    const scale  = Math.min(scaleX, scaleY);

    // Centre the linkage in the canvas
    const offX = pad + (canvasW - 2 * pad - vpW * scale) / 2;
    const offY = pad + (canvasH - 2 * pad - vpH * scale) / 2;

    return {
      scale,
      toCanvas(lx, ly) {
        return [
          offX + (lx - vp.xmin) * scale,
          offY + (vp.ymax - ly) * scale,   // flip y
        ];
      },
      toLinkage(cx, cy) {
        return [
          (cx - offX) / scale + vp.xmin,
          vp.ymax - (cy - offY) / scale,   // flip y
        ];
      },
    };
  }

  /** Draw a filled circle */
  function dot(ctx, x, y, r, fill, stroke = "#fff", sw = 1.5) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = sw;
    ctx.stroke();
  }

  /** Draw a square (fixed pivot) */
  function square(ctx, x, y, r, fill) {
    ctx.fillStyle = fill;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x - r, y - r, r * 2, r * 2);
    // ground lines
    ctx.strokeStyle = "#888";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x - 8, y + r + 2);
    ctx.lineTo(x + 8, y + r + 2);
    ctx.stroke();
  }

  /**
   * Draw workspace scatter cloud.
   * workspacePts: [{x,y}] in linkage coords.
   * unreachable:  true  -> tint red instead of blue (for clicked-outside feedback).
   */
  function drawWorkspace(ctx, t, workspacePts, unreachable = false) {
    if (!workspacePts || workspacePts.length === 0) return;
    ctx.save();
    const r = Math.max(2, t.scale * 1.5);
    ctx.fillStyle = unreachable
      ? COLORS.unreachable
      : COLORS.workspace;
    for (const pt of workspacePts) {
      const [cx, cy] = t.toCanvas(pt.x, pt.y);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  /**
   * Main draw call.
   *
   * opts = {
   *   cfg,           // parsed config
   *   pts,           // {name: {x,y}} from /fk or /ik
   *   workspace,     // [{x,y}] workspace points
   *   target,        // {x,y} | null — IK click target in linkage coords
   *   ikValid,       // bool | null
   *   endEffector,   // string — name of EE point
   *   vp,            // viewport
   * }
   */
  function draw(canvas, opts) {
    const { cfg, pts, workspace, target, ikValid, endEffector, vp } = opts;
    if (!cfg || !pts) return;

    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    const t = makeTransform(vp, W, H);

    // Grid
    ctx.save();
    ctx.strokeStyle = "#ebebeb";
    ctx.lineWidth   = 0.5;
    const step = 50;
    for (let x = Math.ceil(vp.xmin / step) * step; x <= vp.xmax; x += step) {
      const [cx] = t.toCanvas(x, 0);
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    }
    for (let y = Math.ceil(vp.ymin / step) * step; y <= vp.ymax; y += step) {
      const [, cy] = t.toCanvas(0, y);
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
    }
    ctx.restore();

    // Axes
    ctx.save();
    ctx.strokeStyle = "#ccc";
    ctx.lineWidth   = 1;
    const [ox, oy] = t.toCanvas(0, 0);
    ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(W, oy); ctx.stroke();
    ctx.restore();

    // Workspace
    drawWorkspace(ctx, t, workspace, target !== null && ikValid === false);

    // Edges
    const edges = cfg.edges || [];
    edges.forEach((edge, i) => {
      const a = pts[edge.from], b = pts[edge.to];
      if (!a || !b) return;
      const [ax, ay] = t.toCanvas(a.x, a.y);
      const [bx, by] = t.toCanvas(b.x, b.y);
      ctx.save();
      ctx.strokeStyle = COLORS.edge[i % COLORS.edge.length];
      ctx.lineWidth   = 2.5;
      ctx.lineCap     = "round";
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();

      // Length label
      const mx = (ax + bx) / 2, my = (ay + by) / 2;
      ctx.fillStyle = COLORS.edge[i % COLORS.edge.length];
      ctx.font      = "10px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(edge.length.toFixed(2), mx, my - 4);
      ctx.restore();
    });

    // Joints
    const fixed   = new Set(Object.entries(cfg.points)
                              .filter(([,p]) => p.fixed).map(([n]) => n));
    const driven  = new Set((cfg.inputs || []).map(i => i.point));

    for (const [name, pos] of Object.entries(pts)) {
      if (!pos) continue;
      const [cx, cy] = t.toCanvas(pos.x, pos.y);
      const isEE = name === endEffector;

      if (fixed.has(name)) {
        square(ctx, cx, cy, 6, COLORS.fixed);
      } else if (driven.has(name)) {
        dot(ctx, cx, cy, 6, COLORS.driven);
      } else if (isEE) {
        dot(ctx, cx, cy, 7, COLORS.endEffector, "#fff", 2);
      } else {
        dot(ctx, cx, cy, 5, COLORS.free);
      }

      // Label
      ctx.save();
      ctx.font      = "500 11px system-ui";
      ctx.fillStyle = "#222";
      ctx.fillText(name, cx + 7, cy - 5);
      ctx.restore();
    }

    // IK target cross-hair
    if (target !== null) {
      const [tx, ty] = t.toCanvas(target.x, target.y);
      const r = 10;
      ctx.save();
      ctx.strokeStyle = ikValid === false ? COLORS.danger : COLORS.target;
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(tx - r, ty); ctx.lineTo(tx + r, ty); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx, ty - r); ctx.lineTo(tx, ty + r); ctx.stroke();
      ctx.setLineDash([]);
      dot(ctx, tx, ty, 4,
          ikValid === false ? "#D85A30" : "#1D9E75", "#fff", 1.5);
      ctx.restore();
    }
  }

  return { draw, makeTransform };
})();
