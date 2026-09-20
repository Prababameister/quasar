/**
 * renderer.js — draws quadruped legs on a canvas.
 *
 * Each leg is drawn into its own rectangular cell (a canvas quadrant).
 * Linkage mm coords are y-up; canvas is y-down. Right-side legs are drawn
 * with the x-axis flipped so they look like real right legs.
 */

const Renderer = (() => {

  const COLORS = {
    edge: ["#3B8BD4","#D85A30","#1D9E75","#7F77DD",
           "#BA7517","#A32D2D","#0F6E56","#993556"],
    fixed:  "#333333",
    driven: "#D85A30",
    free:   "#3B8BD4",
    target: "#1D9E75",
    danger: "#D85A30",
    workspace: "rgba(59,139,212,0.12)",
    endEffector: "#1D9E75",
    unreachable: "rgba(216,90,48,0.18)",
    cell:   "#e0dfd8",
    label:  "#6b6b66",
  };

  /**
   * Transform for one cell. rect = {x, y, w, h} in canvas px.
   * mirror = true flips the linkage x-axis.
   */
  function makeTransform(vp, rect, mirror = false) {
    const pad  = 14;
    const vpW  = vp.xmax - vp.xmin;
    const vpH  = vp.ymax - vp.ymin;
    const scale = Math.min((rect.w - 2 * pad) / vpW, (rect.h - 2 * pad) / vpH);

    const offX = rect.x + pad + (rect.w - 2 * pad - vpW * scale) / 2;
    const offY = rect.y + pad + (rect.h - 2 * pad - vpH * scale) / 2;

    return {
      scale,
      toCanvas(lx, ly) {
        const xx = mirror ? (vp.xmax - lx) : (lx - vp.xmin);
        return [offX + xx * scale, offY + (vp.ymax - ly) * scale];
      },
      toLinkage(cx, cy) {
        const xx = (cx - offX) / scale;
        const lx = mirror ? (vp.xmax - xx) : (xx + vp.xmin);
        return [lx, vp.ymax - (cy - offY) / scale];
      },
    };
  }

  function dot(ctx, x, y, r, fill, stroke = "#fff", sw = 1.5) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke();
  }

  function square(ctx, x, y, r, fill) {
    ctx.fillStyle = fill;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
    ctx.strokeRect(x - r, y - r, r * 2, r * 2);
  }

  function drawWorkspace(ctx, t, pts, unreachable) {
    if (!pts || !pts.length) return;
    ctx.save();
    const r = Math.max(1.5, t.scale * 1.2);
    ctx.fillStyle = unreachable ? COLORS.unreachable : COLORS.workspace;
    for (const p of pts) {
      const [cx, cy] = t.toCanvas(p.x, p.y);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  function clear(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  /**
   * Draw one leg into a cell.
   *
   * opts = {
   *   cfg, pts,            // pts: {name:{x,y}} canonical-frame FK/IK points
   *   workspace,           // [{x,y}] canonical-frame reachable cloud
   *   target,              // {x,y} canonical-frame IK target | null
   *   ikValid,             // bool | null
   *   endEffector,         // EE point name
   *   vp,                  // viewport
   *   mirror,              // flip x (right legs)
   *   label,               // leg name text
   *   selected,            // highlight cell border
   * }
   */
  function drawLeg(canvas, rect, opts) {
    const ctx = canvas.getContext("2d");
    const { cfg, pts, workspace, target, ikValid,
            endEffector, vp, mirror, label, selected } = opts;

    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();

    // Cell background + border
    ctx.fillStyle = "#fafaf8";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

    const t = makeTransform(vp, rect, mirror);

    // Axes
    ctx.strokeStyle = "#e6e6e2"; ctx.lineWidth = 1;
    const [ox, oyTop] = t.toCanvas(0, vp.ymax);
    const [, oyBot]   = t.toCanvas(0, vp.ymin);
    ctx.beginPath(); ctx.moveTo(ox, oyTop); ctx.lineTo(ox, oyBot); ctx.stroke();
    const [xL, oy] = t.toCanvas(vp.xmin, 0);
    const [xR]     = t.toCanvas(vp.xmax, 0);
    ctx.beginPath(); ctx.moveTo(xL, oy); ctx.lineTo(xR, oy); ctx.stroke();

    drawWorkspace(ctx, t, workspace, target != null && ikValid === false);

    if (cfg && pts) {
      const edges = cfg.edges || [];
      edges.forEach((edge, i) => {
        const a = pts[edge.from], b = pts[edge.to];
        if (!a || !b) return;
        const [ax, ay] = t.toCanvas(a.x, a.y);
        const [bx, by] = t.toCanvas(b.x, b.y);
        ctx.strokeStyle = COLORS.edge[i % COLORS.edge.length];
        ctx.lineWidth = 2.5; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      });

      const fixed  = new Set(Object.entries(cfg.points || {})
                              .filter(([, p]) => p.fixed).map(([n]) => n));
      const driven = new Set((cfg.inputs || []).map(i => i.point));

      for (const [name, pos] of Object.entries(pts)) {
        if (!pos) continue;
        const [cx, cy] = t.toCanvas(pos.x, pos.y);
        if (fixed.has(name))       square(ctx, cx, cy, 5, COLORS.fixed);
        else if (driven.has(name)) dot(ctx, cx, cy, 5, COLORS.driven);
        else if (name === endEffector) dot(ctx, cx, cy, 6, COLORS.endEffector, "#fff", 2);
        else dot(ctx, cx, cy, 4, COLORS.free);
      }
    }

    if (target != null) {
      const [tx, ty] = t.toCanvas(target.x, target.y);
      const r = 9;
      ctx.strokeStyle = ikValid === false ? COLORS.danger : COLORS.target;
      ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(tx - r, ty); ctx.lineTo(tx + r, ty); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx, ty - r); ctx.lineTo(tx, ty + r); ctx.stroke();
      ctx.setLineDash([]);
      dot(ctx, tx, ty, 3, ikValid === false ? COLORS.danger : COLORS.target, "#fff", 1.5);
    }

    // Label
    ctx.font = "600 12px system-ui";
    ctx.fillStyle = COLORS.label;
    ctx.textAlign = "left";
    ctx.fillText(label || "", rect.x + 10, rect.y + 18);

    ctx.restore();

    // Cell border (outside clip)
    ctx.strokeStyle = selected ? COLORS.target : COLORS.cell;
    ctx.lineWidth = selected ? 2 : 1;
    ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
  }

  return { makeTransform, drawLeg, clear };
})();
