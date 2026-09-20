/**
 * config.js — loads config from /config endpoint and exposes helpers.
 * Also handles reload-config button.
 */

const Config = (() => {
  let _cfg = null;

  async function load() {
    const res = await fetch("/config");
    if (!res.ok) throw new Error(`Failed to load config: ${res.status}`);
    _cfg = await res.json();
    return _cfg;
  }

  async function reload() {
    // Tell server to re-read YAML
    const res = await fetch("/config/reload", { method: "POST" });
    if (!res.ok) throw new Error(`Reload failed: ${res.status}`);
    // Then refresh our local copy
    await load();
    return _cfg;
  }

  function get() {
    if (!_cfg) throw new Error("Config not loaded yet");
    return _cfg;
  }

  /** Return {name, angle_deg, min_deg, max_deg} for each input */
  function inputs() {
    return (get().inputs || []).map(inp => ({
      name:      inp.name,
      angle_deg: inp.angle_deg ?? 0,
      min_deg:   inp.min_deg   ?? -180,
      max_deg:   inp.max_deg   ??  180,
    }));
  }

  /** Return the viewport from config or sensible defaults */
  function viewport() {
    const vp = get().viewport || {};
    return {
      xmin: vp.xmin ?? -50,
      xmax: vp.xmax ??  250,
      ymin: vp.ymin ?? -200,
      ymax: vp.ymax ??  150,
    };
  }

  /** Name of the end effector point */
  function endEffector() {
    return get().end_effector ?? "D";
  }

  /** Return [{name, side, channels}] for the four legs */
  function legs() {
    const raw = get().legs || {};
    return Object.entries(raw).map(([name, l]) => ({
      name,
      side:     l.side,
      channels: l.channels,
    }));
  }

  /** Return the gait block with sensible defaults */
  function gait() {
    const g = get().gait || {};
    return {
      cycle_time_s:      g.cycle_time_s      ?? 1.2,
      duty:              g.duty              ?? 0.5,
      step_length_mm:    g.step_length_mm    ?? 40,
      step_height_mm:    g.step_height_mm    ?? 25,
      stance_x_mm:       g.stance_x_mm       ?? 150,
      stance_y_mm:       g.stance_y_mm       ?? -120,
      update_rate_hz:    g.update_rate_hz    ?? 30,
      forward_axis_sign: g.forward_axis_sign ?? 1,
    };
  }

  return { load, reload, get, inputs, viewport, endEffector, legs, gait };
})();
