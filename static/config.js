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

  return { load, reload, get, inputs, viewport, endEffector };
})();
