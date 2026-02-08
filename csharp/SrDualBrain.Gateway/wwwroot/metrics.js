const $ = (id) => document.getElementById(id);

const els = {
  metricsSubtitle: $("metricsSubtitle"),
  mCoherence: $("mCoherence"),
  mTension: $("mTension"),
  mRouting: $("mRouting"),
  mAction: $("mAction"),
  mTemp: $("mTemp"),
  mLatency: $("mLatency"),
  mSystem2: $("mSystem2"),
  mMetaAction: $("mMetaAction"),
  mMetaCoverage: $("mMetaCoverage"),
  mMetaFlags: $("mMetaFlags"),
  activeModules: $("activeModules"),
  modulePath: $("modulePath"),
  brainActivity: $("brainActivity"),
  brainHistory: $("brainHistory"),
  moduleHistory: $("moduleHistory"),
  executiveMemo: $("executiveMemo"),
  executiveObserverMemo: $("executiveObserverMemo"),
  telemetryRaw: $("telemetryRaw"),
  dialogueFlowRaw: $("dialogueFlowRaw"),
  btnExportBundle: $("btnExportBundle"),
  btnExportTrace: $("btnExportTrace"),
  btnExportTelemetry: $("btnExportTelemetry"),
  btnExportDialogueFlow: $("btnExportDialogueFlow"),
  btnExportBrainHistory: $("btnExportBrainHistory"),
  btnExportModuleHistory: $("btnExportModuleHistory"),
  btnDock: $("btnDock"),
};

const brainHistoryBySession = new Map();
const brainHistoryLimit = 24;
const selectedQidBySession = new Map();

const moduleHistoryBySession = new Map();
const moduleHistoryLimit = 24;

let lastPayload = null;
const pendingChatRequests = new Map();

function nowStamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function downloadText(filename, text, { mime = "text/plain" } = {}) {
  const blob = new Blob([String(text)], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadJson(filename, obj) {
  downloadText(filename, JSON.stringify(obj, null, 2), { mime: "application/json" });
}

async function fetchTrace(sessionId, qid, { includeTelemetry, includeDialogueFlow, includeExecutive }) {
  const qs = new URLSearchParams({
    session_id: sessionId,
    include_telemetry: includeTelemetry ? "true" : "false",
    include_dialogue_flow: includeDialogueFlow ? "true" : "false",
    include_executive: includeExecutive ? "true" : "false",
  });
  const res = await fetch(`/v1/trace/${encodeURIComponent(qid)}?${qs.toString()}`, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Trace HTTP ${res.status}: ${text}`);
  }
  return await res.json();
}

async function getFullTracePayload(payload) {
  const base = payload && typeof payload === "object" ? payload : null;
  const sessionId = (base?.session_id || "default").trim() || "default";
  const qid = String(base?.qid || "").trim();
  if (!qid) throw new Error("No qid available yet.");

  const hasTelemetry = Array.isArray(base?.telemetry);
  const hasDialogueFlow = base?.dialogue_flow && typeof base.dialogue_flow === "object";
  const hasExecutive = base?.executive != null || base?.executive_observer != null;
  if (hasTelemetry && hasDialogueFlow && hasExecutive) {
    const clone = { ...base };
    delete clone._client;
    return clone;
  }

  let trace = null;
  try {
    trace = await fetchTrace(sessionId, qid, {
      includeTelemetry: true,
      includeDialogueFlow: true,
      includeExecutive: true,
    });
  } catch {
    trace = null;
  }
  if (!trace?.found) {
    const clone = { ...base };
    delete clone._client;
    return clone;
  }

  const merged = {
    ...base,
    qid: trace?.qid ?? qid,
    session_id: trace?.session_id ?? sessionId,
    metrics: trace?.metrics ?? {},
    telemetry: trace?.telemetry ?? [],
    dialogue_flow: trace?.dialogue_flow ?? {},
    executive: trace?.executive ?? null,
    executive_observer: trace?.executive_observer ?? null,
    ts: trace?.ts ?? null,
  };
  delete merged._client;
  return merged;
}

function renderModulePath(stages) {
  if (!els.modulePath) return;
  els.modulePath.innerHTML = "";
  const list = Array.isArray(stages) ? stages : [];
  if (!list.length) {
    const empty = document.createElement("div");
    empty.className = "path__empty";
    empty.textContent = "—";
    els.modulePath.appendChild(empty);
    return;
  }

  const stageLimit = 8;
  for (const stage of list.slice(0, stageLimit)) {
    const nameRaw = stage?.stage ?? "";
    const name = String(nameRaw || "").replaceAll("_", " ") || "stage";
    const mods = Array.isArray(stage?.modules) ? stage.modules.map(String) : [];

    const wrapper = document.createElement("div");
    wrapper.className = "path__stage";

    const title = document.createElement("div");
    title.className = "path__name";
    title.textContent = name;
    wrapper.appendChild(title);

    const chips = document.createElement("div");
    chips.className = "path__modules";
    const limit = 14;
    const shown = mods.slice(0, limit);
    for (const mod of shown) {
      const chip = document.createElement("span");
      chip.className = "chip chip--thin";
      chip.textContent = mod;
      chips.appendChild(chip);
    }
    const rest = mods.length - shown.length;
    if (rest > 0) {
      const chip = document.createElement("span");
      chip.className = "chip chip--thin";
      chip.textContent = `+${rest} more`;
      chips.appendChild(chip);
    }
    wrapper.appendChild(chips);
    els.modulePath.appendChild(wrapper);
  }
}

function clamp01(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function renderBrainActivity(metrics) {
  if (!els.brainActivity) return;
  els.brainActivity.innerHTML = "";

  const brain = metrics?.brain ?? null;
  if (!brain || typeof brain !== "object") {
    const empty = document.createElement("div");
    empty.className = "brain__empty";
    empty.textContent = "—";
    els.brainActivity.appendChild(empty);
    return;
  }

  const pfc = brain?.prefrontal ?? {};
  const amg = brain?.amygdala ?? {};
  const basal = brain?.basal_ganglia ?? {};
  const neuro = brain?.neural ?? {};
  const psychoid = brain?.psychoid ?? {};
  const hemi = brain?.hemisphere ?? {};

  const nodes = [
    {
      label: "Prefrontal (focus)",
      meta: `metric=${pfc?.metric ?? "—"} rel=${pfc?.relevance ?? "—"} hip=${pfc?.hippocampal_overlap ?? "—"}`,
      intensity: clamp01(pfc?.metric),
      accent: "var(--primary-rgb)",
    },
    {
      label: "Amygdala (risk)",
      meta: `risk=${amg?.risk ?? "—"} v=${amg?.valence ?? "—"} a=${amg?.arousal ?? "—"}`,
      intensity: clamp01(amg?.risk),
      accent: "var(--danger-rgb)",
    },
    {
      label: "Basal ganglia (go)",
      meta: `go=${basal?.go_probability ?? "—"} dopamine=${basal?.dopamine_level ?? "—"}`,
      intensity: clamp01(basal?.go_probability),
      accent: "var(--good-rgb)",
    },
    {
      label: "Basal ganglia (inhibit)",
      meta: `inh=${basal?.inhibition ?? "—"} action=${basal?.recommended_action ?? "—"}`,
      intensity: clamp01(basal?.inhibition),
      accent: "var(--warn-rgb)",
    },
    {
      label: "Hemisphere routing",
      meta: `mode=${hemi?.mode ?? "—"} intensity=${hemi?.intensity ?? "—"}`,
      intensity: clamp01(hemi?.intensity),
      accent: "var(--primary-rgb)",
    },
    {
      label: "Neural impulses",
      meta: `active=${neuro?.active_ratio ?? "—"} total=${neuro?.total_impulses ?? "—"}`,
      intensity: clamp01(neuro?.active_ratio),
      accent: "var(--good-rgb)",
    },
    {
      label: "Psychoid attention",
      meta: `norm=${psychoid?.norm ?? "—"} tension=${psychoid?.psychoid_tension ?? "—"}`,
      intensity: clamp01(psychoid?.psychoid_tension),
      accent: "var(--warn-rgb)",
    },
    {
      label: "Coherence / tension",
      meta: `coh=${metrics?.coherence?.combined ?? "—"} ten=${metrics?.coherence?.tension ?? "—"}`,
      intensity: clamp01(metrics?.coherence?.combined),
      accent: "var(--primary-rgb)",
    },
  ];

  for (const node of nodes) {
    const el = document.createElement("div");
    el.className = "brain__node";
    el.style.setProperty("--i", String(node.intensity));
    el.style.setProperty("--accent", node.accent);

    const title = document.createElement("div");
    title.className = "brain__label";
    title.textContent = node.label;
    el.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "brain__meta";
    meta.textContent = node.meta;
    el.appendChild(meta);

    const bar = document.createElement("div");
    bar.className = "brain__bar";
    const fill = document.createElement("div");
    fill.className = "brain__fill";
    bar.appendChild(fill);
    el.appendChild(bar);

    els.brainActivity.appendChild(el);
  }
}

function getBrainHistory(sessionId) {
  const key = (sessionId || "default").trim() || "default";
  let history = brainHistoryBySession.get(key);
  if (!history) {
    history = [];
    brainHistoryBySession.set(key, history);
  }
  return history;
}

function getModuleHistory(sessionId) {
  const key = (sessionId || "default").trim() || "default";
  let history = moduleHistoryBySession.get(key);
  if (!history) {
    history = [];
    moduleHistoryBySession.set(key, history);
  }
  return history;
}

function snapshotBrain(metrics, { qid }) {
  const brain = metrics?.brain ?? null;
  if (!brain || typeof brain !== "object") return null;

  const pfc = brain?.prefrontal ?? {};
  const amg = brain?.amygdala ?? {};
  const basal = brain?.basal_ganglia ?? {};
  const neuro = brain?.neural ?? {};
  const psychoid = brain?.psychoid ?? {};
  const hemi = brain?.hemisphere ?? {};

  return {
    qid: qid || "",
    ts: Date.now(),
    values: {
      pfc_focus: clamp01(pfc?.metric),
      amygdala_risk: clamp01(amg?.risk),
      basal_go: clamp01(basal?.go_probability),
      basal_inhibit: clamp01(basal?.inhibition),
      hemisphere: clamp01(hemi?.intensity),
      neural_active: clamp01(neuro?.active_ratio),
      psychoid_tension: clamp01(psychoid?.psychoid_tension),
      coherence: clamp01(metrics?.coherence?.combined),
      tension: clamp01(metrics?.coherence?.tension),
    },
  };
}

function snapshotModules(metrics, { qid }) {
  const modules = metrics?.modules?.active ?? [];
  const list = Array.isArray(modules) ? modules.map(String).filter(Boolean) : [];
  const stageByModule = {};
  const stages = metrics?.modules?.stages ?? [];
  if (Array.isArray(stages)) {
    for (const stage of stages) {
      const stageKey = String(stage?.stage || "").trim();
      const mods = Array.isArray(stage?.modules) ? stage.modules : [];
      for (const mod of mods) {
        const name = String(mod || "").trim();
        if (!name) continue;
        if (!stageByModule[name]) stageByModule[name] = stageKey;
      }
    }
  }
  return { qid: qid || "", ts: Date.now(), modules: list, stage_by_module: stageByModule };
}

function upsertBrainHistory(sessionId, snapshot) {
  if (!snapshot) return [];
  const history = getBrainHistory(sessionId);
  const last = history.length ? history[history.length - 1] : null;
  if (last && last.qid && snapshot.qid && last.qid === snapshot.qid) {
    history[history.length - 1] = snapshot;
  } else {
    history.push(snapshot);
  }
  while (history.length > brainHistoryLimit) history.shift();
  return history;
}

function upsertModuleHistory(sessionId, snapshot) {
  if (!snapshot) return [];
  const history = getModuleHistory(sessionId);
  const last = history.length ? history[history.length - 1] : null;
  if (last && last.qid && snapshot.qid && last.qid === snapshot.qid) {
    history[history.length - 1] = snapshot;
  } else {
    history.push(snapshot);
  }
  while (history.length > moduleHistoryLimit) history.shift();
  return history;
}

function stageAccent(stageName) {
  const key = String(stageName || "")
    .trim()
    .toLowerCase()
    .replaceAll(" ", "_");
  if (key === "perception") return "var(--primary-rgb)";
  if (key === "inner_dialogue" || key === "inner") return "var(--warn-rgb)";
  if (key === "integration") return "var(--good-rgb)";
  if (key === "memory") return "var(--memory-rgb)";
  return "var(--primary-rgb)";
}

function renderBrainHistory(sessionId) {
  if (!els.brainHistory) return;
  els.brainHistory.innerHTML = "";

  const history = getBrainHistory(sessionId);
  const selectedQid = selectedQidBySession.get(sessionId) || "";
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "bh__empty";
    empty.textContent = "—";
    els.brainHistory.appendChild(empty);
    return;
  }

  const head = document.createElement("div");
  head.className = "bh__head";
  const headLeft = document.createElement("div");
  headLeft.textContent = `History (last ${history.length})`;
  const headRight = document.createElement("div");
  headRight.className = "bh__hint";
  headRight.textContent = "oldest → newest";
  head.appendChild(headLeft);
  head.appendChild(headRight);
  els.brainHistory.appendChild(head);

  const rows = [
    { key: "pfc_focus", label: "PFC focus", accent: "var(--primary-rgb)" },
    { key: "amygdala_risk", label: "Amygdala risk", accent: "var(--danger-rgb)" },
    { key: "basal_go", label: "Basal go", accent: "var(--good-rgb)" },
    { key: "basal_inhibit", label: "Basal inhibit", accent: "var(--warn-rgb)" },
    { key: "hemisphere", label: "Hemisphere", accent: "var(--primary-rgb)" },
    { key: "neural_active", label: "Neural active", accent: "var(--good-rgb)" },
    { key: "psychoid_tension", label: "Psychoid tension", accent: "var(--warn-rgb)" },
    { key: "coherence", label: "Coherence", accent: "var(--primary-rgb)" },
    { key: "tension", label: "Tension", accent: "var(--warn-rgb)" },
  ];

  for (const row of rows) {
    const rowEl = document.createElement("div");
    rowEl.className = "bh__row";

    const label = document.createElement("div");
    label.className = "bh__label";
    label.textContent = row.label;
    rowEl.appendChild(label);

    const cells = document.createElement("div");
    cells.className = "bh__cells";
    for (let i = 0; i < history.length; i++) {
      const snap = history[i] || {};
      const value = clamp01(snap?.values?.[row.key]);
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "bh__cell";
      cell.style.setProperty("--i", String(value));
      cell.style.setProperty("--accent", row.accent);
      const ts = snap?.ts ? new Date(snap.ts).toLocaleTimeString() : "";
      const qid = String(snap?.qid || "").slice(0, 8);
      cell.title = `${ts}${qid ? ` · qid ${qid}` : ""} · ${row.label}=${value.toFixed(2)}`;
      if (snap?.qid && selectedQid && snap.qid === selectedQid) {
        cell.classList.add("bh__cell--selected");
      }
      cell.addEventListener("click", async () => {
        const fullQid = String(snap?.qid || "").trim();
        if (!fullQid) return;
        postToOpener({ type: "srdb.trace.jump", payload: { session_id: sessionId, qid: fullQid } });
        if (!window.opener || window.opener.closed) {
          try {
            const trace = await fetchTrace(sessionId, fullQid, {
              includeTelemetry: true,
              includeDialogueFlow: true,
              includeExecutive: true,
            });
            if (trace?.found) {
              renderMetrics({
                qid: trace?.qid ?? fullQid,
                session_id: trace?.session_id ?? sessionId,
                metrics: trace?.metrics ?? {},
                telemetry: trace?.telemetry ?? [],
                dialogue_flow: trace?.dialogue_flow ?? {},
                executive: trace?.executive ?? null,
                executive_observer: trace?.executive_observer ?? null,
                ts: trace?.ts ?? null,
              });
            }
          } catch (err) {
            console.warn("trace jump failed", err);
          }
        }
      });
      cells.appendChild(cell);
    }
    rowEl.appendChild(cells);
    els.brainHistory.appendChild(rowEl);
  }
}

function renderModuleHistory(sessionId) {
  if (!els.moduleHistory) return;
  els.moduleHistory.innerHTML = "";

  const history = getModuleHistory(sessionId);
  const selectedQid = selectedQidBySession.get(sessionId) || "";

  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "mh__empty";
    empty.textContent = "—";
    els.moduleHistory.appendChild(empty);
    return;
  }

  const counts = new Map();
  for (const snap of history) {
    const mods = Array.isArray(snap?.modules) ? snap.modules : [];
    for (const m of mods) {
      const k = String(m || "");
      if (!k) continue;
      counts.set(k, (counts.get(k) || 0) + 1);
    }
  }
  const modules = Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))
    .map(([m]) => m);

  const head = document.createElement("div");
  head.className = "mh__head";
  const headLeft = document.createElement("div");
  headLeft.textContent = `History (last ${history.length})`;
  const headRight = document.createElement("div");
  headRight.className = "mh__hint";
  headRight.textContent = "oldest → newest";
  head.appendChild(headLeft);
  head.appendChild(headRight);
  els.moduleHistory.appendChild(head);

  const legend = document.createElement("div");
  legend.className = "mh__legend";
  const legendItems = [
    { label: "perception", accent: "var(--primary-rgb)" },
    { label: "inner_dialogue", accent: "var(--warn-rgb)" },
    { label: "integration", accent: "var(--good-rgb)" },
    { label: "memory", accent: "var(--memory-rgb)" },
  ];
  for (const item of legendItems) {
    const el = document.createElement("div");
    el.className = "mh__legendItem";
    const dot = document.createElement("span");
    dot.className = "mh__legendDot";
    dot.style.setProperty("--accent", item.accent);
    const label = document.createElement("span");
    label.textContent = item.label;
    el.appendChild(dot);
    el.appendChild(label);
    legend.appendChild(el);
  }
  els.moduleHistory.appendChild(legend);

  for (const mod of modules) {
    const rowEl = document.createElement("div");
    rowEl.className = "mh__row";

    const label = document.createElement("div");
    label.className = "mh__label";
    label.textContent = mod;
    label.title = mod;
    rowEl.appendChild(label);

    const cells = document.createElement("div");
    cells.className = "mh__cells";

    for (const snap of history) {
      const active = Array.isArray(snap?.modules) ? snap.modules.includes(mod) : false;
      const stage = snap?.stage_by_module?.[mod] ?? null;
      const accent = stageAccent(stage);
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "mh__cell";
      cell.style.setProperty("--accent", accent);
      cell.style.setProperty("--i", active ? "1" : "0");
      const ts = snap?.ts ? new Date(snap.ts).toLocaleTimeString() : "";
      const qid = String(snap?.qid || "").slice(0, 8);
      const stageLabel = stage ? ` · stage ${stage}` : "";
      cell.title = `${ts}${qid ? ` · qid ${qid}` : ""}${stageLabel} · ${mod}: ${active ? "on" : "off"}`;
      if (snap?.qid && selectedQid && snap.qid === selectedQid) {
        cell.classList.add("mh__cell--selected");
      }
      cell.addEventListener("click", () => {
        const fullQid = String(snap?.qid || "").trim();
        if (!fullQid) return;
        postToOpener({ type: "srdb.trace.jump", payload: { session_id: sessionId, qid: fullQid } });
      });
      cells.appendChild(cell);
    }

    rowEl.appendChild(cells);
    els.moduleHistory.appendChild(rowEl);
  }
}

function renderMetrics(response) {
  lastPayload = response;
  const qid = response?.qid ?? "";
  const sessionId = response?.session_id ?? "";
  const stamp = new Date().toLocaleTimeString();
  const subtitle = [qid ? `qid ${qid}` : "", sessionId ? `session ${sessionId}` : "", stamp].filter(Boolean).join(" · ");
  els.metricsSubtitle.textContent = subtitle || "—";
  selectedQidBySession.set(sessionId || "default", qid || "");

  const telemetry = response?.telemetry ?? [];
  const dialogueFlow = response?.dialogue_flow ?? {};
  const metrics = response?.metrics ?? null;
  const executive = response?.executive ?? dialogueFlow?.executive ?? null;
  const executiveObserver = response?.executive_observer ?? dialogueFlow?.executive_observer ?? null;

  const combined = metrics?.coherence?.combined ?? null;
  const tension = metrics?.coherence?.tension ?? null;
  const routing = metrics?.coherence?.mode ?? (metrics?.leading ?? null);
  const action = metrics?.policy?.action ?? null;
  const temp = metrics?.policy?.temperature ?? null;
  const latency = metrics?.latency_ms ?? null;
  const system2 = metrics?.system2 ?? null;
  const meta = metrics?.metacognition ?? null;

  els.mCoherence.textContent = combined == null ? "—" : Number(combined).toFixed(3);
  els.mTension.textContent = tension == null ? "—" : Number(tension).toFixed(3);
  els.mRouting.textContent = routing == null ? "—" : String(routing);
  els.mAction.textContent = action == null ? "—" : String(action);
  els.mTemp.textContent = temp == null ? "—" : Number(temp).toFixed(2);
  els.mLatency.textContent = latency == null ? "—" : `${Math.round(Number(latency))}ms`;
  if (els.mSystem2) {
    const enabled = system2?.enabled ?? null;
    const mode = system2?.mode ?? null;
    let label = "—";
    if (enabled === true) label = "on";
    else if (mode) label = String(mode);

    const roundsRaw = system2?.rounds;
    const roundTargetRaw = system2?.round_target;
    const rounds = Number(roundsRaw);
    const roundTarget = Number(roundTargetRaw);
    if (Number.isFinite(rounds)) {
      if (Number.isFinite(roundTarget)) {
        label += ` r${Math.max(0, Math.trunc(rounds))}/${Math.max(1, Math.trunc(roundTarget))}`;
      } else {
        label += ` r${Math.max(0, Math.trunc(rounds))}`;
      }
    }

    const initIssuesRaw = system2?.initial_issues;
    const finalIssuesRaw = system2?.final_issues;
    const initIssues = Number(initIssuesRaw);
    const finalIssues = Number(finalIssuesRaw);
    if (Number.isFinite(initIssues) && Number.isFinite(finalIssues)) {
      label += ` ${Math.max(0, Math.trunc(initIssues))}→${Math.max(0, Math.trunc(finalIssues))}`;
    }
    if (system2?.resolved === true) {
      label += " resolved";
    }
    if (system2?.followup_revision === true) {
      label += " +followup";
    }
    els.mSystem2.textContent = label;
  }
  if (els.mMetaAction) {
    els.mMetaAction.textContent = meta?.action ? String(meta.action) : "—";
  }
  if (els.mMetaCoverage) {
    const cov = meta?.coverage;
    els.mMetaCoverage.textContent = cov == null ? "—" : Number(cov).toFixed(2);
  }
  if (els.mMetaFlags) {
    const flags = Array.isArray(meta?.flags) ? meta.flags.map(String).filter(Boolean) : [];
    els.mMetaFlags.textContent = flags.length ? flags.join(", ") : "—";
  }

  const modules = metrics?.modules?.active ?? [];
  if (els.activeModules) {
    els.activeModules.innerHTML = "";
    const limit = 18;
    const shown = Array.isArray(modules) ? modules.slice(0, limit) : [];
    for (const name of shown) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = String(name);
      els.activeModules.appendChild(chip);
    }
    const rest = Array.isArray(modules) ? modules.length - shown.length : 0;
    if (rest > 0) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `+${rest} more`;
      els.activeModules.appendChild(chip);
    }
    if (!shown.length) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = "modules: —";
      els.activeModules.appendChild(chip);
    }
  }

  renderModulePath(metrics?.modules?.stages ?? dialogueFlow?.architecture ?? null);
  renderBrainActivity(metrics);
  if (Array.isArray(response?._client?.brain_history)) {
    brainHistoryBySession.set(sessionId || "default", response._client.brain_history);
  } else {
    upsertBrainHistory(sessionId || "default", snapshotBrain(metrics, { qid }));
  }
  renderBrainHistory(sessionId || "default");
  if (Array.isArray(response?._client?.module_history)) {
    moduleHistoryBySession.set(sessionId || "default", response._client.module_history);
  } else {
    upsertModuleHistory(sessionId || "default", snapshotModules(metrics, { qid }));
  }
  renderModuleHistory(sessionId || "default");

  if (els.executiveMemo) {
    if (executive) {
      const memo = executive?.memo ?? "";
      const mixIn = executive?.mix_in ?? "";
      const directives = executive?.directives ?? null;
      const meta = `source=${executive?.source ?? "?"} conf=${executive?.confidence ?? "?"} latency=${Math.round(executive?.latency_ms ?? 0)}ms`;
      const body = memo ? memo : "";
      const mix = mixIn ? `\n\n---\nMix-in:\n${String(mixIn)}` : "";
      const dir = directives ? `\n\n---\nDirectives:\n${JSON.stringify(directives, null, 2)}` : "";
      els.executiveMemo.textContent = `${body}${mix}\n\n---\n${meta}${dir}`.trim();
    } else {
      els.executiveMemo.textContent = "—";
    }
  }

  if (els.executiveObserverMemo) {
    if (executiveObserver) {
      const memo = executiveObserver?.memo ?? "";
      const mixIn = executiveObserver?.mix_in ?? "";
      const directives = executiveObserver?.directives ?? null;
      const meta = `source=${executiveObserver?.source ?? "?"} conf=${executiveObserver?.confidence ?? "?"} latency=${Math.round(executiveObserver?.latency_ms ?? 0)}ms`;
      const mode = executiveObserver?.observer_mode ? `mode=${String(executiveObserver.observer_mode)}` : "";
      const body = memo ? memo : "";
      const mix = mixIn ? `\n\n---\nMix-in:\n${String(mixIn)}` : "";
      const dir = directives ? `\n\n---\nDirectives:\n${JSON.stringify(directives, null, 2)}` : "";
      els.executiveObserverMemo.textContent = `${body}${mix}\n\n---\n${[meta, mode].filter(Boolean).join(" ")}${dir}`.trim();
    } else {
      els.executiveObserverMemo.textContent = "—";
    }
  }

  els.telemetryRaw.textContent = JSON.stringify(telemetry, null, 2);
  if (els.dialogueFlowRaw) {
    els.dialogueFlowRaw.textContent = JSON.stringify(dialogueFlow, null, 2);
  }
}

function postToOpener(msg) {
  try {
    if (window.opener && !window.opener.closed) {
      window.opener.postMessage(msg, window.location.origin);
    }
  } catch {
    // ignore
  }
}

function requestChatFromOpener(sessionId) {
  if (!window.opener || window.opener.closed) return Promise.resolve(null);
  const requestId = (globalThis.crypto && crypto.randomUUID) ? crypto.randomUUID() : `req_${Date.now()}_${Math.random().toString(16).slice(2)}`;

  return new Promise((resolve) => {
    pendingChatRequests.set(requestId, { resolve, ts: Date.now() });
    postToOpener({
      type: "srdb.chat.request",
      payload: { request_id: requestId, session_id: sessionId || null },
    });
    setTimeout(() => {
      const pending = pendingChatRequests.get(requestId);
      if (!pending) return;
      pendingChatRequests.delete(requestId);
      resolve(null);
    }, 1200);
  });
}

window.addEventListener("message", (e) => {
  if (e.origin !== window.location.origin) return;
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "srdb.metrics") {
    renderMetrics(msg.payload);
    return;
  }

  if (msg.type === "srdb.chat.response") {
    const payload = msg.payload ?? {};
    const requestId = String(payload?.request_id || "").trim();
    if (!requestId) return;
    const pending = pendingChatRequests.get(requestId);
    if (!pending) return;
    pendingChatRequests.delete(requestId);
    const messages = Array.isArray(payload?.messages) ? payload.messages : null;
    pending.resolve(messages);
    return;
  }
});

els.btnExportBundle?.addEventListener("click", async () => {
  try {
    const full = await getFullTracePayload(lastPayload);
    const sid = (full?.session_id || "default").trim() || "default";
    const qid = String(full?.qid || "").slice(0, 8) || "trace";
    const brainHistory = getBrainHistory(sid);
    const moduleHistory = getModuleHistory(sid);
    const chatMessages = await requestChatFromOpener(sid);
    const chat = chatMessages ? { session_id: sid, messages: chatMessages } : null;

    downloadJson(`srdb_bundle_${sid}_${qid}_${nowStamp()}.json`, {
      version: "srdb_bundle_v1",
      exported_at: new Date().toISOString(),
      session_id: sid,
      qid: full?.qid ?? "",
      trace: full,
      brain_history: { session_id: sid, limit: brainHistoryLimit, items: brainHistory },
      module_history: { session_id: sid, limit: moduleHistoryLimit, items: moduleHistory },
      chat,
    });
  } catch (err) {
    console.warn("export bundle failed", err);
  }
});

els.btnExportTrace?.addEventListener("click", async () => {
  try {
    const full = await getFullTracePayload(lastPayload);
    const sid = (full?.session_id || "default").trim() || "default";
    const qid = String(full?.qid || "").slice(0, 8) || "trace";
    downloadJson(`srdb_trace_${sid}_${qid}_${nowStamp()}.json`, full);
  } catch (err) {
    console.warn("export trace failed", err);
  }
});

els.btnExportTelemetry?.addEventListener("click", async () => {
  try {
    const full = await getFullTracePayload(lastPayload);
    const sid = (full?.session_id || "default").trim() || "default";
    const qid = String(full?.qid || "").slice(0, 8) || "trace";
    downloadJson(`srdb_telemetry_${sid}_${qid}_${nowStamp()}.json`, full?.telemetry ?? []);
  } catch (err) {
    console.warn("export telemetry failed", err);
  }
});

els.btnExportDialogueFlow?.addEventListener("click", async () => {
  try {
    const full = await getFullTracePayload(lastPayload);
    const sid = (full?.session_id || "default").trim() || "default";
    const qid = String(full?.qid || "").slice(0, 8) || "trace";
    downloadJson(`srdb_dialogue_flow_${sid}_${qid}_${nowStamp()}.json`, full?.dialogue_flow ?? {});
  } catch (err) {
    console.warn("export dialogue failed", err);
  }
});

els.btnExportBrainHistory?.addEventListener("click", () => {
  try {
    const sid = (lastPayload?.session_id || "default").trim() || "default";
    const history = getBrainHistory(sid);
    downloadJson(`srdb_brain_history_${sid}_${nowStamp()}.json`, { session_id: sid, limit: brainHistoryLimit, items: history });
  } catch (err) {
    console.warn("export brain history failed", err);
  }
});

els.btnExportModuleHistory?.addEventListener("click", () => {
  try {
    const sid = (lastPayload?.session_id || "default").trim() || "default";
    const history = getModuleHistory(sid);
    downloadJson(`srdb_module_history_${sid}_${nowStamp()}.json`, { session_id: sid, limit: moduleHistoryLimit, items: history });
  } catch (err) {
    console.warn("export module history failed", err);
  }
});

els.btnDock?.addEventListener("click", () => {
  postToOpener({ type: "srdb.metrics.dock" });
  window.close();
});

postToOpener({ type: "srdb.metrics.ready" });
