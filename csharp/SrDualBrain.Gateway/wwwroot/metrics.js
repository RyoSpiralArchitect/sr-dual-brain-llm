const $ = (id) => document.getElementById(id);

const els = {
  metricsSubtitle: $("metricsSubtitle"),
  mCoherence: $("mCoherence"),
  mTension: $("mTension"),
  mRouting: $("mRouting"),
  mAction: $("mAction"),
  mTemp: $("mTemp"),
  mLatency: $("mLatency"),
  activeModules: $("activeModules"),
  modulePath: $("modulePath"),
  brainActivity: $("brainActivity"),
  executiveMemo: $("executiveMemo"),
  executiveObserverMemo: $("executiveObserverMemo"),
  telemetryRaw: $("telemetryRaw"),
  dialogueFlowRaw: $("dialogueFlowRaw"),
  btnDock: $("btnDock"),
};

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

function renderMetrics(response) {
  const qid = response?.qid ?? "";
  const sessionId = response?.session_id ?? "";
  const stamp = new Date().toLocaleTimeString();
  const subtitle = [qid ? `qid ${qid}` : "", sessionId ? `session ${sessionId}` : "", stamp].filter(Boolean).join(" · ");
  els.metricsSubtitle.textContent = subtitle || "—";

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

  els.mCoherence.textContent = combined == null ? "—" : Number(combined).toFixed(3);
  els.mTension.textContent = tension == null ? "—" : Number(tension).toFixed(3);
  els.mRouting.textContent = routing == null ? "—" : String(routing);
  els.mAction.textContent = action == null ? "—" : String(action);
  els.mTemp.textContent = temp == null ? "—" : Number(temp).toFixed(2);
  els.mLatency.textContent = latency == null ? "—" : `${Math.round(Number(latency))}ms`;

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

window.addEventListener("message", (e) => {
  if (e.origin !== window.location.origin) return;
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "srdb.metrics") {
    renderMetrics(msg.payload);
  }
});

els.btnDock?.addEventListener("click", () => {
  postToOpener({ type: "srdb.metrics.dock" });
  window.close();
});

postToOpener({ type: "srdb.metrics.ready" });
