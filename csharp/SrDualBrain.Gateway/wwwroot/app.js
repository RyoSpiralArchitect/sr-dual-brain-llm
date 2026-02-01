const $ = (id) => document.getElementById(id);

const els = {
  status: $("status"),
  chatLog: $("chatLog"),
  composer: $("composer"),
  question: $("question"),
  btnSend: $("btnSend"),
  btnPopMetrics: $("btnPopMetrics"),
  btnClear: $("btnClear"),
  btnReset: $("btnReset"),
  sessionId: $("sessionId"),
  leadingBrain: $("leadingBrain"),
  llmProvider: $("llmProvider"),
  llmModel: $("llmModel"),
  llmMaxOutputTokens: $("llmMaxOutputTokens"),
  executiveMode: $("executiveMode"),
  executiveObserverMode: $("executiveObserverMode"),
  blobFile: $("blobFile"),
  btnUploadBlob: $("btnUploadBlob"),
  blobChips: $("blobChips"),
  useStreaming: $("useStreaming"),
  returnExecutive: $("returnExecutive"),
  returnTelemetry: $("returnTelemetry"),
  returnDialogueFlow: $("returnDialogueFlow"),
  traceInline: $("traceInline"),
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
  brainHistory: $("brainHistory"),
  executiveMemo: $("executiveMemo"),
  executiveObserverMemo: $("executiveObserverMemo"),
  telemetryRaw: $("telemetryRaw"),
  dialogueFlowRaw: $("dialogueFlowRaw"),
  btnExportTrace: $("btnExportTrace"),
  btnExportTelemetry: $("btnExportTelemetry"),
  btnExportDialogueFlow: $("btnExportDialogueFlow"),
  btnExportBrainHistory: $("btnExportBrainHistory"),
  btnExportChat: $("btnExportChat"),
};

let lastLlmSignature = null;
const sessionLlmSignature = new Map();

let metricsPopout = null;
let metricsPopoutPoll = null;
let lastMetricsPayload = null;
let pendingBlobs = [];
const brainHistoryBySession = new Map();
const brainHistoryLimit = 24;
const selectedQidBySession = new Map();

function isMetricsPopoutOpen() {
  return !!metricsPopout && !metricsPopout.closed;
}

function setMetricsPopoutUi(isOpen) {
  if (isOpen) {
    document.body.classList.add("metrics-popout");
    if (els.btnPopMetrics) els.btnPopMetrics.textContent = "Dock metrics";
  } else {
    document.body.classList.remove("metrics-popout");
    if (els.btnPopMetrics) els.btnPopMetrics.textContent = "Pop out metrics";
  }
}

function publishMetricsToPopout(payload) {
  lastMetricsPayload = payload;
  if (!payload) return;
  if (!isMetricsPopoutOpen()) return;
  try {
    const sessionId = payload?.session_id ?? "default";
    const history = brainHistoryBySession.get(sessionId) ?? [];
    const clientState = {
      brain_history: history.map((h) => ({
        qid: h.qid ?? "",
        ts: h.ts ?? 0,
        values: h.values ?? {},
      })),
    };
    metricsPopout.postMessage(
      { type: "srdb.metrics", payload: { ...payload, _client: clientState } },
      window.location.origin,
    );
  } catch {
    // ignore
  }
}

function dockMetricsPopout({ closeWindow } = { closeWindow: true }) {
  if (metricsPopoutPoll) {
    clearInterval(metricsPopoutPoll);
    metricsPopoutPoll = null;
  }
  if (closeWindow && metricsPopout && !metricsPopout.closed) {
    try {
      metricsPopout.close();
    } catch {
      // ignore
    }
  }
  metricsPopout = null;
  setMetricsPopoutUi(false);
}

function openMetricsPopout() {
  if (isMetricsPopoutOpen()) return;

  const width = 520;
  const height = 900;
  const left = Math.max(0, window.screenX + window.outerWidth - width - 20);
  const top = Math.max(0, window.screenY + 60);

  metricsPopout = window.open(
    "/metrics.html",
    "srdb_metrics",
    `popup=yes,width=${width},height=${height},left=${left},top=${top}`,
  );

  if (!metricsPopout) {
    // likely blocked by the browser
    setStatus("warn", "Pop-up blocked (allow pop-ups to open metrics window)");
    return;
  }

  setMetricsPopoutUi(true);

  if (metricsPopoutPoll) clearInterval(metricsPopoutPoll);
  metricsPopoutPoll = setInterval(() => {
    if (!isMetricsPopoutOpen()) dockMetricsPopout({ closeWindow: false });
  }, 500);

  publishMetricsToPopout(lastMetricsPayload);
}

function setStatus(kind, text) {
  const cls =
    kind === "good"
      ? "pill pill--good"
      : kind === "warn"
        ? "pill pill--warn"
        : kind === "bad"
          ? "pill pill--bad"
          : "pill pill--muted";
  els.status.innerHTML = `<span class="${cls}">${escapeHtml(text)}</span>`;
}

function renderBlobs() {
  if (!els.blobChips) return;
  els.blobChips.innerHTML = "";
  if (!pendingBlobs.length) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = "attachments: —";
    els.blobChips.appendChild(chip);
    return;
  }
  for (const blob of pendingBlobs) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    const shortId = String(blob.blob_id || "").slice(0, 8);
    chip.textContent = `${blob.content_type || "blob"}:${shortId}`;
    chip.title = `${blob.file_name || ""} (${blob.size_bytes || blob.size || "?"} bytes)`;
    chip.addEventListener("click", () => {
      pendingBlobs = pendingBlobs.filter((b) => b.blob_id !== blob.blob_id);
      renderBlobs();
    });
    els.blobChips.appendChild(chip);
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

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

function openDetailsFor(ids) {
  for (const id of ids) {
    const el = $(id);
    const details = el?.closest?.("details");
    if (details) details.open = true;
  }
}

function collectChatFromDom() {
  const bubbles = Array.from(els.chatLog?.querySelectorAll?.(".bubble") ?? []);
  return bubbles.map((bubble) => {
    const isUser = bubble.classList.contains("bubble--user");
    const meta = bubble.querySelector(".bubble__meta")?.textContent ?? "";
    const content = bubble.querySelector(".bubble__content")?.textContent ?? "";
    return {
      role: isUser ? "user" : "assistant",
      meta: meta.trim(),
      content: content,
    };
  });
}

function appendBubble(role, content, meta = {}) {
  const el = document.createElement("div");
  el.className = `bubble ${role === "user" ? "bubble--user" : "bubble--assistant"}`;

  const left = role === "user" ? "you" : "assistant";
  const right = meta.right ?? "";
  const ts = meta.ts ?? new Date().toLocaleTimeString();
  const mono = meta.mono ? "mono" : "";

  el.innerHTML = `
    <div class="bubble__meta">
      <div>${escapeHtml(left)} · ${escapeHtml(ts)}</div>
      <div>${escapeHtml(right)}</div>
    </div>
    <div class="bubble__content ${mono}">${escapeHtml(content)}</div>
  `;

  els.chatLog.appendChild(el);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
  return el;
}

function setBusy(isBusy) {
  els.btnSend.disabled = isBusy;
  els.btnReset.disabled = isBusy;
  els.btnClear.disabled = isBusy;
  els.question.disabled = isBusy;
}

function llmSignature() {
  const provider = (els.llmProvider.value || "").trim();
  const model = (els.llmModel.value || "").trim();
  const maxOutputTokens = (els.llmMaxOutputTokens?.value || "").trim();
  if (!provider || !model) return "default";
  return `${provider}:${model}:${maxOutputTokens || ""}`;
}

async function refreshHealth() {
  try {
    const res = await fetch("/v1/health", { cache: "no-store" });
    if (!res.ok) {
      setStatus("bad", `health ${res.status}`);
      return;
    }
    const payload = await res.json();
    const pg = payload?.engine?.postgres;
    if (pg?.enabled) {
      setStatus("good", "gateway ok · postgres on");
    } else if (pg?.error) {
      setStatus("warn", "gateway ok · postgres off (see /v1/health)");
    } else {
      setStatus("good", "gateway ok");
    }
  } catch (err) {
    setStatus("bad", `offline: ${err}`);
  }
}

function lastEvent(telemetry, name) {
  if (!Array.isArray(telemetry)) return null;
  for (let i = telemetry.length - 1; i >= 0; i--) {
    const ev = telemetry[i];
    if (ev && ev.event === name) return ev;
  }
  return null;
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
      cell.addEventListener("click", () => jumpToTrace(sessionId, snap?.qid));
      cells.appendChild(cell);
    }
    rowEl.appendChild(cells);
    els.brainHistory.appendChild(rowEl);
  }
}

function renderMetrics(response, opts = {}) {
  const renderSource = opts?.source ?? "live";
  const qid = response?.qid ?? "";
  els.metricsSubtitle.textContent = qid ? `qid ${qid}${renderSource === "trace" ? " · trace" : ""}` : "—";

  const telemetry = response?.telemetry ?? [];
  const dialogueFlow = response?.dialogue_flow ?? {};
  const metrics = response?.metrics ?? null;
  const executive = response?.executive ?? dialogueFlow?.executive ?? null;
  const executiveObserver = response?.executive_observer ?? dialogueFlow?.executive_observer ?? null;

  let combined = null;
  let tension = null;
  let routing = null;
  let action = null;
  let temp = null;
  let leading = null;
  let latency = null;

  if (metrics) {
    combined = metrics?.coherence?.combined ?? null;
    tension = metrics?.coherence?.tension ?? null;
    routing = metrics?.coherence?.mode ?? null;
    action = metrics?.policy?.action ?? null;
    temp = metrics?.policy?.temperature ?? null;
    leading = metrics?.leading ?? null;
    latency = metrics?.latency_ms ?? null;
  } else {
    const coherenceEv = lastEvent(telemetry, "coherence_signal");
    const policyEv = lastEvent(telemetry, "policy_decision");
    const leadEv = lastEvent(telemetry, "leading_brain");
    const completeEv = lastEvent(telemetry, "interaction_complete");

    const signal = coherenceEv?.signal ?? null;
    combined = signal?.combined ?? null;
    tension = signal?.tension ?? null;
    routing = signal?.mode ?? null;

    action = policyEv?.action ?? null;
    temp = policyEv?.temperature ?? null;
    leading = leadEv?.leading ?? null;
    latency = completeEv?.latency_ms ?? null;
  }

  els.mCoherence.textContent = combined == null ? "—" : combined.toFixed(3);
  els.mTension.textContent = tension == null ? "—" : tension.toFixed(3);
  els.mRouting.textContent = routing ? String(routing) : (leading ? String(leading) : "—");
  els.mAction.textContent = action == null ? "—" : String(action);
  els.mTemp.textContent = temp == null ? "—" : Number(temp).toFixed(2);
  els.mLatency.textContent = latency == null ? "—" : `${Math.round(latency)}ms`;

  const modules = metrics?.modules?.active ?? [];
  if (els.activeModules) {
    els.activeModules.innerHTML = "";
    const limit = 14;
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
  const sessionId = response?.session_id ?? "default";
  selectedQidBySession.set(sessionId, qid || "");
  if (renderSource !== "trace") {
    upsertBrainHistory(sessionId, snapshotBrain(metrics, { qid }));
  }
  renderBrainHistory(sessionId);

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

  publishMetricsToPopout(response);
}

async function buildProcessBody(questionText, { includeTraceInline }) {
  const sessionId = (els.sessionId.value || "default").trim() || "default";
  const leading = (els.leadingBrain.value || "auto").trim() || "auto";
  const executiveMode = (els.executiveMode?.value || "off").trim() || "off";
  const executiveObserverMode = (els.executiveObserverMode?.value || "off").trim() || "off";
  const wantExecutive = !!els.returnExecutive?.checked;
  const wantTelemetry = !!els.returnTelemetry.checked;
  const wantDialogueFlow = !!els.returnDialogueFlow.checked;
  const traceInline = includeTraceInline && !!els.traceInline?.checked;

  const body = {
    session_id: sessionId,
    question: questionText,
    leading_brain: leading,
    executive_mode: executiveMode,
    executive_observer_mode: executiveObserverMode,
    return_executive: wantExecutive,
    return_telemetry: wantTelemetry && traceInline,
    return_dialogue_flow: wantDialogueFlow && traceInline,
  };
  if (pendingBlobs.length) {
    body.attachments = pendingBlobs.map((b) => ({
      blob_id: b.blob_id,
      content_type: b.content_type || null,
      file_name: b.file_name || null,
      size_bytes: b.size_bytes || null,
    }));
  }

  const provider = (els.llmProvider.value || "").trim();
  const model = (els.llmModel.value || "").trim();
  if (provider && model) {
    body.llm = { provider, model };
    const maxOutRaw = (els.llmMaxOutputTokens?.value || "").trim();
    const maxOut = Number.parseInt(maxOutRaw, 10);
    if (Number.isFinite(maxOut) && maxOut > 0) body.llm.max_output_tokens = maxOut;
  }

  const currentSig = llmSignature();
  const previousSig = sessionLlmSignature.get(sessionId);
  if ((previousSig && previousSig !== currentSig) || (!previousSig && currentSig !== "default")) {
    // Session LLM config is sticky; reset to make switching deterministic.
    await fetch("/v1/reset", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
  }

  return { sessionId, wantExecutive, wantTelemetry, wantDialogueFlow, traceInline, body, llmSig: currentSig };
}

async function callProcess(body) {
  const res = await fetch("/v1/process", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return await res.json();
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

function mergeTrace(result, trace) {
  if (!trace || !trace.found) return result;
  return {
    ...result,
    telemetry: trace.telemetry ?? result.telemetry,
    dialogue_flow: trace.dialogue_flow ?? result.dialogue_flow,
    executive: trace.executive ?? result.executive,
    executive_observer: trace.executive_observer ?? result.executive_observer,
  };
}

function normaliseTraceToPayload(trace, { sessionIdFallback } = {}) {
  const sid = trace?.session_id ?? sessionIdFallback ?? "default";
  return {
    qid: trace?.qid ?? "",
    session_id: sid,
    metrics: trace?.metrics ?? {},
    telemetry: trace?.telemetry ?? [],
    dialogue_flow: trace?.dialogue_flow ?? {},
    executive: trace?.executive ?? null,
    executive_observer: trace?.executive_observer ?? null,
    ts: trace?.ts ?? null,
  };
}

async function jumpToTrace(sessionId, qid) {
  const sid = (sessionId || "default").trim() || "default";
  const id = String(qid || "").trim();
  if (!id) return;

  try {
    const trace = await fetchTrace(sid, id, {
      includeTelemetry: true,
      includeDialogueFlow: true,
      includeExecutive: true,
    });
    if (!trace?.found) {
      throw new Error("trace not found (engine may have restarted)");
    }
    renderMetrics(normaliseTraceToPayload(trace, { sessionIdFallback: sid }), { source: "trace" });
    openDetailsFor(["executiveMemo", "executiveObserverMemo", "telemetryRaw", "dialogueFlowRaw"]);
  } catch (err) {
    setStatus("warn", `trace load failed: ${String(err)}`);
  }
}

async function getFullTracePayload(payload) {
  const base = payload && typeof payload === "object" ? payload : null;
  const sessionId = (base?.session_id || "default").trim() || "default";
  const qid = String(base?.qid || "").trim();
  if (!qid) {
    throw new Error("No qid available yet.");
  }

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
  return {
    ...base,
    ...normaliseTraceToPayload(trace, { sessionIdFallback: sessionId }),
    _client: undefined,
  };
}

async function callProcessStream(body, { onDelta, onFinal, onReset }) {
  const res = await fetch("/v1/process/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  if (!res.body) {
    throw new Error("Streaming response body not available.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;

  function handleEventBlock(block) {
    const lines = block.split("\n");
    let eventName = "message";
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    const dataText = dataLines.join("\n");
    if (!dataText) return;
    let payload = null;
    try {
      payload = JSON.parse(dataText);
    } catch {
      payload = { text: dataText };
    }

    if (eventName === "delta") {
      const chunk = payload?.text ?? "";
      if (chunk && onDelta) onDelta(chunk);
    } else if (eventName === "reset") {
      if (onReset) onReset(payload);
    } else if (eventName === "final") {
      finalResult = payload;
      if (onFinal) onFinal(payload);
    } else if (eventName === "error") {
      const msg = payload?.message ?? "unknown error";
      throw new Error(msg);
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, idx).trimEnd();
      buffer = buffer.slice(idx + 2);
      if (block) handleEventBlock(block);
    }
  }

  return finalResult;
}

async function onSend() {
  const text = (els.question.value || "").trim();
  if (!text) return;

  appendBubble("user", text);
  els.question.value = "";
  const placeholder = appendBubble("assistant", "thinking…", { mono: true });

  setBusy(true);
  try {
    const wantStream = !!els.useStreaming?.checked;
    const req = await buildProcessBody(text, { includeTraceInline: !wantStream });

    let result;
    if (wantStream) {
      let streamed = "";
      result = await callProcessStream(req.body, {
        onDelta: (chunk) => {
          streamed += chunk;
          placeholder.querySelector(".bubble__content").textContent = streamed;
        },
        onReset: () => {
          streamed = "";
          placeholder.querySelector(".bubble__content").textContent = "";
        },
        onFinal: (payload) => {
          if (payload?.answer) {
            placeholder.querySelector(".bubble__content").textContent = payload.answer;
          }
        },
      });
      if (!result) {
        result = { qid: "", answer: streamed, session_id: req.sessionId, metrics: null };
      }
    } else {
      result = await callProcess(req.body);
      placeholder.querySelector(".bubble__content").textContent = result.answer || "(no answer)";
    }

    sessionLlmSignature.set(req.sessionId, req.llmSig);
    lastLlmSignature = req.llmSig;

    // Keep chat bubbles clean; qid/metrics are shown in the side panel.
    placeholder.querySelector(".bubble__meta > div:last-child").textContent = "";
    renderMetrics(result);

    const hasExecutiveInline = !!(result?.executive || result?.dialogue_flow?.executive);
    if (((req.wantTelemetry || req.wantDialogueFlow) && !req.traceInline) || (req.wantExecutive && !hasExecutiveInline)) {
      try {
        const trace = await fetchTrace(req.sessionId, result.qid, {
          includeTelemetry: req.wantTelemetry,
          includeDialogueFlow: req.wantDialogueFlow,
          includeExecutive: req.wantExecutive,
        });
        const merged = mergeTrace(result, trace);
        renderMetrics(merged);
      } catch {
        // ignore trace fetch failures
      }
    }
  } catch (err) {
    placeholder.querySelector(".bubble__content").textContent = String(err);
    placeholder.classList.add("bubble--error");
  } finally {
    setBusy(false);
    els.question.focus();
  }
}

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  await onSend();
});

els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    onSend();
  }
});

els.btnClear.addEventListener("click", () => {
  els.chatLog.innerHTML = "";
  els.telemetryRaw.textContent = "{}";
  if (els.dialogueFlowRaw) els.dialogueFlowRaw.textContent = "{}";
  if (els.executiveMemo) els.executiveMemo.textContent = "—";
  if (els.executiveObserverMemo) els.executiveObserverMemo.textContent = "—";
  if (els.activeModules) els.activeModules.innerHTML = "";
  if (els.modulePath) els.modulePath.innerHTML = "";
  if (els.brainActivity) els.brainActivity.innerHTML = "";
  if (els.brainHistory) els.brainHistory.innerHTML = "";
  brainHistoryBySession.clear();
  selectedQidBySession.clear();
  els.metricsSubtitle.textContent = "—";
  els.mCoherence.textContent = "—";
  els.mTension.textContent = "—";
  els.mRouting.textContent = "—";
  els.mAction.textContent = "—";
  els.mTemp.textContent = "—";
  els.mLatency.textContent = "—";
});

els.btnReset.addEventListener("click", async () => {
  const sessionId = (els.sessionId.value || "default").trim() || "default";
  setBusy(true);
  try {
    await fetch("/v1/reset", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    appendBubble("assistant", `session reset: ${sessionId}`, { mono: true, right: "reset" });
    pendingBlobs = [];
    renderBlobs();
    brainHistoryBySession.delete(sessionId);
    selectedQidBySession.delete(sessionId);
    if (els.brainHistory) els.brainHistory.innerHTML = "";
  } catch (err) {
    appendBubble("assistant", `reset failed: ${err}`, { mono: true, right: "error" });
  } finally {
    setBusy(false);
  }
});

els.btnUploadBlob?.addEventListener("click", async () => {
  const file = els.blobFile?.files?.[0];
  if (!file) return;
  const sessionId = (els.sessionId.value || "default").trim() || "default";

  setBusy(true);
  try {
    const form = new FormData();
    form.append("session_id", sessionId);
    form.append("file", file, file.name);

    const res = await fetch("/v1/blobs", { method: "POST", body: form });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const payload = await res.json();
    pendingBlobs.push(payload);
    renderBlobs();
    els.blobFile.value = "";
  } catch (err) {
    appendBubble("assistant", `upload failed: ${err}`, { mono: true, right: "error" });
  } finally {
    setBusy(false);
  }
});

els.btnPopMetrics?.addEventListener("click", () => {
  if (isMetricsPopoutOpen()) {
    dockMetricsPopout();
  } else {
    openMetricsPopout();
  }
});

window.addEventListener("message", (e) => {
  if (e.origin !== window.location.origin) return;
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "srdb.metrics.ready") {
    publishMetricsToPopout(lastMetricsPayload);
    return;
  }

  if (msg.type === "srdb.trace.jump") {
    const payload = msg.payload ?? {};
    const sessionId = payload?.session_id ?? (els.sessionId.value || "default");
    const qid = payload?.qid ?? "";
    jumpToTrace(sessionId, qid);
    return;
  }

  if (msg.type === "srdb.metrics.dock") {
    dockMetricsPopout();
  }
});

els.btnExportTrace?.addEventListener("click", async () => {
  try {
    const full = await getFullTracePayload(lastMetricsPayload);
    const sid = (full?.session_id || "default").trim() || "default";
    const qid = String(full?.qid || "").slice(0, 8) || "trace";
    downloadJson(`srdb_trace_${sid}_${qid}_${nowStamp()}.json`, full);
  } catch (err) {
    setStatus("warn", `export trace failed: ${String(err)}`);
  }
});

els.btnExportTelemetry?.addEventListener("click", async () => {
  try {
    const full = await getFullTracePayload(lastMetricsPayload);
    const sid = (full?.session_id || "default").trim() || "default";
    const qid = String(full?.qid || "").slice(0, 8) || "trace";
    downloadJson(`srdb_telemetry_${sid}_${qid}_${nowStamp()}.json`, full?.telemetry ?? []);
  } catch (err) {
    setStatus("warn", `export telemetry failed: ${String(err)}`);
  }
});

els.btnExportDialogueFlow?.addEventListener("click", async () => {
  try {
    const full = await getFullTracePayload(lastMetricsPayload);
    const sid = (full?.session_id || "default").trim() || "default";
    const qid = String(full?.qid || "").slice(0, 8) || "trace";
    downloadJson(`srdb_dialogue_flow_${sid}_${qid}_${nowStamp()}.json`, full?.dialogue_flow ?? {});
  } catch (err) {
    setStatus("warn", `export dialogue failed: ${String(err)}`);
  }
});

els.btnExportBrainHistory?.addEventListener("click", () => {
  try {
    const sessionId = (els.sessionId.value || "default").trim() || "default";
    const history = getBrainHistory(sessionId);
    downloadJson(`srdb_brain_history_${sessionId}_${nowStamp()}.json`, {
      session_id: sessionId,
      limit: brainHistoryLimit,
      items: history,
    });
  } catch (err) {
    setStatus("warn", `export brain history failed: ${String(err)}`);
  }
});

els.btnExportChat?.addEventListener("click", () => {
  try {
    const sessionId = (els.sessionId.value || "default").trim() || "default";
    const chat = collectChatFromDom();
    downloadJson(`srdb_chat_${sessionId}_${nowStamp()}.json`, { session_id: sessionId, messages: chat });
  } catch (err) {
    setStatus("warn", `export chat failed: ${String(err)}`);
  }
});

refreshHealth();
renderBlobs();
setInterval(refreshHealth, 10_000);
