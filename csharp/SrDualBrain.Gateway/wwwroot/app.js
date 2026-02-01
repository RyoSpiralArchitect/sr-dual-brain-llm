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
  executiveMemo: $("executiveMemo"),
  executiveObserverMemo: $("executiveObserverMemo"),
  telemetryRaw: $("telemetryRaw"),
  dialogueFlowRaw: $("dialogueFlowRaw"),
};

let lastLlmSignature = null;
const sessionLlmSignature = new Map();

let metricsPopout = null;
let metricsPopoutPoll = null;
let lastMetricsPayload = null;
let pendingBlobs = [];

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
    metricsPopout.postMessage({ type: "srdb.metrics", payload }, window.location.origin);
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

function renderMetrics(response) {
  const qid = response?.qid ?? "";
  els.metricsSubtitle.textContent = qid ? `qid ${qid}` : "—";

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

  if (msg.type === "srdb.metrics.dock") {
    dockMetricsPopout();
  }
});

refreshHealth();
renderBlobs();
setInterval(refreshHealth, 10_000);
