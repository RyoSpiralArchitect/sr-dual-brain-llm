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
  executiveMemo: $("executiveMemo"),
  telemetryRaw: $("telemetryRaw"),
  dialogueFlowRaw: $("dialogueFlowRaw"),
};

let lastLlmSignature = null;
const sessionLlmSignature = new Map();

let metricsPopout = null;
let metricsPopoutPoll = null;
let lastMetricsPayload = null;

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

function renderMetrics(response) {
  const qid = response?.qid ?? "";
  els.metricsSubtitle.textContent = qid ? `qid ${qid}` : "—";

  const telemetry = response?.telemetry ?? [];
  const dialogueFlow = response?.dialogue_flow ?? {};
  const metrics = response?.metrics ?? null;
  const executive = response?.executive ?? dialogueFlow?.executive ?? null;

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

  if (els.executiveMemo) {
    if (executive) {
      const memo = executive?.memo ?? "";
      const directives = executive?.directives ?? null;
      const meta = `source=${executive?.source ?? "?"} conf=${executive?.confidence ?? "?"} latency=${Math.round(executive?.latency_ms ?? 0)}ms`;
      const body = memo ? memo : "";
      const dir = directives ? `\n\n---\nDirectives:\n${JSON.stringify(directives, null, 2)}` : "";
      els.executiveMemo.textContent = `${body}\n\n---\n${meta}${dir}`.trim();
    } else {
      els.executiveMemo.textContent = "—";
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
  const wantExecutive = !!els.returnExecutive?.checked;
  const wantTelemetry = !!els.returnTelemetry.checked;
  const wantDialogueFlow = !!els.returnDialogueFlow.checked;
  const traceInline = includeTraceInline && !!els.traceInline?.checked;

  const body = {
    session_id: sessionId,
    question: questionText,
    leading_brain: leading,
    executive_mode: executiveMode,
    return_telemetry: wantTelemetry && traceInline,
    return_dialogue_flow: wantDialogueFlow && traceInline,
  };

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
  if (els.activeModules) els.activeModules.innerHTML = "";
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
  } catch (err) {
    appendBubble("assistant", `reset failed: ${err}`, { mono: true, right: "error" });
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
setInterval(refreshHealth, 10_000);
