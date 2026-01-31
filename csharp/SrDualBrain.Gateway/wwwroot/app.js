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
  returnTelemetry: $("returnTelemetry"),
  returnDialogueFlow: $("returnDialogueFlow"),
  metricsSubtitle: $("metricsSubtitle"),
  mCoherence: $("mCoherence"),
  mTension: $("mTension"),
  mRouting: $("mRouting"),
  mAction: $("mAction"),
  mTemp: $("mTemp"),
  mLatency: $("mLatency"),
  telemetryRaw: $("telemetryRaw"),
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
  const metrics = response?.metrics ?? null;

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

  els.telemetryRaw.textContent = JSON.stringify(telemetry, null, 2);

  publishMetricsToPopout(response);
}

async function callProcess(questionText) {
  const sessionId = (els.sessionId.value || "default").trim() || "default";
  const leading = (els.leadingBrain.value || "auto").trim() || "auto";

  const body = {
    session_id: sessionId,
    question: questionText,
    leading_brain: leading,
    return_telemetry: !!els.returnTelemetry.checked,
    return_dialogue_flow: !!els.returnDialogueFlow.checked,
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

  const res = await fetch("/v1/process", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  const payload = await res.json();
  sessionLlmSignature.set(sessionId, currentSig);
  lastLlmSignature = currentSig;
  return payload;
}

async function onSend() {
  const text = (els.question.value || "").trim();
  if (!text) return;

  appendBubble("user", text);
  els.question.value = "";
  const placeholder = appendBubble("assistant", "thinking…", { mono: true });

  setBusy(true);
  try {
    const result = await callProcess(text);
    placeholder.querySelector(".bubble__content").textContent = result.answer || "(no answer)";
    // Keep chat bubbles clean; qid/metrics are shown in the side panel.
    placeholder.querySelector(".bubble__meta > div:last-child").textContent = "";
    renderMetrics(result);
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
