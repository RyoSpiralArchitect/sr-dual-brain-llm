const $ = (id) => document.getElementById(id);

const els = {
  status: $("status"),
  chatLog: $("chatLog"),
  composer: $("composer"),
  question: $("question"),
  btnSend: $("btnSend"),
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
  const coherenceEv = lastEvent(telemetry, "coherence_signal");
  const policyEv = lastEvent(telemetry, "policy_decision");
  const leadEv = lastEvent(telemetry, "leading_brain");
  const completeEv = lastEvent(telemetry, "interaction_complete");

  const signal = coherenceEv?.signal ?? null;
  const combined = signal?.combined;
  const tension = signal?.tension;
  const routing = signal?.mode;

  const action = policyEv?.action;
  const temp = policyEv?.temperature;
  const leading = leadEv?.leading;
  const latency = completeEv?.latency_ms;

  els.mCoherence.textContent = combined == null ? "—" : combined.toFixed(3);
  els.mTension.textContent = tension == null ? "—" : tension.toFixed(3);
  els.mRouting.textContent = routing ? String(routing) : (leading ? String(leading) : "—");
  els.mAction.textContent = action == null ? "—" : String(action);
  els.mTemp.textContent = temp == null ? "—" : Number(temp).toFixed(2);
  els.mLatency.textContent = latency == null ? "—" : `${Math.round(latency)}ms`;

  els.telemetryRaw.textContent = JSON.stringify(telemetry, null, 2);
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
    placeholder.querySelector(".bubble__meta > div:last-child").textContent = result.qid ? `qid ${result.qid}` : "";
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

refreshHealth();
setInterval(refreshHealth, 10_000);
