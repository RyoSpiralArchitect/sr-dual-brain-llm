const $ = (id) => document.getElementById(id);

const els = {
  metricsSubtitle: $("metricsSubtitle"),
  mCoherence: $("mCoherence"),
  mTension: $("mTension"),
  mRouting: $("mRouting"),
  mAction: $("mAction"),
  mTemp: $("mTemp"),
  mLatency: $("mLatency"),
  telemetryRaw: $("telemetryRaw"),
  btnDock: $("btnDock"),
};

function renderMetrics(response) {
  const qid = response?.qid ?? "";
  const sessionId = response?.session_id ?? "";
  const stamp = new Date().toLocaleTimeString();
  const subtitle = [qid ? `qid ${qid}` : "", sessionId ? `session ${sessionId}` : "", stamp].filter(Boolean).join(" · ");
  els.metricsSubtitle.textContent = subtitle || "—";

  const telemetry = response?.telemetry ?? [];
  const metrics = response?.metrics ?? null;

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

  els.telemetryRaw.textContent = JSON.stringify(telemetry, null, 2);
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

