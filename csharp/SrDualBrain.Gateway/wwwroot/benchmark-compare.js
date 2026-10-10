/* Local-only viewing of a saved System2 A/B report. */
(function () {
  "use strict";

  const fileInput = document.getElementById("reportFile");
  const select = document.getElementById("questionSelect");
  const status = document.getElementById("loadStatus");
  const theaters = {
    off: window.TraceTheater.create(document.getElementById("offTrace")),
    on: window.TraceTheater.create(document.getElementById("onTrace")),
  };
  let cases = { off: new Map(), on: new Map() };

  function showCase(id) {
    const off = cases.off.get(id);
    const on = cases.on.get(id);
    if (!off || !on) return;
    document.getElementById("questionId").textContent = id;
    document.getElementById("questionText").textContent = off.question || on.question || "";
    for (const [mode, item] of [["off", off], ["on", on]]) {
      const issues = item.initial_issues == null ? "—" : `${item.initial_issues} → ${item.final_issues ?? "—"}`;
      const validity = item.critic_validity || "unknown";
      document.getElementById(`${mode}Meta`).textContent =
        `Critic: ${validity} · Issues: ${issues} · ${Math.round(item.latency_ms || 0)} ms${item.error ? ` · Error: ${item.error}` : ""}`;
      document.getElementById(`${mode}Answer`).textContent = item.answer || "Full answer was not saved in this report.";
      theaters[mode].render(item.dialogue_flow || null, item.qid || `${mode}-${id}`,
        item.dialogue_flow ? "ready" : "unavailable");
    }
  }

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    select.disabled = true;
    status.textContent = "Reading local report…";
    try {
      const report = JSON.parse(await file.text());
      const off = report?.modes?.off?.cases;
      const on = report?.modes?.on?.cases;
      if (!Array.isArray(off) || !Array.isArray(on)) throw new Error("The report needs off and on cases.");
      const next = { off: new Map(off.map((item) => [String(item.id), item])),
        on: new Map(on.map((item) => [String(item.id), item])) };
      const ids = [...next.off.keys()].filter((id) => next.on.has(id));
      if (!ids.length) throw new Error("No matching question IDs were found.");
      cases = next;
      select.replaceChildren(...ids.map((id) => {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        return option;
      }));
      select.disabled = false;
      status.textContent = `${ids.length} paired questions · ${report.run_id || file.name}`;
      showCase(ids[0]);
    } catch (error) {
      status.textContent = `Unable to read report: ${error.message}`;
    }
  });
  select.addEventListener("change", () => showCase(select.value));
})();
