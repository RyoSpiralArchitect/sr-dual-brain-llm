/* Blind human ratings: same-origin private server, no model or reveal data. */
(async function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const dimensions = [
    ["correctness", "正しさ", ["0 · 誤っている", "1 · 一部正しい", "2 · 正しい"]],
    ["completeness", "十分さ", ["0 · 中心的な要求を満たさない", "1 · 一部不足している", "2 · 要求を満たしている"]],
    ["unsupported_claims", "根拠のない主張", ["0 · ない", "1 · 小さなものがある", "2 · 重大なものがある"]],
  ];
  let packets = [], rows = new Map(), index = 0, token = "", locked = false, dirty = false, busy = false;

  function status(message, error = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", error);
  }
  function blank(id) {
    const scores = () => Object.fromEntries(dimensions.map(([key]) => [key, null]));
    return { packet_id: id, answer_a: scores(), answer_b: scores(), preferred: "", rationale: "" };
  }
  function complete(row) {
    if (row.preferred === "uncertain") return !!row.rationale.trim();
    return ["a", "b", "tie"].includes(row.preferred)
      && ["answer_a", "answer_b"].every((side) => dimensions.every(([key]) => row[side][key] !== null))
      && (row.preferred === "tie" || !!row.rationale.trim());
  }
  function progress() {
    const n = [...rows.values()].filter(complete).length;
    $("progress").textContent = `${n} / ${packets.length} 組を採点${locked ? " · 確定済み" : ""}`;
    $("lock").disabled = locked || busy || n !== packets.length;
    for (let i = 0; i < packets.length; i++) {
      $("packetSelect").options[i].textContent = `${complete(rows.get(packets[i].packet_id)) ? "✓" : "○"} ${i + 1} / ${packets.length}`;
    }
  }
  function capture() {
    if (!packets.length || locked) return;
    const row = rows.get(packets[index].packet_id);
    for (const side of ["a", "b"]) for (const [key] of dimensions) {
      const value = $(`${side}-${key}`).value;
      row[`answer_${side}`][key] = value === "" ? null : Number(value);
    }
    row.preferred = $("preferred").value;
    row.rationale = $("rationale").value;
    dirty = true;
    status("変更があります。「途中保存」または「保存して次へ」で保存できます。");
    progress();
  }
  function show() {
    const p = packets[index], r = rows.get(p.packet_id);
    $("questionHeading").textContent = `問題 · ${index + 1} / ${packets.length}`;
    $("questionText").textContent = p.question;
    $("answerA").textContent = p.answer_a;
    $("answerB").textContent = p.answer_b;
    for (const side of ["a", "b"]) for (const [key] of dimensions) $(`${side}-${key}`).value = r[`answer_${side}`][key] ?? "";
    $("preferred").value = r.preferred;
    $("rationale").value = r.rationale;
    $("packetSelect").value = String(index);
    $("previous").disabled = index === 0;
    $("next").disabled = index === packets.length - 1;
    $("next").textContent = locked ? "次の組" : "保存して次へ";
    progress();
  }
  async function save(final = false) {
    if (locked) return true;
    if (busy) return false;
    busy = true;
    for (const element of document.querySelectorAll(".review input, .review select, .review textarea, .review button")) element.disabled = true;
    progress();
    try {
      const payload = { reviewer: $("reviewer").value.trim(), exposure: $("exposure").value,
        ratings: packets.map((p) => rows.get(p.packet_id)) };
      const response = await fetch(final ? "/api/lock" : "/api/draft", {
        method: "POST", headers: { "Content-Type": "application/json", "X-Review-Token": token }, body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "保存できませんでした");
      dirty = false;
      if (final) {
        locked = true;
        for (const element of document.querySelectorAll(".score-fields select, .verdict select, .verdict textarea, .review-settings input, .review-settings select")) element.disabled = true;
        $("export").hidden = false;
      }
      status(final ? "採点を確定しました。このチャットで「採点完了」と伝えてください。" : "この端末に保存しました。いつでも再開できます。");
      return true;
    } catch (error) {
      status(`保存できませんでした。評価者名と入力を確認してください。${error.message}`, true);
      return false;
    } finally {
      busy = false;
      for (const element of document.querySelectorAll(".score-fields select, .verdict select, .verdict textarea, .review-settings input, .review-settings select")) element.disabled = locked;
      $("packetSelect").disabled = false;
      $("previous").disabled = index === 0;
      $("next").disabled = index === packets.length - 1;
      $("save").disabled = locked;
      progress();
    }
  }
  async function navigate(next) {
    if (busy) { $("packetSelect").value = String(index); return; }
    if (dirty && !(await save())) { $("packetSelect").value = String(index); return; }
    index = next;
    show();
    $("questionHeading").scrollIntoView({ block: "start" });
  }
  try {
    const response = await fetch("/api/session");
    if (!response.ok) throw new Error("採点サーバーに接続できません");
    const session = await response.json();
    packets = session.packets;
    token = session.token;
    rows = new Map(packets.map((p) => [p.packet_id, blank(p.packet_id)]));
    for (const r of session.review.ratings) rows.set(r.packet_id, r);
    locked = session.review.state === "locked";
    $("reviewer").value = session.review.reviewer;
    $("exposure").value = session.review.exposure;
    $("packetSelect").replaceChildren(...packets.map((_, i) => new Option(String(i + 1), String(i))));
    for (const side of ["a", "b"]) for (const [key, title, choices] of dimensions) {
      const label = document.createElement("label"), select = document.createElement("select");
      label.textContent = title;
      select.id = `${side}-${key}`;
      select.setAttribute("aria-label", `回答${side.toUpperCase()} ${title}`);
      select.append(new Option("選んでください", ""), ...choices.map((text, i) => new Option(text, String(i))));
      select.disabled = locked;
      select.addEventListener("change", capture);
      label.append(select);
      $(side === "a" ? "scoresA" : "scoresB").append(label);
    }
    for (const id of ["reviewer", "exposure", "preferred", "rationale"]) {
      $(id).disabled = locked;
      $(id).addEventListener(id === "rationale" || id === "reviewer" ? "input" : "change", capture);
    }
    $("previous").addEventListener("click", () => navigate(index - 1));
    $("next").addEventListener("click", () => navigate(index + 1));
    $("packetSelect").addEventListener("change", () => navigate(Number($("packetSelect").value)));
    $("save").addEventListener("click", () => save());
    $("lock").addEventListener("click", () => save(true));
    $("save").disabled = locked;
    $("export").hidden = !locked;
    window.addEventListener("beforeunload", (event) => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
    $("workspace").hidden = false;
    index = Math.max(0, packets.findIndex((p) => !complete(rows.get(p.packet_id))));
    show();
    status(locked ? "採点は確定済みです。このチャットで「採点完了」と伝えてください。" : "準備できました。回答の内容だけを見て評価してください。");
  } catch (error) {
    status(`読み込めませんでした。${error.message}`, true);
  }
})();
