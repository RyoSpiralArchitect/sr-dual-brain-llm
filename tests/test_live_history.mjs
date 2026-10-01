import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { createContext, runInContext } from "node:vm";

const root = new URL("../csharp/SrDualBrain.Gateway/wwwroot/", import.meta.url);
const selectionSource = readFileSync(new URL("trace-selection.js", root), "utf8");

function deferred() {
  let resolve;
  const promise = new Promise((res) => { resolve = res; });
  return { promise, resolve };
}

class FakeElement {
  constructor() {
    this.children = [];
    this.listeners = new Map();
    this.style = { setProperty() {} };
    this.className = "";
    this.classList = {
      add: (name) => { this.className += ` ${name}`; },
      remove: (name) => { this.className = this.className.split(" ").filter((part) => part !== name).join(" "); },
      contains: (name) => this.className.split(" ").includes(name),
    };
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this._text = "";
    this.selectors = new Map();
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  set innerHTML(value) {
    this._text = "";
    this.children = [];
    // Only chat bubbles use selectors in these interaction tests.
    if (value.includes('class="bubble__meta"')) {
      this.selectors.set(".bubble__content", new FakeElement());
      this.selectors.set(".bubble__meta > div:last-child", new FakeElement());
    }
  }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(name, listener) { this.listeners.set(name, listener); }
  querySelector(selector) { return this.selectors.get(selector) ?? null; }
  focus() {}
  click() { if (!this.disabled) return this.listeners.get("click")?.(); }
}

function findAll(root, className) {
  return [
    ...(root.classList.contains(className) ? [root] : []),
    ...root.children.flatMap((child) => findAll(child, className)),
  ];
}

function setup(script = "app.js") {
  const elements = new Map();
  const listeners = new Map();
  const requests = [];
  const messages = [];
  const renders = [];
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new FakeElement());
    return elements.get(id);
  };
  const origin = "http://localhost";
  const window = {
    location: { origin },
    addEventListener: (name, callback) => listeners.set(name, callback),
    TraceTheater: { create: () => ({ render: (...args) => renders.push(args) }) },
  };
  const context = createContext({
    window, URLSearchParams, TextDecoder, console,
    document: { getElementById: element, createElement: () => new FakeElement(), body: new FakeElement() },
    setInterval: () => 1,
    clearInterval() {},
    fetch: (url, options) => {
      if (url === "/v1/health") return Promise.resolve({ ok: true, json: async () => ({}) });
      const response = deferred();
      requests.push({ url, options, response });
      return response.promise;
    },
  });
  runInContext(selectionSource, context);
  runInContext(readFileSync(new URL(script, root), "utf8"), context);
  if (script === "app.js") {
    context.popout = { closed: false, postMessage: (message) => messages.push(structuredClone(message)) };
    runInContext("metricsPopout = popout", context);
  }
  return {
    element, requests, messages, renders,
    run: (code) => runInContext(code, context),
    receive: (data) => listeners.get("message")({ origin, data }),
  };
}

function payload(qid, { steps = true, session = "default" } = {}) {
  return {
    qid, session_id: session, answer: `answer ${qid}`,
    metrics: { brain: { prefrontal: { metric: 0.75 } }, modules: { active: ["LeftBrainModel"] } },
    ...(steps ? { dialogue_flow: { steps: [{ role: "left", content: qid }] } } : {}),
  };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

async function complete(request, value, streaming = false) {
  if (streaming) {
    let sent = false;
    request.response.resolve({
      ok: true,
      body: { getReader: () => ({ read: async () => {
        if (sent) return { done: true };
        sent = true;
        return { done: false, value: new TextEncoder().encode(`event: final\ndata: ${JSON.stringify(value)}\n\n`) };
      } }) },
    });
  } else {
    request.response.resolve({ ok: true, json: async () => value });
  }
  await flush();
}

async function send(app, value, streaming = false) {
  app.element("question").value = `question ${value.qid}`;
  app.element("useStreaming").checked = streaming;
  const task = app.run("onSend()");
  await flush();
  await complete(app.requests.at(-1), value, streaming);
  await task;
}

function historyQids(app, kind) {
  return Array.from(app.run(`get${kind}History("default")`), (item) => item.qid);
}

function assertHistory(app, qids) {
  assert.deepEqual(historyQids(app, "Brain"), qids);
  assert.deepEqual(historyQids(app, "Module"), qids);
  assert.equal(findAll(app.element("brainHistory"), "bh__cells")[0].children.length, qids.length);
  assert.equal(findAll(app.element("moduleHistory"), "mh__cells")[0].children.length, qids.length);
}

for (const streaming of [false, true]) {
  for (const resolveHistoryFirst of [false, true]) {
    test(`completed ${streaming ? "streaming" : "JSON"} turn survives a ${resolveHistoryFirst ? "loaded" : "pending"} history selection`, async () => {
      const app = setup();
      await send(app, payload("old"));
      app.element("question").value = "new question";
      app.element("useStreaming").checked = streaming;
      const liveTask = app.run("onSend()");
      await flush();
      const liveRequest = app.requests.at(-1);
      const oldCell = findAll(app.element("brainHistory"), "bh__cell")[0];
      assert.equal(oldCell.disabled, false);
      const historyTask = oldCell.click();
      const historyRequest = app.requests.at(-1);
      assert.match(historyRequest.url, /\/v1\/trace\/old\?/);
      if (resolveHistoryFirst) await complete(historyRequest, { ...payload("old"), found: true });
      await complete(liveRequest, payload("new", { steps: false }), streaming);
      await liveTask;

      assertHistory(app, ["old", "new"]);
      assert.equal(app.run('selectedQidBySession.get("default")'), "old");
      assert.match(app.element("metricsSubtitle").textContent, /^qid old/);
      assert.equal(app.renders.at(-1)[1], "old");
      assert.equal(app.element("chatLog").children.at(-1).querySelector(".bubble__content").textContent, "answer new");
      const update = app.messages.at(-1);
      assert.equal(update.type, "srdb.metrics.history");
      assert.deepEqual(update.payload.brain_history.map((entry) => entry.qid), ["old", "new"]);
      assert.deepEqual(update.payload.module_history.map((entry) => entry.qid), ["old", "new"]);
      assert.equal(app.requests.length, 3, "a stale live selection must not fetch or replace its trace");

      if (!resolveHistoryFirst) await complete(historyRequest, { ...payload("old"), found: true });
      await historyTask;
      assert.match(app.element("metricsSubtitle").textContent, /^qid old/);
      assertHistory(app, ["old", "new"]);

      // The newly completed trace is reachable through the refreshed history rail.
      const newCell = findAll(app.element("moduleHistory"), "mh__cell").at(-1);
      const newTask = newCell.click();
      assert.match(app.requests.at(-1).url, /\/v1\/trace\/new\?/);
      await complete(app.requests.at(-1), { ...payload("new"), found: true });
      await newTask;
      assert.match(app.element("metricsSubtitle").textContent, /^qid new/);
      assertHistory(app, ["old", "new"]);
    });
  }
}

test("trace hydration neither duplicates history nor overwrites a later selection", async () => {
  const app = setup();
  await send(app, payload("old"));
  app.element("question").value = "new question";
  const liveTask = app.run("onSend()");
  await flush();
  await complete(app.requests.at(-1), payload("new", { steps: false }));
  const hydration = app.requests.at(-1);
  assert.match(hydration.url, /\/v1\/trace\/new\?/);
  assertHistory(app, ["old", "new"]);

  const historyTask = findAll(app.element("brainHistory"), "bh__cell")[0].click();
  await complete(app.requests.at(-1), { ...payload("old"), found: true });
  await historyTask;
  await complete(hydration, { ...payload("new"), found: true });
  await liveTask;
  assert.match(app.element("metricsSubtitle").textContent, /^qid old/);
  assert.equal(app.renders.at(-1)[1], "old");
  assertHistory(app, ["old", "new"]);
});

test("current live trace hydration keeps exactly one history entry", async () => {
  const app = setup();
  app.element("question").value = "new question";
  const task = app.run("onSend()");
  await flush();
  await complete(app.requests.at(-1), payload("new", { steps: false }));
  await complete(app.requests.at(-1), { ...payload("new"), found: true });
  await task;
  assertHistory(app, ["new"]);
  assert.equal(app.renders.at(-1)[2], undefined);
  assert.equal(app.renders.at(-1)[1], "new");
});

test("popout history updates preserve the detail pane and pending trace selection", async () => {
  const app = setup("metrics.js");
  app.receive({ type: "srdb.metrics", payload: payload("old") });
  const task = findAll(app.element("brainHistory"), "bh__cell")[0].click();
  const request = app.requests.at(-1);
  const update = {
    session_id: "default",
    brain_history: [{ qid: "old" }, { qid: "new" }],
    module_history: [{ qid: "old", modules: ["LeftBrainModel"] }, { qid: "new", modules: ["LeftBrainModel"] }],
  };
  app.receive({ type: "srdb.metrics.history", payload: update });
  assertHistory(app, ["old", "new"]);
  assert.match(app.element("metricsSubtitle").textContent, /^qid old/);
  assert.equal(app.renders.length, 1, "a history refresh must not restart the trace theater");

  app.receive({ type: "srdb.metrics.history", payload: { ...update, session_id: "other", brain_history: [], module_history: [] } });
  assertHistory(app, ["old", "new"]);
  await complete(request, { ...payload("old"), found: true });
  await task;
  assert.equal(app.renders.length, 2, "a history refresh must not invalidate the pending trace ticket");
});
