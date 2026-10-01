import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

const source = readFileSync(new URL("../csharp/SrDualBrain.Gateway/wwwroot/trace-theater.js", import.meta.url), "utf8");

class FakeElement {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attributes = new Map();
    this.listeners = new Map();
    this.style = {};
    this.className = "";
    this.hidden = false;
    this.disabled = false;
    this.open = false;
    this._text = "";
  }

  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }

  get textContent() {
    return this._text + this.children.map((child) => child.textContent).join("");
  }

  set innerHTML(_value) { throw new Error("Untrusted payload must not use innerHTML"); }

  get firstChild() { return this.children[0] || null; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this._text = ""; this.children = children; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  removeAttribute(name) { this.attributes.delete(name); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  addEventListener(name, listener) { this.listeners.set(name, listener); }
  click() { if (!this.disabled) this.listeners.get("click")?.(); }
}

function find(root, className) {
  if (root.className.split(" ").includes(className)) return root;
  for (const child of root.children) {
    const match = find(child, className);
    if (match) return match;
  }
  return null;
}

function setup() {
  const root = new FakeElement("div");
  const timers = new Map();
  let nextTimer = 1;
  let motionListener = null;
  const motion = {
    matches: false,
    addEventListener(name, listener) { motionListener = listener; },
    removeEventListener() { motionListener = null; },
    change(matches) { this.matches = matches; motionListener?.({ matches }); },
  };
  const window = { matchMedia: () => motion };
  runInNewContext(source, {
    document: { createElement: (tag) => new FakeElement(tag) },
    window,
    setInterval: (fn) => { const id = nextTimer++; timers.set(id, fn); return id; },
    clearInterval: (id) => timers.delete(id),
  });
  return { root, theater: window.TraceTheater.create(root), timers, motion };
}

test("empty and malformed flows remain usable", () => {
  const { root, theater } = setup();
  theater.render(null, "one");
  assert.equal(find(root, "tt__count").textContent, "No steps");
  assert.equal(find(root, "tt__control--play").disabled, true);
  theater.render(null, "one", "loading");
  assert.match(find(root, "tt__empty").textContent, /Loading recorded steps/);
  theater.render(null, "one", "unavailable");
  assert.match(find(root, "tt__empty").textContent, /Trace unavailable/);
  theater.render({ steps: [null, "bad"], architecture: null }, "two");
  assert.equal(find(root, "tt__count").textContent, "No steps");
  assert.match(find(root, "tt__empty").textContent, /No dialogue steps were recorded/);
  theater.destroy();
  assert.equal(root.children.length, 0);
});

test("steps navigate, stay collapsed, and cap payload text", () => {
  const { root, theater } = setup();
  theater.render({
    steps: [
      { role: "left", phase: "left_draft", content: "<img src=x onerror=alert(1)>" },
      { role: "right", phase: "callosum_response", content: "x".repeat(10000), meta: { note: "y".repeat(1000) } },
    ],
    architecture: [{ stage: "inner_dialogue", modules: ["LeftBrainModel", "RightBrainModel"] }],
  }, "turn-a");

  assert.equal(find(root, "tt__count").textContent, "Step 1 of 2");
  assert.equal(find(root, "tt__sr-only").textContent, "Step 1 of 2: left, left draft");
  assert.equal(find(root, "tt__details").open, false);
  assert.equal(find(root, "tt__content").textContent, "<img src=x onerror=alert(1)>");
  assert.match(find(root, "tt__stages").textContent, /inner dialogue2/);
  find(root, "tt__controls").children[2].click();
  assert.equal(find(root, "tt__count").textContent, "Step 2 of 2");
  assert.equal(find(root, "tt__sr-only").textContent, "Step 2 of 2: right, callosum response");
  assert.equal(find(root, "tt__details").open, false);
  assert.ok(find(root, "tt__content").textContent.length < 2500);
  assert.ok(find(root, "tt__meta").textContent.length < 300);
  theater.destroy();
});

test("new qid stops playback and reduced motion disables it", () => {
  const { root, theater, timers, motion } = setup();
  const flow = { steps: [{ role: "left", phase: "draft" }, { role: "right", phase: "consult" }] };
  theater.render(flow, "turn-a");
  find(root, "tt__control--play").click();
  assert.equal(timers.size, 1);
  theater.render(flow, "turn-b");
  assert.equal(timers.size, 0);
  assert.equal(find(root, "tt__count").textContent, "Step 1 of 2");
  find(root, "tt__control--play").click();
  assert.equal(timers.size, 1);
  motion.change(true);
  assert.equal(timers.size, 0);
  assert.equal(find(root, "tt__control--play").disabled, true);
  theater.destroy();
});

test("long flows disclose the gap and keep the ending reachable", () => {
  const { root, theater } = setup();
  const steps = Array.from({ length: 120 }, (_, index) => ({
    role: index === 119 ? "integrator" : "left",
    phase: `phase_${index + 1}`,
  }));
  theater.render({ steps }, "long-turn");
  const rail = find(root, "tt__rail");
  assert.equal(rail.children.length, 81);
  assert.match(rail.children[40].textContent, /40 middle steps omitted/);
  assert.equal(find(root, "tt__count").textContent, "Step 1 of 120 · 40 omitted");

  rail.children[39].firstChild.click();
  assert.equal(find(root, "tt__count").textContent, "Step 40 of 120 · 40 omitted");
  find(root, "tt__controls").children[2].click();
  assert.equal(find(root, "tt__count").textContent, "Step 81 of 120 · 40 omitted");
  rail.children[80].firstChild.click();
  assert.equal(find(root, "tt__count").textContent, "Step 120 of 120 · 40 omitted");
  assert.equal(find(root, "tt__phase").textContent, "phase 120");
  theater.destroy();
});
