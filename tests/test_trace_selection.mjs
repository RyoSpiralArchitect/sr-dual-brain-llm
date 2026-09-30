import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

const source = readFileSync(new URL("../csharp/SrDualBrain.Gateway/wwwroot/trace-selection.js", import.meta.url), "utf8");

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function setup() {
  const window = {};
  runInNewContext(source, { window });
  return window.TraceSelection.create();
}

test("a late history response cannot replace a newer selection", async () => {
  const selection = setup();
  const older = deferred();
  const newer = deferred();
  const shown = [];

  const oldTicket = selection.begin();
  const oldTask = selection.apply(oldTicket, () => older.promise, (value) => shown.push(value));
  const newTicket = selection.begin();
  const newTask = selection.apply(newTicket, () => newer.promise, (value) => shown.push(value));

  newer.resolve("newer trace");
  await newTask;
  older.resolve("older trace");
  await oldTask;
  assert.deepEqual(shown, ["newer trace"]);
});

test("a stale failure is silent after a new selection or clear", async () => {
  const selection = setup();
  const pending = deferred();
  const errors = [];
  const ticket = selection.begin();
  const task = selection.apply(ticket, () => pending.promise, () => {}, (error) => errors.push(error.message));
  selection.begin();
  pending.reject(new Error("expired trace"));
  await task;
  assert.deepEqual(errors, []);
});
