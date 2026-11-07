# Consultation cleanup before fallback

A consultation timeout requests cancellation; it does not prove that the provider
operation and its asynchronous cleanup have stopped. The serial controller now
waits for the channel to become idle before leaving its consultation wrapper,
including before a direct critic/detail fallback. The callosum's existing
standalone API still reports timeouts/cancellation promptly and tracks physical
capacity until cleanup ends; this change is at the controller handoff.

`drain_consultation` shields the idle wait from later owner cancellation, drains
it, then propagates cancellation. It adds no local `wait_for` timer, injected
cancellation or new provider call. A native local timeout cannot spend the RPC
cancel-once guard. Existing model timeouts and RPC ownership remain in force;
timeout reporting can include the owned provider's cleanup time. A client that
has not acknowledged cleanup cannot authorize overlapping fallback work.

The idle wait also reaps completed operations whose done callback is still
queued. Python 3.12 can complete a gather of already-done futures synchronously;
repeatedly gathering that same set could otherwise starve the callback needed
to empty it. Exception consumption and physical-capacity accounting remain
idempotent when the callback subsequently runs.

The ordered journal is unchanged. It still requires exact request identity,
transport starts and actual response/error/cancellation terminals. It never
injects cancellation or silently reorders a historical tape.

Regression controls cover native critic timeout, asynchronous cleanup and direct
fallback through the real controller/worker/channel, JSON-round-tripped exact
scheduled replay, repeated owner cancellation during draining, later RPC
cancellation/deadline and reuse, and a subprocess-bounded done-callback case.
The original #101 failed replay remains unchanged. A separate validation applies
only this fix to its measured historical runtime: all 33 requests / 66 events
and non-timing canonical outcomes match with zero provider calls. That is a
historical-source overlay validation, not a claim that #102's intentionally
changed ordinary-generation behavior equals the old answers.

The following [fresh reasoning-effort protocol](benchmarks/reasoning-effort-protocol.md)
compares provider-default and low effort on the current completion-aware runtime,
under the same fixed ceilings. Its old-envelope validation is development
provenance; fresh answer quality requires its own observations.

The [2026-11-08 results](benchmarks/reasoning-effort-2026-11-08.md) retain all
three replay provenances and the fresh profile comparison.
