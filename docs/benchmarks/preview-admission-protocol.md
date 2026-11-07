# Speculative preview admission and scheduled replay — 2026-11-07

Freeze code, fresh fixture, expectations, ordering, budget and replay contract
before provider outputs. This is the bounded operational followup to #98. The
#97 trace and its original replay failure remain immutable diagnostic evidence,
not an independent holdout or a score to repair. This study freshly instantiates
the same twelve-step acceptance structure with new identities, values and
wordings; it is not a new broad capability benchmark.

## Runtime change

The existing TRN registry gains `speculative_preview_suppression`, default OFF.
OFF preserves preview admission. ON suppresses eligible optional previews. AUTO
suppresses only a locally typed pending record choice/reselection recognized by
`record_choice_clarifications`, before generation. It uses no draft, confidence,
eventual policy action, model classification or new evidence check. Other missing
evidence, unknown support and ordinary questions retain preview admission.

Source writes, explicitly requested right lead, explicit System2 ON/OFF,
precision priority and Director consultation overrides veto suppression, as do
missing conversation/evidence machinery and stages without a requested preview.
Automatically rotated right lead may omit its speculative preview; the later
ordinary routing, consultation and integration remain unchanged. Suppression
does not certify that consultation will be unnecessary or supply an answer.

The preview stage owns and drains its child before returning, even outside an
RPC dispatcher and during abandonment. `preview_execution` / `metrics.preview`
distinguish not requested, suppressed, cancelled before coroutine start,
cancelled after start, completed, empty and failed work. Coroutine start is not
an HTTP receipt. Only the provider journal counts actual attempts and cost.

## Single natural persistent trajectory

Execute `preview_admission_trajectory.json` once in its fixed order. Use normal
RightBrainPolicy, no forced action or substituted draft, actual `gpt-6-luna`
generation, the existing opt-in all-AUTO balanced profile and Director/executive
OFF. Keep the production nonstream JSON path, 2,048-token generation ceiling,
4,096-token support budget, ordinary-detail stage budgets, strict completion,
repair/fallback authority and 60-second transport timeout. Keep the prospectively
fixed logical clock and existing session RNG seeds. These settings do not
promote a new product default.

The twelve turns comprise two model-generated primary writes; ambiguous replay;
listed source selection/quotation; actual model-generated correction; stale
reference reuse; current reference reuse after restart; relisted corrected
quotation; current-period lookup; bare parameter; explicit Japanese missing
rate; explicit missing period in another scope. Only actual successful writes
can satisfy dependencies. Preserve failed writes, missing choices and skipped
dependent turns; do not seed substitute sources, choose another record, repeat
an answer or repair a favorable result.

Use real isolated PostgreSQL with a fresh owned namespace and ignored private
outbox. Preserve commit, source identity, explicit correction, restart, scope,
source-approval and reference privacy contracts. Listing makes zero model calls.
Both quotation checks compare the actual current source bytes. Source-free
answers retain their separate V3 source-validity failures; all semantic grades
are explicitly **not assessed**. No judge, new calibration, reused historical
score or independent human quality claim is part of this operational study.

Report pre-draft gate decisions, preview lifecycle, actual HTTP allocation,
completion and failed/unknown receipts, persistence and exact-source contracts,
per-handler/listing/restart times and total episode time. Episode excludes user
think time and teardown. Report whether suppression was actually exercised,
including zero exposure if it was not. There is **no paired live comparison**:
do not estimate quality, latency or request savings from this one trace or
compare totals with #97. The paired synthetic-delay Fake control establishes
admission mechanics only; its one-versus-zero attempts are not live savings.

## Cost and replay

Hard cap **48 actual provider HTTP attempts total**, counting all support,
generation, preview, detail, critic, repair, integration, retry, fallback,
cancellation and failures. One fixed order, no resampling, continuation, paid
rescue or extra paid diagnostics. A real transport/auth/quota failure keeps the
existing sticky paid-admission latch closed. Cancellation with no provider
receipt retains unknown usage and known subtotals, never free cost.

An opt-in `ScheduledJournal` observes each HTTP admission and its response,
error or actual cancellation in one ordered event tape, separately hashed.
Validate all event shapes, identities, admission order, terminal outcomes and
complete coverage before replay. Replay has no provider transport object and
runs under the offline HTTP guard. It waits for recorded transport order, not
elapsed wall time. A recorded cancellation must arrive from the actual owning
task; never inject it to manufacture the desired controller branch. Missing or
different order, unused events and a bounded replay wait failure remain explicit
failures, even if a production guard catches an exception.

Latency-only feedback replay restores measured Hypothalamus/BasalGanglia inputs
while verifying every non-timing input and ordered coverage. Never replay an
action, reward or success as an oracle. Require every exact ordered request and
equal canonical non-timing rows, failed/unknown receipts, feedback digest and
transport-event digest. This is not a wall-time replay or a claim that all
possible asyncio scheduling is reproducible. Preserve any real divergence
without post-output tuning or a paid repeat.

Raw prompts, responses, references, judge-free private context and outbox stay
under ignored samples; publish filtered observations and artifact hashes. Local
delayed-provider controls cover pre-admission cancellation (zero attempts),
in-flight cancellation (one unknown-cost attempt), completed preview, transport
failure and owning-turn cancel/deadline with no surviving task. Python 3.10 CI
executes the lightweight ownership controls; real PostgreSQL executes the full
delayed trajectory and JSON-roundtrip scheduled replay.

```bash
python sr-dual-brain-llm/scripts/benchmark_preview_admission.py --model gpt-6-luna \
  --max-calls 48 \
  --output-dir sr-dual-brain-llm/samples/module-gating-preview-admission-2026-11-07-live
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_preview_admission.py \
  --journal sr-dual-brain-llm/samples/module-gating-preview-admission-2026-11-07-live/journal.json \
  --max-calls 48 \
  --output-dir sr-dual-brain-llm/samples/module-gating-preview-admission-2026-11-07-replay
```
