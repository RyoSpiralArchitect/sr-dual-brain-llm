# Prospective operational verification after PR #91

This is a single current-profile acceptance pilot. Freeze this protocol, the
fresh `contract_live_verification.json` fixture, runner and transport tests in a
clean commit **before** paid HTTP. No historical benchmark, fixture, rubric,
availability rule or grade is changed. It does not close the natural-answer and
judge-calibration work in #89 or estimate a before/after improvement.

## Fixed order and evidence

Use the production nonstream JSON client with `openai/gpt-6-luna`, 2,048 output
tokens, no automatic continuation, and its existing timeouts/retry/repair rules.
Use the existing all-AUTO module profile with Director, executive advice and
executive metrics observation disabled. Round-target minimum remains zero.
Pass explicit System2 ON and OFF only for the two declared critic controls;
ordinary memory probes use AUTO. Source writes use the existing left-only seed
profile with OFF, and the next read opens a fresh normal session. This controlled
setup is not a claim about every production environment or streaming behavior.

Run all **15** fixed turns once, in fixture order:

1. Five-turn real PostgreSQL trajectory: write a fictional multiword project
   setting (11 days), reopen and quote it, reopen and explicitly supersede it
   with 7 days, reopen and quote the corrected source in Japanese, then request
   a pronoun-only quote in the same session. The equality oracle is the actual
   complete primary record committed before that probe, never an authored answer
   selected after seeing output. Record matching durable operation receipts,
   active primary IDs and source fingerprints. No replacement generation is
   allowed after a failed write; dependent probes keep their failures.
2. Ten isolated current-memory cases: caller-declared scope with global and
   other-project distractors; explicit period missing; explicit rate missing;
   bare retention; ambiguous reference; ambiguous whole record; missing whole
   record; current oversized record whose support remains unknown; natural
   critic ON; the same question with critic OFF. Oversized source eligibility
   is a declared controlled negative, not a simulated provider outage. Other
   support decisions and all generated text come from the actual model.

For each case retain the final answer, provider/client failures separately from
answer availability, actual exposed/approved sources, reference/intent/missing
condition, routing and actual critic controller attempts. A controller attempt
is not an HTTP request: retries, repair, support and ordinary detail calls are
counted separately in the HTTP ledger. OFF forbids critic dispatch but can still
allow ordinary detail consultation.

Record deterministic UTF-8 equality and V3 source-validity diagnostics using
observed current/complete/scope/approval checks. The V3 semantic criteria remain
**unknown / not assessed**; no model judge is called. An empty exposed-source set
is a source-availability failure, not evidence that a correct clarification is
bad. Do not combine all axes into an aggregate answer score. For ambiguous or
missing records, report exact equality as unavailable rather than treating a
clarification as a failed quote. Public contract checks of literal canary values
are limited string checks, not semantic grounding certification.

Natural responses are inspected against each fixture's prospective
`manual_criterion`, retaining literal answer excerpts and every observed gap.
This is an unblinded assistant inspection, not independent adjudication or a
calibrated semantic grade. No rubric adjustment, selective resampling, warm-up,
paid rescue, best-of-N selection or retrospective improvement claim is allowed.
If a missing-condition or source-selection expectation fails, preserve it in
the result and describe the next investigation; do not silently fix and rescore.

## Costs, stopping and replay

Hard cap: **128 actual provider HTTP attempts for the entire pilot**, including
both writes, all support, generation, detail, critic, retry, repair and fallback
calls. No judge allocation. Missing usage stays unknown with known subtotals;
cancelled/invalid responses are not free successes. A real transport/auth/quota
failure closes paid admission for the remaining run. Production retries may
still be attempted locally and must be recorded as denied admissions without
additional paid HTTP. Every planned turn remains represented if the model stage
returns an error or fallback. Harness/infrastructure failures leave incomplete
checkpoints and cannot be represented as a complete pilot.

The fresh raw journal retains **every** HTTP occurrence, including duplicate seed
requests, in admission order. Headers and credentials are never serialized;
only the endpoint hash, full request payload, raw returned JSON envelope, safe
error category and usage are retained privately. Crucially the journal does not
prevalidate or normalize empty/length/refusal envelopes: production completion
and critic validators see the actual response. Atomic private checkpoints and
per-occurrence digests preserve failures if execution stops.

Replay consumes exact ordered requests and raw responses with no transport
object, under the offline HTTP guard. It must consume every occurrence and match
the canonical non-timing outcomes. A mismatch remains a replay failure; it does
not authorize new paid calls or timing changes. Source-code, fixture, runtime and
model configuration hashes must match the frozen run. Original failure receipts
cannot be removed or edited. Unsupported transport exception types make exact
failure replay unavailable rather than being normalized into successful output.

Both modes require the isolated **DUALBRAIN_TEST_PG_DSN**, allocate a fresh owned
table prefix, and use a private outbox inside their ignored output directory.
Cleanup drops only those owned tables. Application DSNs/outboxes are not used.
Raw journals, prompts and full traces remain ignored/private; publish safe
metrics, hashes, selected fictional answer excerpts and this protocol.

## Commands

From the repository root, with the isolated PostgreSQL environment sourced:

```bash
python sr-dual-brain-llm/scripts/benchmark_contract_live.py \
  --model gpt-6-luna --max-calls 128 \
  --output-dir sr-dual-brain-llm/samples/module-gating-contract-live-2026-11-05-live
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_contract_live.py \
  --journal sr-dual-brain-llm/samples/module-gating-contract-live-2026-11-05-live/journal.json \
  --max-calls 128 \
  --output-dir sr-dual-brain-llm/samples/module-gating-contract-live-2026-11-05-replay
```
