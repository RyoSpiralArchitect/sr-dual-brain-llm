# Prospective live integration confirmation of PR #93

The earlier `fd69cf8` prototype consumed 32 real calls before the pre-existing
PR #93 was discovered. Preserve that prototype's frozen code, inputs, receipts,
metrics and wording limitation. Its observations do not establish PR #93's
broader generation-boundary guarantees. PR #93's reviewed core snapshot is
`b5d90f052f5963f37f87236f5156146d3ab3180a` and predates those model outputs.

For integration, use PR #93's core files unchanged. Freeze this separate protocol,
the new `record_selection_integration.json` fixture and compatible harness before
paid calls. There is no post-output tuning of PR #93 or regrading of earlier
observations. Record the prototype and integration as distinct studies.

## Allocation and fixed order

Integration allocation: **32 real HTTP attempts maximum**, including setup
support, generation, review, detail, integration, retries and repair. Together
with the earlier 32-call prototype, the work remains within **64 real attempts**.
Do not use unused allocation for warm-up, resampling, paid rescue or a judge.
Retain every failure and unknown receipt; a transport/auth/quota failure closes
paid admission through the existing sticky raw journal.

Use `openai/gpt-6-luna`, production nonstream `EngineSession`, existing all-AUTO
module settings with Director/executive advice and metrics observation OFF,
2,048 output tokens and unchanged timeouts/retries. Fresh in-memory fixtures do
not constitute a new PostgreSQL write benchmark. Run these **five fixed cases
once**, with new names and values:

1. Two current records, English whole quotation, explicit System2 ON.
2. Two current records, Japanese whole quotation, AUTO.
3. Two current records, caller-declared pronoun request, explicit System2 OFF.
4. One current quote plus an expired second record, AUTO.
5. Ordinary factual lookup with two current same-project records, AUTO.

Before each ambiguous case, submit exactly the two listed source identities once
to the real ordinary support backend with the exact runtime retrieval query.
Keep all true/false/unknown decisions. No response is forced positive. These
receipts cannot supply a unique binding or bypass PR #93's selection gate. Unique
quotation must still obtain its own typed full-source approval.

## Observations and limits

Freeze expected actions, bindings, source counts, delivery expectations and
manual criteria in the fixture. Report the actual `replay_resolution` status,
ordinary support receipts, source delivery, current missing condition, final
answer, controller requests and HTTP usage independently.

For the three ambiguous cases expect `ambiguous_record`,
`current_unselected / unknown / clarify_record`, no selected binding, no exposed
QAs and no stored-value canaries in final text or generation/review/integration
packets. The support model is allowed to see source values. Preserve any optional
request to paste text as a wording limitation; structural success is not a
semantic grade. Read-only caller identity does not grant factual approval.

For the current/expired control expect exact UTF-8 equality with the named actual
current source and observed V3 source-validity checks. The expired record must
not create ambiguity or leak its canary. The factual control must retain the
ordinary source path and report the recorded value with attribution. Stored
claims remain unverified. Explicit ON/OFF and final accounting must remain
consistent. All V3 semantic grades stay unknown; inspection is unblinded by the
implementing assistant. No paired performance or general quality claim is made.

Keep raw journals/private prompts ignored, publish safe metrics and fictional
excerpts. Replay all exact ordered requests with no transport object under the
offline HTTP guard. Compare canonical non-timing outcomes, including any failure
and wording limitation; do not edit the measured code or criteria to repair a
replay divergence. Queued/lifecycle changes are covered by PR #93's offline
regressions, not claimed as naturally sampled live events in these five cases.

```bash
python sr-dual-brain-llm/scripts/benchmark_record_selection.py \
  --model gpt-6-luna --max-calls 32 \
  --output-dir sr-dual-brain-llm/samples/module-gating-record-selection-integration-2026-11-05-live
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_record_selection.py \
  --journal sr-dual-brain-llm/samples/module-gating-record-selection-integration-2026-11-05-live/journal.json \
  --max-calls 32 \
  --output-dir sr-dual-brain-llm/samples/module-gating-record-selection-integration-2026-11-05-replay
```
