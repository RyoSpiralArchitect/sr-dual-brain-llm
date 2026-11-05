# Record replay intent — prospective protocol

2026-11-05 (Asia/Tokyo), issue #86. Freeze this document, code, both complete
new fixtures, expectations, rubric/schema wording, order, repeat-zero receipt
rule and the 96-attempt budget in a clean commit before paid calls. Preserve
#85 and all earlier outcomes, including its false literal-replay grade.

## Runtime contract

A local finite grammar recognizes a whole-record quotation after declared alias
resolution. It supports the existing literal forms and bounded Japanese/English
deictic followups. The reference must resolve to one scope with exactly one live
primary source **before** candidate limits/ranking; duplicate identities, pending
primary writes, unknown/multiple scopes, inactive/derived/incomplete sources,
extra clauses and nonliteral transformations do not receive this intent.

The immutable intent binds exact normalized query, complete source contents,
source ID and scope. Answerability cache identity includes the intent and replay
policy. The checker receives a typed `recorded_answer_replay` operation: the
requested object is historical answer text, rather than an unnamed factual
attribute or independently verified truth. A positive response must quote the
entire answer exactly. A denial, malformed/partial quote or changed source never
opens delivery. Ordinary fact queries retain the original policy and cache
contract; quotation and factual approvals cannot be reused interchangeably.

Approved quotations use an explicit JSON context with source question and complete
`recorded_answer`, asking generation to preserve the text without labels or added
claims. The existing generation/completion path remains active; this is not a
deterministic rendering bypass. Existing selective-consultation AUTO still needs
its risk/confidence/mode/lead/context guards and exact approved source equality;
the replay path can handle qualitative text without numeric interpretation.
System2 ON, primary-write completion, precision/Director/ACC authority and source
lifecycle stay mandatory. No module is added and no AUTO default is promoted.
The selective policy provenance becomes `literal_replay_v2`.

## Independent answer-grader calibration

`answer_assessment_v2` is an explicit API for future answer studies. Historical
graders/callers/scores stay unchanged; the private-advice evaluator is separate.
The new contract distinguishes requested historical quotation, factual assertion,
actual exposed evidence, a missing current source and an ambiguous reference.
Faithful requested quotation can be grounded even when its record is unverified.
Extra framing may fail exact task while remaining grounded; invented quoted
content fails both. A comparison reference does not grant access to withheld text.
Clarification depends on the supplied scope/evidence, not invented requirements.

Four fresh cases, three authored variants each, two reversed label presentations
and two rubric arms yield **16 judge packets**, 24 output presentations and 72
criterion checks per rubric, but only **12 distinct variants in four cases**.
The arms receive the same packet, explicit requirements, actual available context
and labels. Expected values/variant names are never sent. Label seed is fixed
by case ID; rubric order alternates with case index plus presentation order.

Report each criterion's fixed-expectation agreement, verdict counts, paired
corrections/regressions, order disagreements, failed/unattempted packets and
cost. Focus: exact unverified/current-corrected quotation task/grounding passes,
extra framing task failure, changed/stale quotation grounding failures, correctly
targeted missing/ambiguous-source clarification. Direct equality remains an
independent diagnostic and never overwrites a model grade. Valid JSON and model
agreement are not truth or evidence of general judge reliability.

## Support as a separate repeated factor

Six fresh scenarios: explicit literal quote, actual declared correction plus
pronoun followup, another project amid distractor records, expired retention
with a current unrelated owner record, an extra verification clause, and a source
above the complete-answer limit. Three fresh checker/gate instances per arm and
scenario yield **36 planned preparations**. Ineligible/empty preparations make
zero provider calls, retained as such. Positive expected source IDs are fixed;
source absence, denial, invalid response and transport failure stay distinct.

`generic_contract` removes typed intent/context from the current runtime;
`replay_contract` retains it. This is a feature ablation, not an exact historical
binary comparison. Both share the proposed stricter AUTO boundary; the ablation
cannot establish whole-version cost savings versus old releases. The ordinary
fact policy stays unchanged. Cases with no replay intent use the same checker
payload across arms; their repeated occurrences are not different policies.
Alternate arm order by scenario plus repeat index. Keep every occurrence;
no best-of-three selection, majority vote, retry or post-result adjudication.
Repeated outputs in one authored scenario are not independent problems.

## Bound receipt reuse in production sessions

The first four scenarios each run a fresh real EngineSession per arm. Setup uses
authored records and normal SharedMemory correction/focus operations, not LLM
primary writes or PostgreSQL persistence. Existing real-PostgreSQL tests cover
their separate transaction/restart contracts. This pilot is not a retained
six-turn trajectory, a source-generation test or a durable-memory benchmark.

For each scenario/arm, runtime **only reuses support repeat zero**, including its
false or absent approval. Import keys only when the actual normalized query,
complete source, intent, policy and backend identity match the measured key.
Missing receipt withholds evidence and makes no new support request. Never copy
another arm/repeat's better decision. All support-factor cost remains in the full
ledger. Production approval/lifecycle/completion checks still run; no expected
answer is installed into an approval cache.

Both arms generate natural initial answers independently; they do not share a
draft. This measures the combined intent/support/context handoff, not a pure
routing intervention. Use fixed logical wall time and matched local seeds.
Capture actual exposure, imported receipt hashes, route, completion availability,
usage and handler duration. Exact approved-answer equality is reported separately
from source acceptance and model grades. Each available runtime answer is judged
once with its **own** actual context (at most eight calls), never another arm's
record approval. No paid rescue or repeated judging.

Acceptance is bounded: demonstrate the requested quotation/correction operation
through production preparation, context and routing with evidence guards intact;
report every remaining support/quotation/judge failure rather than claiming
general reliability or speed. A stronger future claim needs independent cases
and replication. A failed focus test or missing response remains a failed result.

## Budget, replay and artifacts

The hard **96 HTTP-attempt cap** covers all 16 calibration calls, at most 36 support
calls, production drafting/critique/integration/fallback and up to eight runtime
judges. Use `gpt-6-luna` production/support and `gpt-6-sol` judges, with output caps
2048 for generation/judges and the existing 4096 for support. Provider timeouts
are 60 seconds for generation/judging and the existing bounded support timeout.
No continuation, warmup, budget reset, resampling or added model. A transport,
auth or quota error blocks subsequent paid transport through the monotonic latch.
Unknown usage is not zero. Returned invalid completions remain counted.

Freeze fixture/source/config/rubric hashes in the journal; live CLI requires a
clean committed tree. Exact replay has `transport=None`, consumes every recorded
occurrence and verifies no missing/unused request. Calibration scores, support
results/receipt bindings, natural answers/status, exposure/routes, final grades
and usage must match; real elapsed durations are excluded. Replay is not an
independent model trial, latency validation or regrading. Keep raw text/quotes
private under ignored `samples/module-gating-record-replay-2026-11-05-*`.
Publish safe aggregates, errors and artifact hashes.

```bash
python sr-dual-brain-llm/scripts/benchmark_record_replay.py \
  --model gpt-6-luna --judge-model gpt-6-sol --max-calls 96 \
  --output-dir sr-dual-brain-llm/samples/module-gating-record-replay-2026-11-05-live
python sr-dual-brain-llm/scripts/benchmark_record_replay.py \
  --journal sr-dual-brain-llm/samples/module-gating-record-replay-2026-11-05-live/journal.json \
  --judge-model gpt-6-sol --max-calls 96 \
  --output-dir sr-dual-brain-llm/samples/module-gating-record-replay-2026-11-05-replay
```
