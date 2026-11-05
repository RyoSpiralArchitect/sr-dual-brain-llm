# Record replay: reference and assessment followup

2026-11-05 (Asia/Tokyo). Issue #86 was closed by merged PR #88 before this
followup began. Its production quotation path and prospective measurements are
the baseline. This work addresses two reproduced reference failures and makes
the requested final-answer dimensions independently inspectable. It does not
re-run or replace #85/#88 model judgments.

## Bounded production changes

The existing finite literal grammar remains the authority for whole-record
quotation. A declared multiword alias must survive the separate unknown-project
check. A deictic request accepted by that grammar must use the same Unicode and
whitespace normalization when carrying the preceding, unambiguous focus.

Neither change selects a source from ambiguous identity, invents a new alias,
ignores an explicitly unknown project, or classifies a semantic request as a
quotation. An interrupted focus, multiple records, extra instructions,
transformations and incomplete or inactive sources remain negative controls.
Current/complete/scope checks, exact query/source/policy approval, source
completion and manual System2 modes retain their existing authority.

## Four independent assessment dimensions

Use a new explicitly versioned contract. Existing answer rubrics and their
callers, fixtures and saved scores remain unchanged.

| Dimension | Evidence and interpretation |
| --- | --- |
| Exact match | Compare UTF-8 bytes to an explicitly declared target without stripping, normalization, paraphrase or model adjudication. The target is diagnostic data; it does not become answering evidence. |
| Source validity | Inspect actual exposure, target identity and explicit current, complete, scope and support-approval checks. Missing or malformed facts remain unknown. A known absent, denied or invalid source is reported as such, even if the output happens to match its stored bytes. |
| Attribution | Separately assess whether the answer represents the source and its epistemic status faithfully. Requested quotation of an unverified record does not by itself claim factual verification. Extra attribution can be faithful while violating byte equality. |
| Information sufficiency | Separately assess how the answer handles evidence availability or ambiguity under the actual supplied context. Neither equality nor the comparison reference grants missing evidence. |

The two semantic dimensions have their own strict response fields and literal
evidence validation. A provider error, malformed judgment or uncertain verdict
must remain distinguishable from failure. A deterministic equality result cannot
fill either semantic field. No generic semantic classifier is added to memory
retrieval or intent resolution.

Do not combine these dimensions into one success score: an appropriate request
for missing evidence may coexist with an unavailable source and no exact quote.
Keep the original model grade beside the independent diagnostic and explicitly
report disagreement with its `task` field. Unknown or absent legacy grades are
not silently counted as disagreement or agreement. A new diagnostic does not
edit a historical model grade.

The source snapshot supplies `evaluation_time` for expiry comparisons; the
assessor never substitutes its own wall clock. Explicit contradictory metadata
(for example, `superseded` alongside a passing current-source flag) cannot pass.
Multiple records are allowed for ordinary fact questions when each has explicit
`source_checks_by_qid`; the single-record constraint belongs to whole-record
replay. Semantic packets contain only allowlisted actual context, excluding the
equality target, comparison reference, expected scores and legacy judgments.

## Fixed-condition development comparison

The new runner uses an explicitly scripted Fake Provider and the actual
EngineSession preparation, generation, completion and routing path. It offers
no live-provider mode. Its scenarios and expected source sets are authored
before the comparison and included in the source tree. Record hashes of the
fixture, code, assessment contract and runtime configuration with the results.

Compare the existing generic-contract ablation with typed replay using the
same declared records, correction operations, logical clock and local seeds.
Measure support separately with three fresh preparations per arm. Alternate
arm order deterministically. Reuse only preparation zero's exact bound receipt
for that scenario and arm, including denial or failure. Never substitute a
later approval, expected source set or another arm's receipt.

Both arms generate their own answers, so the comparison covers the
intent/support/context path. It cannot isolate a consultation effect. Keep
source identities and conditions visible at each boundary. Include multiword
identity, corrected deictic continuation, ordinary semantic and extra-clause
requests, cross-scope records, and inactive/incomplete or denied sources.

The checked-in development fixture declares eleven cases: `multiword_quote`,
`corrected_japanese`, `corrected_normalized_pronoun`, `cross_project`,
`expired_source`, `superseded_source`, `incomplete_source`, `repeat_zero_denial`,
`partial_approval`, `extra_clause` and `semantic_question`. Two arms and three
support preparations give 66 planned preparations, followed by 22 runtime
sessions and up to 22 semantic-assessment packets. Empty/ineligible preparations
make no provider-shaped call. The shared attempt cap is **192**.

Fake responses exercise the response contract; their expected semantic scores
are authored fixtures, not evidence of model-judge accuracy. Likewise, scripted
support differences demonstrate wiring and guard behavior, not the probability
of an actual provider approving a query. The older measured comparisons retain
their own scope and denominators.

## Receipts, replay and reporting

Use the fixed all-attempt cap and the existing occurrence-based journal. Retain
all attempted calls, returned invalid responses, unknown usage and unavailable
cells. A transport failure blocks later attempts. Do not retry or replace a
failed answer or judgment. Tests separately exercise failure and denial paths.

An exact second execution consumes the recorded journal with transport removed.
Every recorded occurrence must be consumed; missing and extra requests fail
replay. Compare support decisions and bindings, answer bytes, exposure, routing,
the four dimensions, legacy judgments and usage. Exclude measured elapsed time
from exact replay equality. Record fake transport calls separately from real
provider calls, which must remain zero in both executions.

For a failed transport, retain its original error category from the immutable
provider receipt. The inherited replay layer raises a local wrapper exception;
record that wrapper separately and exclude it, like elapsed time, from canonical
comparison. A changed wrapper must not replace the recorded failure or turn
unknown usage into zero. Returned malformed support decisions still preserve
their own validation failure.

Elapsed times describe this local scripted execution and its receipt-reuse
boundaries. They are not API latency estimates or a production speedup. Public
summaries retain outcomes, disagreement indicators and receipt hashes while
omitting answer text, source text, model evidence quotes and receipt cache
contents. Preserve private detailed reports for exact debugging.

## Acceptance and remaining limits

Acceptance requires deterministic reproductions for both reference bugs;
positive replay and ordinary-query/unknown-identity controls; independent-axis
examples including exact text with an invalid source and faithful framing with
unequal bytes; strict malformed/unknown judgment behavior; immutable legacy
grades; and a complete zero-transport journal replay.

This followup does not implement the separate missing-parameter clarification
prototype in #89 or the empty-critic reliability work in #87. It makes no claim
about real-provider stability, durable PostgreSQL writes or general reliability
from its authored development cases.

## Run locally

Both commands use the same checked-out code and fixture. Output directories must
be new. The first command uses the scripted provider; the second has no transport.

```bash
python sr-dual-brain-llm/scripts/benchmark_record_replay_development.py \
  --output-dir sr-dual-brain-llm/samples/module-gating-record-replay-development-fake
python sr-dual-brain-llm/scripts/benchmark_record_replay_development.py \
  --journal sr-dual-brain-llm/samples/module-gating-record-replay-development-fake/journal.json \
  --output-dir sr-dual-brain-llm/samples/module-gating-record-replay-development-replay
```
