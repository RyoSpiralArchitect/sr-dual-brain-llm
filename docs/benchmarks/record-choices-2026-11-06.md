# Read-only memory record choices — 2026-11-06

**Four selected quotations match their actual source exactly.** A changed source
and an old reference after restart both request a new selection without exposing
record bodies. Relisting after restart completes the quotation. Candidate lists
make no model calls. These are six fixed acceptance cases, not a general quality
or latency comparison.

Implementation, inputs, API contract, protocol and runner were frozen before paid
calls at `47588c2f5318023a9e2253296a3730fd555d40b5`, on merged PR #94
(`271c21c63492f58caed1a63ae0ca5212e9b66e73`). The
[prospective protocol](record-choices-protocol.md),
[fixture](../../sr-dual-brain-llm/examples/record_choices_confirmation.json),
[safe metrics](record-choices-2026-11-06.metrics.json) and
[API contract](../conversation-memory.md#read-only-record-choices) are available
for inspection. Earlier #92–#94 studies and scores are unchanged.

## Selection round trip

The production `record_choices` / `process` dispatch entrypoints list short
question labels, dates and opaque references, then receive the chosen reference
in read-only caller metadata. A reference binds this memory incarnation and the
complete source fingerprint; it supplies identity, never factual approval. Scope,
lifecycle, uniqueness, completeness and the typed full-source support decision
are checked again. No different or newer record substitutes for a failed choice.

The actual `openai/gpt-6-luna` outputs were:

| Fixed case | Selection outcome | Final answer | Calls |
| --- | --- | --- | ---: |
| English, choose backup, System2 ON | `bound`, backup only | `Recorded backup: Yokohama.` | 3 |
| Japanese, choose retention, OFF | `bound`, period only | `Recorded retention: 19 days.` | 3 |
| Change chosen source after positive approval | `stale_selection`, no source body | `Please choose the record again from the current record choices.` | 4 |
| PostgreSQL, choose backup | `bound`, backup only | `Recorded backup: Wakayama.` | 2 |
| Restart, submit old reference as a fresh operation | `invalid_selection`, no source body | `Please select the recorded answer again from the current record choices.` | 2 |
| Relist in restarted session, choose backup | `bound`, backup only | `Recorded backup: Wakayama.` | 2 |

Each quotation passes exact UTF-8 equality against the prospectively named actual
current source, and the observed V3 source-validity checks pass independently.
The four approvals are distinct exact typed replay-policy decisions. The fifth
support call, before the content change, also returns true; that prior approval
cannot preserve a changed selection. All receipts are retained, with no favorable
resampling.

Both invalidation cases expose no QA and retain
`selection_unavailable / unknown / reselect_record`. Their prospectively listed
old/changed/unselected value canaries are absent from the final response and all
runtime provider packets. The before-change support packet deliberately sees
the source. These are literal canary observations, not a general semantic leak
certificate. Signed references are absent from provider payloads in every case.

V3 source validity remains **fail** for the two intentionally source-free
clarifications (`no_exposed_record`); it has not been changed to a pass. Equality
is not applicable there. Attribution and information-sufficiency semantic grades
remain **unknown / not assessed** for all six cases. Response wording was
inspected unblinded by the implementing assistant, with no paid model judge.

## Persistence and execution

The durable trajectory uses actual PostgreSQL 16, a fresh owned table namespace
and a private SQLite outbox under the ignored output directory. It seeds two
authored primary fixtures, commits three read operations and retains exactly
those two primary identities throughout. This is not a model source-write quality
measurement. Cleanup drops only the invocation's owned tables.

A fresh memory instance rejects the old reference. New inventory references
work after restart without re-entering answer text. Identical completed operation
retries are a separate historical-result path, tested offline: the prior result
returns with `replayed:true` and zero new generation. A fresh operation using the
old reference still requests selection; changing its reference under the same
operation ID conflicts. Historical receipts never restore a live source binding.

The ON control completes one valid external critic review; OFF dispatches no
critic despite an ordinary detail call. AUTO cases dispatch no critic in this
sample. All **6/6** final execution reports agree with observed critic attempts,
rounds and latest validity. Completion validity is separate from semantic quality.

Regular and streaming selector behavior, changed/expired/superseded/duplicate
records, cross-instance/scope references, awaited approvals, actual queued-worker
dispatch and externally committed corrections are offline regression coverage.
The live study exercises nonstream JSON dispatch. A real terminal subprocess
inventory smoke also passes; no browser or Gateway selection UI is introduced.

## Cost and the remaining empty-detail observation

The study uses **16 / 32** admitted provider HTTP attempts: **5 support, 6 draft,
1 critic and 4 detail**, including the pre-change setup call and existing direct
fallback. Known usage is **11,506 input / 1,249 output tokens**, with no unknown
usage, transport failure, cancellation or denied admission. There is no warm-up,
extra rescue allocation, continuation, paid judge or old rescoring.

Fifteen envelopes end with `stop`. Occurrence 8 (zero-based), the first ordinary
detail call for the changed selection, ends with `length` and **empty content**:
1,355 input and 450 output tokens, all 450 output tokens recorded as reasoning.
The actual detail request has a 450-token budget under the existing production
policy; 2,048 is the experiment's ceiling, not every stage's request size.
The existing controller's empty-detail direct fallback makes one more detail
request and receives `stop`. The final answer remains the original selection
clarification. Both occurrences and costs remain in the journal and exact replay.
The ordinary read path does not classify this receipt as a complete-source or
critic validation failure; transport `failed_calls:0` does **not** mean every
stage produced usable text.

This suggests a separate next experiment: avoid additive consultation when a
bounded clarification is already sufficient, and expose empty/nonterminal
ordinary-detail outcomes without conflating them with transport or critic
failures. Preserve explicit ON/OFF authority and test it on fresh inputs. No
policy, stage budget or grading rule was tuned after these outputs.

## Replay and verification

Exact offline replay consumes all **16 ordered HTTP occurrences**, including the
empty `length` envelope and direct fallback, with **0 new provider calls** and
**0 unexpected blocked HTTP attempts**. Canonical final text, selection states,
source exposure, diagnostics, execution, commit outcomes and usage are identical:

`2a817e802b98e004a03259428ca4f13959beb243044de83f929534830d123c8f`

The metrics include raw artifact SHA-256 hashes and post-run envelope/canary
observations. Full prompts, journals, references and outbox remain private in
`samples/module-gating-record-choices-2026-11-06-{live,replay}`. The result commit
adds documentation and filtered metrics only; measured code and fixtures retain
their frozen bytes.

Local verification:

- Full Python suite with isolated PostgreSQL: **1,847 passed / 1 intentional
  local-HTTP deselection**, zero unexpected blocked provider requests.
- Final focused set: **37 passed**, including six actual queued-worker selection
  checks added after that full run and the Fake six-turn API/PostgreSQL replay.
- Gateway plus the intentional local-HTTP case: **18 passed / 2 platform skips**.
- Trace UI: **13 passed**. Real stdio inventory smoke passes without credentials.

The GitHub checks cover Python 3.10 consultation lifetime, the full Python/Node
suite, Gateway contracts and PostgreSQL atomic commits plus the new selector
and journal tests. All eight checks pass at the implementation freeze in the
[push](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/actions/runs/42194524476)
and [PR](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/actions/runs/42194532710)
runs. Review is implementing-assistant manual inspection, without
independent external adjudication or automated approval. The broader semantic
calibration in #89 remains open.

References are limited to one memory incarnation; listings/counts use the existing
bounded resident source view. Persistent selected requests refresh committed
sources at request start, then use the existing resident checks at generation
boundaries. Independent database writes are not serialized with an ongoing
generation, and already-emitted text cannot be revoked.
