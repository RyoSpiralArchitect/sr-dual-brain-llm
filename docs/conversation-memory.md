# Caller-assisted conversation memory

The stdio engine supports opt-in, explicit project memory ingestion:

```bash
export DUALBRAIN_MEMORY_ANSWERABILITY_MODEL=gpt-6-luna
export DUALBRAIN_MEMORY_CONVERSATION_CONTEXT=1
```

The conversation flag applies to the metadata-capable stdio engine. Interactive
`run_server.py` retains its existing memory policy because stdin supplies no
trusted write/correction metadata.

The support flag sends historical Q&A to the configured provider and incurs cost.
The conversation flag changes which answers become retrievable primary records.
Both are disabled by default. No automatic scope or correction extraction is
provided. Callers using the controller directly can construct
`SharedMemory(conversation_context=True, answerability_gate=...)`.

Supply `memory_context` in `process` or `process_stream` params (or as a controller
keyword), together with a unique `qid` for each turn:

```json
{
  "qid": "deadline-update",
  "question": "Project Alpha's deadline is now seven days. Repeat and remember it.",
  "memory_context": {
    "id": "alpha",
    "aliases": ["Alpha", "アルファ"],
    "remember": true,
    "supersedes": ["prior-deadline-source"]
  }
}
```

Scope metadata is trusted application input, not model output. `remember: true`
declares a primary source write; ordinary answers remain in QA history but are
excluded from factual retrieval in this mode. Source answers remain unverified.
Primary source drafts require a configured left model, nonempty text and a confirmed
successful terminal completion. Provider errors, empty output, exhausted token-limit
continuations and missing or unsuccessful finish reasons fail the write. No new source
or supersession is stored on failure; concurrent preview/observer tasks are cancelled.
Streamed partial text may already have reached the caller, but it is not promoted.
The built-in left model requests `require_complete=True` from its LLM client;
custom client adapters must enforce that contract, including after continuations.
Hugging Face source writes request generation details and require a successful
`details.finish_reason`; responses without this metadata cannot become sources.
Custom left models must implement `generate_source_answer(...)` with this success
contract. The controller keeps a request-local record of confirmed answer text,
and checks the exact final stored answer before any memory writes. Completion
requirements also apply to later LLM integration and right previews within a
primary write. A fallback is eligible only when it retains an already confirmed
answer unchanged; an assembled local fallback or a director-truncated answer is
rejected. Complete polishing and complete right previews remain eligible. Custom
later-stage adapters must call `core.source_completion.record_completed_answer`
only after validating their own terminal success, or retain the verified draft.
Ordinary answer turns retain their existing fallback behavior. Successful generation
does not certify factual truth or semantic completeness.
Only explicit, unique, still-current source IDs in the same scope can be superseded;
stale/already superseded and expired targets are rejected. No records
are persisted as superseded until the new answer is stored. Pending correction
targets are already hidden from every SharedMemory lookup during that turn; a
failed/cancelled write leaves their stored status unchanged. PostgreSQL persists the new source
and old status changes in one transaction; legacy rows default to primary,
unverified records. The conversation working set prioritizes active primary sources over derived,
expired and superseded entries when it reaches its 1,024-entry cap. PostgreSQL
session creation and post-commit reload use `load_conversation_sources` to filter
those entries before the cap. Ordinary QAs remain durable in the history API,
but are not restored into this evidence working set. Missing origin metadata
retains the existing legacy primary default. More than 1,024 active sources still
require a separate retrieval policy; the cap is not an unlimited archive search.
The in-session focus is deliberately not restored on restart.
Persistent stdio sessions use a durable local outbox and PostgreSQL receipts so
an uncertain commit can be retried without duplicating or reapplying a correction.
See [memory commit recovery](memory-commit-recovery.md) for operation IDs, restart
requirements, compatibility boundaries and isolated database validation.
Existing rows without origin metadata retain their legacy primary-source default.
Use a fresh session or explicitly curate historical records when enabling this
mode; it cannot identify previously saved answers that were derived from recall.

In this mode historical factual text is delivered only through the SharedMemory
source route. Raw working-memory, hippocampal/replay, schema and pitfall summaries
lack exact source/lifecycle approvals and are suppressed in model contexts, even
under director `keep` directives. Those components still index history and compute
control signals. Cortical line filtering is bypassed for complete approved Q&A so
it cannot remove late qualifications. This deliberately limits other conversation
history hints; the controlled quality pilot does not measure the full engine's
answer quality. Offline engine regressions capture actual model prompts with all
components instantiated and verify that old/answer-only records do not enter them.

An explicit project mention sets a single focus. Conservative follow-ups starting
with phrases such as `その`, `前の件`, `that` or `the previous` inherit that focus
only while its alias remains uniquely registered. A comparison, unrelated turn,
unknown project or ambiguous alias clears focus. This is a small phrase resolver,
not general pronoun understanding. The resolved cue is shared by support checking,
direct retrieval and latent seed relevance. The user question is preserved.

Whole-record quotation uses the existing bounded literal grammar, including
`Repeat Project Blue Rowan recorded answer verbatim.` and
`その記録をそのまま再掲して`. Declared multiword aliases are matched as complete
identities. Bounded English continuations such as `Repeat that recorded answer
verbatim.` use the grammar's same Unicode and whitespace normalization when
carrying focus; an added instruction is not removed by that normalization.
One current, complete primary record in the resolved scope must still receive
its own exact query/source/policy support approval. The quotation operation asks
for the recorded text, without certifying its claims as true. Ordinary semantic
questions retain their existing support policy.

`replay_resolution(question)` keeps request recognition separate from binding
to an exact source. Its immutable result is recomputed at each lookup:

| Binding status | Meaning |
| --- | --- |
| `not_requested` | This question is outside the bounded whole-record operation |
| `unresolved_reference` | The operation is recognized, but its current reference is unresolved |
| `missing_record` | The reference resolves, but its scope has no current primary record |
| `ambiguous_record` | The scope has more than one current primary record |
| `unavailable_binding` | A unique record exists, but the complete-source binding requirements cannot be met |
| `bound` | One current eligible complete source is bound; its exact replay support approval is still required |

`replay_intent(question)` remains the compatibility projection of that binding.
All current primary records in the resolved resident scope count before
eligibility filtering, relevance ranking, candidate limits or support decisions.
If two records remain, neither an ordinary positive decision for one nor a
negative/unknown decision for the other selects the intended record. Recognized
unbound replay withholds source bodies from direct retrieval, support candidates
and latent seed evidence, and preserves the appropriate clarification condition.
The disabled-checker path also withholds an unapproved complete-record replay.

Within an active turn, only the fact that its exact original/rewritten question
matched the existing bounded grammar is retained. This permits revalidation
after the last source and alias expire; it does not retain source identity,
scope, content, eligibility or approval. That request flag is cleared on the
next turn, abort or committed-memory reload. No historical alias registry,
unknown-name grammar expansion or semantic classifier is added.

Before existing draft and consultation generation boundaries, the controller
rechecks the current binding and retains only an already admitted, still-approved
exact memory block. Copied seed evidence is revalidated as well. Final revision
receives the current clarification condition, including when an earlier draft
contained a now-unavailable quote. Drafts and reviewer notes do not establish a
new selection or approval. The right-worker consumer uses the same check after
queue/admission waits, immediately before its existing generation call. This
adds no model calls or forced review rounds. It checks inputs to the next
existing generation; it does not revoke an already emitted answer or regenerate
a completed draft when no further generation is scheduled.
See the [record-selection followup protocol](benchmarks/record-selection-followup-protocol.md)
for the frozen approval-positive regression comparison and Fake Provider checks.

The [reference and assessment followup](benchmarks/record-replay-followup-protocol.md)
defines a separate development comparison and an additive answer-assessment
contract. Exact UTF-8 equality, source validity, attribution and handling of
information sufficiency are reported independently, alongside unchanged legacy
model grades. The development comparison uses scripted Fake Provider responses;
its results measure path and contract behavior, not model-judge reliability.

Complete source questions and answers up to 2,048 characters each can be checked.
Larger or sanitizer-modified records are withheld entirely. Both right-worker
paths accept an explicit `complete_v1` packet bounded to three entries and 16,000
characters; the default packet retains its previous limits. Conditions at the end
of a record are retained. Arbitrary document segmentation is not implemented.

Exact repeated query/source/model/policy combinations reuse the session support
cache. Generated probe answers cannot create fresh primary candidates and force
rechecking. Similar but different questions still require their own check.
Adding or revising a source checks only missing exact decisions for that question;
unchanged positive and negative decisions are reused. No approved source text is
shortened to reduce transmission. Cached support never overrides source expiry,
supersession, scope rules or pending replacement exclusions.
The [model-free volume experiment](benchmarks/incremental-memory-support-2026-10-21.md)
measures this reuse on growing and changing candidate sets.
These controls do not prove the factual correctness of stored model answers or
prevent every unsupported final answer.

See the [controlled evaluation protocol](benchmarks/conversation-memory-protocol.md).

The [real PostgreSQL retention experiment](benchmarks/persistent-memory-2026-10-26.md)
covers growing derived history, restart recall, corrections and exact receipt
replay without provider calls. It also reports latency and live-retention costs.

Conversation contexts also carry identity-only reference resolution independently
of factual QA approvals. A known project with no approved evidence for a requested
relation should elicit uncertainty about that fact, not a repeated request to
identify the project. The metadata contains only resolution state, no project
IDs, source text or values; alias ambiguity and source lifecycle still apply.
Supported conversation records may be reported with source attribution and the
appropriate uncertainty, without claiming independent verification. Both answer
and critic guidance distinguish that use from asserting verified facts: lack of
external verification alone should lead to an attribution caveat, not deletion of
otherwise supported values and conditions. Evidence status and support approvals
are unchanged; unsupported values and contradictions must still be challenged.
See the [persistent final-answer pilot](benchmarks/persistent-answers-2026-10-28.md)
for measured development failures, repairs, and the frozen evaluation results.

## Explicit missing-evidence conditions

A validated `memory_context` declaration also supplies identity for its own turn
when `remember` is false and there is no current record. It is kept in an
identity-only resolver; factual retrieval, source candidates and approval cache
keys still use the existing source resolver. Declarations are cleared at the next
turn, abort or committed-memory reload. They do not restore expired aliases for
later undeclared turns. Current alias conflicts are rechecked after awaits.
If a new alias resolves the question to a different scope, the reference remains
ambiguous; the surviving caller declaration cannot silently change identity.
Unresolved or conflicting declarations withhold factual candidates and cached approvals for
that turn without requesting another support decision.

When no approved source is available or a whole-record target is unbound, a bounded request can carry a
`missing_evidence_v1` condition through prefrontal context, both brains and final
integration. The packet contains enum values only: reference state and basis,
operation, explicit parameter (if any), parameter status, source availability/approval and
the requested clarification action. It contains no project/source IDs, historical
questions, answers, units or values. `none_current` describes the resident source
view for a resolved scope; `current_unapproved` means resident current sources
exist but none is approved for this exact request; `current_unselected` means
several current primary records remain for a whole-record operation, regardless
of ordinary support decisions. Its exact replay approval is `unknown` because
no target is bound. `unknown` availability is retained when the reference cannot
be resolved. None of these states establishes factual truth or
the absence of a fact outside that view.

The exact cached source decision distinguishes approval, explicit denial and
unknown. A failed/unprepared checker, changed backend identity or ineligible
record remains unknown, so it cannot be reported as a confirmed missing value.
An aggregate denial requires exact negative decisions for every resident current
primary record in the resolved scope. Unknown records outside the support
candidate cap, or filtered out as ineligible, keep the aggregate unknown even
when all checked records were denied.
`allows(...)` remains true only for an exact positive approval. No new support
request is made to construct a clarification condition.

| Explicit request and available identity | Clarification action |
| --- | --- |
| Resolved project and explicit parameter, no current source or explicit denial | Request a current source for that parameter |
| Resolved project, bare `retention` | Ask which measure is intended |
| Resolved project and explicit parameter/literal request, current records with unknown approval | Preserve uncertainty about support; do not claim the stored value is absent |
| Unknown or ambiguous reference in a bounded lookup | Ask for the intended reference/current source |
| Resolved project, bounded literal replay, several current records regardless of ordinary approval | Ask which complete record is intended |
| Resolved project, bounded literal replay with no current record or explicit denial | Request a current complete record |

Recognition is limited to whole question shapes such as `Project Cedar data
retention period?`, `What is the data-retention period for Project Cedar?`, `What
is Project Cedar's retention rate?`, and `Project Cedar retention?`. Japanese
forms include `Cedarのデータ保持期間は？`, `Cedarのデータ保存期間を教えてください。`,
`Cedarのリテンション率は？`, `Cedarの保持率は？`, and `Cedarのリテンションは？`.
The existing Unicode/whitespace normalization is used only for recognition.
Declared aliases may contain multiple words; unknown single-word `Project` names
can trigger reference clarification but do not become registered scopes.

An authoritative declaration can supply the identity for bounded requests such
as `What is its data retention period?`, `That project retention rate?`, or
`そのプロジェクトのデータ保持期間は？`. Parameter identity comes only from the
explicit wording. Bare `retention` stays underspecified even when an unavailable
historical answer contained days or percentages. Extra clauses, unsupported
wordings, ordinary unqualified questions without a reference, and explicitly
named multiple projects receive no new targeted condition. The existing literal
replay grammar retains its own operation and full-source requirements. Primary
source writes and ordinary semantic questions with approved evidence keep their
existing paths.

`tests/test_missing_evidence.py` exercises these boundaries with fixed authored
questions and a Fake Provider. Its EngineSession cases capture the actual
generation, consultation and integration inputs with System2 ON/OFF, including
expired records, caller-declared pronouns and the three retention wordings. HTTP
transport is blocked. The scripted answers verify the packet's delivery and
branch behavior; natural clarification quality and semantic judge calibration
remain separate measurements. Earlier study fixtures, outputs and grades are
unchanged.
