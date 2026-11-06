# Record-selection confirmation — 2026-11-05

**Prototype snapshot:** this report describes the narrower implementation frozen
at `fd69cf8`, before the pre-existing PR #93 was discovered and integrated. Its
results and original code remain reproducible at that commit. It is not evidence
for PR #93's complete generation-boundary guards. The subsequent integration
measurement has its own fixed inputs, allocation and report; no score here was
changed. PR #93's runtime uses `current_unselected` for ambiguous records.

The #92 pilot exposed two approved current records for an unspecific whole-record
quotation and returned one without asking which record was intended. #92 is now
merged at `b1a3d7fdae922c955b62208d7c13c9c3ac10a909`; its original failure and
metrics are unchanged.

The runtime now withholds QA and seed evidence while a bounded whole-record
request has multiple current primary records in its resolved scope. Factual
support receipts remain cached; they cannot choose the record for the user.
The existing `clarify_record` condition consequently reaches drafting, review
and integration. Candidate caps and source eligibility cannot manufacture
uniqueness; lifecycle/scope changes are rechecked after awaits.

Fresh real-model confirmation: **five ambiguous requests all ask which record
is intended**, despite **nine positive ordinary support decisions**. Both unique
quotation controls match their actual sources exactly. All nine turns return
answers and consistent final System2 accounting, with no failed structural
expectation. One clarification still offers optional pasting; this wording
limitation is retained below rather than converted into a semantic pass.

## Freeze, method and cost

- Runtime, protocol, fresh inputs and controls frozen before paid calls at
  `fd69cf82bfc5b1f2d93c37c08911002581c01d7c`.
- [Protocol](record-selection-confirmation-protocol.md),
  [fixture](../../sr-dual-brain-llm/examples/record_selection_confirmation.json),
  [safe metrics](record-selection-confirmation-2026-11-05.metrics.json).
- Real `openai/gpt-6-luna`, production nonstream `EngineSession`, fixed nine-case
  order, existing all-AUTO profile with Director/executive advice and metrics
  observation OFF. Source fixtures use in-memory sessions; prior PostgreSQL
  evidence remains the separate #92 study.
- For the five ambiguous cases, one predeclared support preparation per case
  uses the actual backend and ordinary query/source policy. Both records are
  submitted in four cases; only the eligible record is submitted in the cap-1
  oversized-source case. **All 9 decisions are true.** Runtime reuses exact
  ordinary cache entries but still withholds the ambiguous records.
- No warm-up, resampling, paid rescue, judge calls or post-output change to the
  measured implementation, conditions or criteria.

| HTTP stage | Attempts |
| --- | ---: |
| Support, including all five setup preparations | 9 |
| Drafts | 9 |
| Critic | 3 |
| Ordinary detail | 5 |
| Integration | 6 |
| **Total** | **32 / 64** |

Usage is **20,783 input / 4,499 output tokens**. All 32 responses have `stop`
termination and known usage; transport failures, cancellations and unknown
receipts are zero. There are no repair attempts. Setup costs are included in
each case's total usage as well as the overall total, with runtime usage and
stage counts also retained separately.

The previous 33-call study had different inputs and more turns. These totals
are not a paired cost/latency comparison or a general performance improvement.

## Actual selection responses

The following are literal excerpts inspected against the frozen qualitative
criteria. Inspection is unblinded and performed by the implementing assistant;
V3 attribution and information-sufficiency grades remain **unknown / not assessed**.

| Fresh case | Actual ordinary approvals | Actual response |
| --- | --- | --- |
| English, explicit System2 ON | period=true, backup=true | `There are several current records for Project Moss Harbor. Which complete record do you mean?` |
| Japanese, AUTO | period=true, backup=true | `黒鳶には複数の記録があります。どの記録を再掲すればよいか、タイトルや内容の手がかりを教えてください。` |
| Pronoun after focus, AUTO | period=true, backup=true | `Which recorded answer do you mean? Please identify the record or paste it here.` |
| Caller-declared scope, explicit OFF | period=true, backup=true | `Which recorded answer do you mean? Please identify it by its title, date, or another distinguishing detail.` |
| Cap 1, second source oversized | period=true; oversized record not checked | `I have several current records for Project Cobalt. Which complete record should I repeat? Please identify it by title or date.` |

All five have `clarify_record`, no replay binding, and no exposed source QAs.
The two current primaries are counted before eligibility and ranking. Stored-value
canaries appear in neither final text nor generation/review/integration packets;
the support model deliberately sees source values. Literal canaries are limited
diagnostics, not general semantic leakage certification. Seed delivery is also
blocked by the same runtime gate and covered by the offline regressions.

Ordinary per-record approvals remain visible in the support receipts. The
`missing_evidence_v1` packet describes delivery for the whole request: no unique
record is selected, so its aggregate approval stays unknown and its availability
is `current_unapproved`. This does not mean those ordinary factual checks failed,
the records disappeared, or their contents were proven true.

## Controls and authority

| Control | Observed behavior |
| --- | --- |
| Unique current quote | `Recorded retention: 10 days.` — exact UTF-8 source match, V3 observed source-validity pass |
| Current quote plus expired second record | `Recorded retention: 15 days.` — exact match and source-validity pass; no false selection ambiguity |
| Ordinary factual lookup with two same-project records | `The supplied record lists Project Indigo Maple’s data retention period as 22 days.` Only the period source is exposed; no forced record-selection condition. |
| Extra-clause request | `Recorded answer: “Recorded retention: 28 days.”` followed by the requested summary. Ordinary QA path, no manufactured bounded replay intent; exact-copy scoring is not applicable. |

The explicit ON ambiguous case dispatches one valid external critic. The caller
case with explicit OFF dispatches no critic and can still use ordinary detail
and integration. Across all nine turns, three actual critic requests (including
the factual control's reviews) return valid external reviews. Controller request
counts happen to equal critic HTTP attempts because no retry or repair occurs.
All **9/9** final execution reports are consistent. Completion validity is
separate from semantic correctness.

## Residual limitations and next scope

The pronoun case correctly asks for record identity but adds `or paste it here`.
That is an optional alternative, not a claim that stored records are absent or
a mandatory re-entry requirement. It still creates avoidable friction when
records already exist, so this response is retained as a wording limitation.
The structural checks do not grade that sentence as semantically perfect.

The current guard handles the existing bounded whole-record grammar and current
resident source view. General semantic intent classification and extra-clause
quotation disambiguation are outside this intervention. Primary writes and
ordinary factual evidence paths retain their existing authority.

The next extension is explicit read-only record selection: after asking which
record, a caller should be able to bind a chosen source identity within the
resolved scope, recheck current/complete/approval guards, and quote it without
requiring the user to paste the text. This needs its own caller contract and
fresh measurements. #89's broader semantic-judge calibration remains open.

## Reproducibility and validation

Exact offline replay consumes **all 32 occurrences**, with **0 new provider
calls** and **0 unexpected blocked HTTP attempts**. Support receipts, answers,
conditions, exposure, accounting, usage and the optional-paste wording all match.
Both runs have canonical non-timing outcome SHA-256:

`7c4bf3de8cc123b87cb6523a5094bfc2ce258038af77fa728a52faf0b1728abb`

Original prompts, raw journals and full traces remain private/ignored in the
two `samples/module-gating-record-selection-2026-11-05-*` directories; safe
metrics retain their artifact hashes. Public excerpts contain only fictional
evaluation data. No prior study was overwritten or regraded.

New selection regressions: **26 passed**, covering approval-positive requests,
checker absence, English/Japanese/pronoun/caller identity, caps/ineligible
records, lifecycle/other scopes, late insertion during an awaited full approval,
cache-preserving expiry recovery, primary writes and both brains/final
integration with ON/OFF. Six initial approval-positive variants reproduced the
old delivery failure before the runtime fix.

Related selection/memory/replay/measurement controls: **181 passed**. Full local
Python suite: **1,726 passed / 1 intentional local-HTTP deselection**, isolated
PostgreSQL enabled, zero unexpected blocked HTTP attempts. The final fixture's
full Fake-provider execution and exact replay also pass separately. These Fake
controls do not contribute to the 32 real-call usage or establish natural
semantic quality. CI runs the full suite, PostgreSQL controls, Gateway and Node.
