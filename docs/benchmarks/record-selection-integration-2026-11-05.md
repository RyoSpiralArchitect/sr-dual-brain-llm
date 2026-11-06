# Live confirmation of PR #93 — 2026-11-05

PR #93 was manually reviewed and merged at
`89347806b60b14ba3cc616ac21cd77ead5fa6031`. This study uses its reviewed
`b5d90f052f5963f37f87236f5156146d3ab3180a` core files **byte-for-byte**.
The verification runner, new inputs and protocol were frozen before paid calls
at `924b2304611790eb48757f0901bc6ebe5c6bfd46`.

**All three ambiguous requests ask which existing record is intended**, one
unique current quotation matches its actual source exactly, and ordinary factual
lookup retains the supported-value path. All five answers are available, all
five final System2 execution reports are consistent, and there are no failed
structural expectations. Semantic model-judge grades remain unknown.

## Provenance and allocation

A narrower prototype had already consumed 32 calls before the pre-existing
PR #93 was discovered. Its code and observations remain labeled with `fd69cf8`
in the [prototype report](record-selection-confirmation-2026-11-05.md). They are
not evidence for PR #93's broader generation-boundary guards. The overlapping
prototype runtime change was superseded by PR #93; this verification adds no
different runtime implementation. PR #93's core predates the prototype outputs
and was not tuned using them.

This integration run has a separate prospective [protocol](record-selection-integration-protocol.md),
[five-case fixture](../../sr-dual-brain-llm/examples/record_selection_integration.json)
and [safe metrics](record-selection-integration-2026-11-05.metrics.json).
Its cap is 32 attempts, keeping prototype plus integration within 64 total.
There is no warm-up, paid rescue, resampling, judge or old rescoring.

| Study | Actual attempts | Known input | Known output |
| --- | ---: | ---: | ---: |
| Earlier prototype, `fd69cf8` | 32 | 20,783 | 4,499 |
| **PR #93 integration, `924b230`** | **18 / 32** | **12,936** | **3,460** |
| Total for this work | **50 / 64** | **33,719** | **7,959** |

The integration's 18 attempts comprise 5 support, 5 draft, 3 critic, 2 ordinary
detail and 3 integration calls. All return terminal `stop` responses and known
usage; there are no transport failures, cancellations or unknown receipts.
The served model is `gpt-6-luna`. Setup support is costed separately and also
included in each case's total. These unpaired studies do not establish a cost,
latency or general quality improvement.

## Actual support and selection behavior

Fresh in-memory `EngineSession` cases use the existing all-AUTO module profile,
with Director/executive advice and metrics observation OFF. Source fixtures
remain unverified recorded claims; no new PostgreSQL write result is claimed.

For each ambiguous case, the actual backend checks the two prospectively listed
sources once under the ordinary policy before runtime. It returns **4 positive
and 2 negative decisions**; the caller/pronoun case denies both. These receipts
are retained without selecting a favorable repeat.

| Case | Ordinary support receipts | Literal final response |
| --- | --- | --- |
| English, explicit ON | period=true, backup=true | `Which complete Project Willow Cove record do you mean? Several current records exist, and I can’t determine which recorded answer to repeat.` |
| Japanese, AUTO | period=true, backup=true | `再掲する記録を特定できません。銅鷺のどの回答か、題名や日付などを教えてください。` |
| Caller-declared pronoun, explicit OFF | period=false, backup=false | `Which complete recorded answer do you mean? Please identify the specific record.` |

All three resolve to `ambiguous_record`, no selected binding, no exposed QA,
and `current_unselected / unknown / clarify_record`. Ordinary positive or
negative per-record support cannot choose the intended complete record or
establish that it is absent. The caller's declaration supplies identity only.

Stored-value canaries are absent from final text and from all runtime
generation/review/integration packets. The support model deliberately sees
source values. These are limited literal checks, not general semantic leakage
certification. The two approval-positive cases establish live coverage of the
original bypass; the denied case is reported separately, not relabeled positive.

The earlier prototype's optional `or paste it here` wording remains recorded.
None of these three integration answers requests pasting, but the inputs and
model outcomes differ; this is not a calibrated claim that wording friction is
universally solved. Inspection is unblinded by the implementing assistant, and
V3 attribution/information-sufficiency grades stay **unknown / not assessed**.

## Controls and execution accounting

The unique-current/expired-second control returns exactly
`Recorded retention: 25 days.` It passes exact UTF-8 equality and observed V3
source-validity checks, exposes only `current`, and does not turn the expired
record into false ambiguity. Its exact complete-source approval uses the typed
replay policy independently of ordinary approvals.

The factual control returns:

> The provided information says Project Slate Birch retains data for 26 days; I haven’t independently verified that figure.

It exposes the period record, has `not_requested` replay status and no selection
condition. The quote-specific guard does not displace the ordinary factual path.

The English ON case dispatches one valid external critic; the caller OFF case
dispatches none despite ordinary detail/integration traffic. The factual control
obtains two valid reviews. Across all five cases there are three actual critic
controller requests and three critic HTTP attempts, with no repair or retry.
All **5/5** execution reports agree with observed attempts/rounds/latest validity.
Completion validity is not semantic correctness.

## Replay, review and limits

Exact replay consumes all **18 occurrences** with **0 new provider calls** and
**0 unexpected blocked HTTP attempts**. It preserves actual positive/negative
support, source exposure, final text, diagnostics, execution and usage.
The canonical non-timing outcome SHA-256 is identical:

`59131a9db0d09560e5e10c60e5caa34bbcd040b525260f9af83e615e4b7f937d`

Raw journals, prompts and full traces remain private/ignored in
`samples/module-gating-record-selection-integration-2026-11-05-*`. Safe metrics
contain artifact hashes and every structural outcome; public excerpts are
fictional evaluation data. Original #92 and prototype scores were not rewritten.

Manual review of the exact PR #93 head ran **114 tests** with isolated PostgreSQL
and the offline HTTP guard: 89 new selection/resolution/generation/queued-worker
cases, 16 database commit cases and 9 journal/real-store controls. The integrated
selection/measurement set passes **117 tests**, including Fake execution and
exact replay for both fixture sets. Both record zero unexpected blocked HTTP.

The full local Python suite at the integration freeze passes **1,816 tests /
1 intentional local-HTTP deselection**, with isolated PostgreSQL enabled and
zero unexpected blocked HTTP attempts. All eight checks at the frozen head
passed in the [push](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/actions/runs/42031144412)
and [PR](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/actions/runs/42031151632)
runs. Incorporating the PR #93 main merge afterward changed no measured file
bytes; the following result commit adds only this report and safe metrics.

Queued changes, last-alias loss and refresh at later generation boundaries are
offline-regression evidence, not naturally sampled live events in these five
fixed cases. Existing limits on already-completed/streamed text and resident
source views remain explicit in PR #93's report. No automatic review or
independent external adjudication is claimed by this manual verification.

The next bounded extension is an explicit read-only source-selection key: bind
the record the user chooses, recheck its scope/lifecycle/completeness/approval,
and quote it without requiring pasted text. The broader semantic calibration in
#89 remains open.
