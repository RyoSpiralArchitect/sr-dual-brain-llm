# Literal memory replay: first measurement

2026-11-05 (Asia/Tokyo), issue #86. A locally bound quotation operation carries
the corrected record through support checking, context and normal generation.
All three replay scenarios produce the exact approved stored answer. The generic
contract rejects the corrected Japanese followup; the replay contract accepts it
in all three measured preparations. No evidence/completion guard is bypassed.

Frozen source: `23ad3146916b604d2dbe49cb0271dc4928786144`.
[Prospective protocol](record-replay-protocol.md),
[safe metrics](record-replay-2026-11-05.metrics.json).
Source, fixtures, expectations, wording/schema, order, repeat-zero receipt reuse
and the shared 96-attempt cap were committed before paid calls. Old #85 outputs,
decisions and grades remain unchanged. There was no paid retry, rescue judge,
replacement or post-result expectation/rubric edit.

## Answer-grader contract

Four independent authored cases × three variants × two label orders × two rubric
arms consume all sixteen calibration attempts. Both arms see identical explicit
requirements and actual available evidence. The new contract distinguishes exact
historical quotation, factual truth, changed text, missing current evidence and
ambiguous identity. An unverified record can support a faithful requested quote
without becoming a verified fact or requiring extra words in that quote.

| Fixed-expectation agreement | Original answer rubric | Quotation V2 |
| --- | ---: | ---: |
| Task | 22/24 | 24/24 |
| Grounding | 24/24 | 24/24 |
| Clarification | 24/24 | 24/24 |
| Total | 70/72 | 72/72 |
| All-criteria matching presentations | 22/24 | 24/24 |
| Criterion/variant order disagreements | 0 | 0 |

Both paired corrections are the same corrected-record exact quote in its two
presentations: V1 fails task despite literal source equality, V2 passes. There
are no paired regressions or invalid/missing packets. Exact current/unverified
quotation, changed/stale text, added framing and targeted missing/ambiguous-source
clarification meet the predeclared focus. Twelve variants and repeated labels do
not establish general judge correctness or order robustness.

V1 uses 5,502 input / 2,155 output tokens; V2 uses 7,782 / 2,250, increases of
41.4% and 4.4%. The explicit contract has a cost. Historical answer/private-advice
graders and callers are not replaced. Equality is an independent diagnostic,
never a post hoc rewrite of model grades.

## Repeated support factor

Each scenario/arm uses three fresh checker/gate instances. These 36 preparations
produce 30 HTTP calls; the six ineligible incomplete-source preparations make
none. Each positive replay approval must reproduce the complete answer exactly.
Every measured rejection/failure would remain eligible for repeat-zero reuse;
none is replaced by a later or majority approval.

| Scenario | Generic contract, delivered IDs over 3 preparations | Replay contract |
| --- | --- | --- |
| Explicit Juniper quotation | Correct source 3/3 | Correct source 3/3 |
| Actual Alder correction + Japanese pronoun | No source 3/3 | Corrected source 3/3 |
| Cypress among another project's records | Correct source 3/3 | Correct source 3/3 |
| Expired Birch retention + unrelated current owner | No retention source 3/3 | No retention source 3/3 |
| Literal phrase plus an extra verification clause | Ordinary fact support 3/3; no replay intent | Same; no replay intent |
| Incomplete/oversized answer | Empty preparation 3/3 | Empty preparation 3/3 |

Expected delivered-source-set agreement is 15/18 preparations generic and 18/18
replay. Each scenario's three outcomes agree internally in both arms; these new
cases do not estimate the population frequency of the earlier #85 fluctuation.
The expired value never reaches an answering model. The extra-clause case is not
rewritten into quotation intent or interpreted as proof of a completed live test.

## Production answering with bound receipts

Four scenarios each run one real EngineSession per arm. Normal SharedMemory
operations implement declared correction/focus setup. These are authored source
records, not generated durable primary writes. Each runtime preparation imports
only that scenario/arm's **first measured** support cache entries when query,
source, intent, policy and backend identity match. Missing/false entries withhold
evidence; no new support request is allowed in the runtime phase. No expected
answer is installed into an approval cache. Support cost stays in the full ledger.

| Outcome | Generic contract | Replay contract |
| --- | ---: | ---: |
| Planned / available / graded answers | 4 / 4 / 4 | 4 / 4 / 4 |
| Exact stored answer on the 3 quotation scenarios | 2/3 | 3/3 |
| Task pass | 2/4 | 3/4 |
| Grounding pass | 4/4 | 4/4 |
| Clarification pass | 3/4 | 3/4 |
| All-criteria pass | 2/4 | 3/4 |
| Runtime provider attempts, excluding reused support | 7 | 4 |
| Runtime right-critic attempts | 3 | 0 |

The corrected quote is the bounded positive result: the replay checker approves
the new four-day record, the JSON context specifies its answer text, normal
generation reproduces it, and the existing AUTO router recognizes complete
literal equality. Neither old record revival nor independent verification is
claimed. Explicit System2 ON still consults in end-to-end regression tests;
primary-write success and other mandatory boundaries remain in place.

The two arms generate natural initial drafts independently. They share the
proposed stricter controller, with the generic arm's typed intent disabled. This
is a **feature ablation**, not an exact old-release comparison or a pure routing
effect. Old versions could skip some literal lookups under their previous proof;
the observed 7-versus-4 call difference cannot establish release-to-release savings.
There are three distinct quotation scenarios, not a statistically representative
multilingual workload. Qualitative copy handling is regression-tested offline,
not separately demonstrated by these numeric live records.

Both expired-data outputs fail task/clarification under the fixed judge and pass
grounding. They ask about the retention measure/source rather than specifically
requesting a current data-retention source. The bare user word "retention" can
itself mean a rate or a data-retention policy once the expired record is withheld.
That makes the clarification/rubric boundary a real uncertainty, not a confirmed
unsafe-answer defect. Keep both failed grades; calibrate fresh explicit versus
underspecified parameter cases before changing the runtime or this verdict.

## Full cost and latency limits

All **65/96** allocated attempts are retained: 16 calibration, 30 support,
8 left generation, 3 right critique and 8 runtime judging. Production/support use
`gpt-6-luna`, judges `gpt-6-sol`. There are no failed/cancelled attempts, unknown
usage receipts or paid replacements. Total: **35,815 input / 11,677 output tokens**.

| Slice | Calls | Input / output tokens |
| --- | ---: | ---: |
| Generic support factor, all repeats | 15 | 4,317 / 3,029 |
| Replay support factor, all repeats | 15 | 5,784 / 1,389 |
| Generic runtime with reused receipts | 7 | 4,950 / 1,309 |
| Replay runtime with reused receipts | 4 | 1,425 / 151 |

The replay support contract uses more input text and fewer sampled output tokens.
All repeated support, calibration and final grading remain in total cost. Cached
runtime handlers total 19.52 s generic and 4.25 s replay (medians 4.50 s / 0.93 s).
Those handlers **exclude the original support HTTP time**. Individual repeat-zero
support durations remain in safe metrics. Do not present cached handler durations
as natural end-to-end production latency, a repeated-measure speed estimate or a
historical-version improvement. Provider responses remain stochastic.

## Replay and final hardening

At the frozen source, replay consumes all 65 recorded occurrences with transport
removed and **zero new API calls**. Calibration, support outcomes and exact bound
receipts, natural answers/status, exposed sources/routes, final grades and usage
match exactly. Real elapsed time is excluded. Replay checks the recorded path,
not independent grading accuracy or model stability.

The complete frozen-source preflight exposed one OFF regression (1,357 passed,
one failed): router OFF still prepared the new intent facts. The final controller
checks gate validity/mode before preparing them. A separate reproduced boundary
showed that deleting an alias named "Explain" could erase an extra user clause.
The final parser retains one identity's position and admits only explicit identity
slots; multiple identity matches and aliases occupying extra command clauses
cannot create quotation intent. Normal English/Japanese/possessive forms are
covered. These corrections implement the original OFF/extra-clause contract;
the frozen measurements and scores are not rerun or relabeled.

Final regression coverage includes query/source/policy cache separation, denied
or partial full-answer quotes, expired/superseded/derived/duplicate/incomplete
sources, uniqueness before ranking, real correction/focus carryover, exact
generation through the production process, explicit System2 modes, gate OFF and
alias-position controls. Existing real PostgreSQL and Gateway/UI suites verify
their separate boundaries. Live seeding here does not establish commit/restart
durability. Full final-head local and CI results are recorded in the PR.

Private artifacts: ignored
`samples/module-gating-record-replay-2026-11-05-{live,replay}/`.

| Artifact | SHA-256 |
| --- | --- |
| Live report | `5d8251348785066a9daa532b6399d0b1742365953aeb1a7614b8d7b5bb52766b` |
| HTTP journal | `945261752ccbe74705e4de4257ce3e72bf38828a51bab8e45776c74911ec6e4c` |
| Safe original summary | `fd79734e4c5bd69e6e64876e6b6b666a8e6c3b30180c8024fd973ec1ff39694c` |
