# Record replay followup: reference repairs and independent assessment

2026-11-05 (Asia/Tokyo). The requested investigation of #86 began after
[PR #88](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/pull/88) had already
closed it. This followup audits that implementation, repairs two remaining
reference failures, and adds an independently inspectable four-axis assessment
contract and a deterministic development comparison.

Frozen code, fixtures, contracts and tests:
[`9db4cade14ec773e52efbb00c4ccab540887cf01`](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/tree/9db4cade14ec773e52efbb00c4ccab540887cf01).
Tree: `a4c51ef829fb2f1a73280d3fa4fde5e4cfa045f4`.
[Protocol](record-replay-followup-protocol.md) ·
[Safe results](record-replay-followup-2026-11-05.metrics.json).

## What the path audit found

#88 already carries a locally recognized quotation operation through exact
reference/source binding, a distinct support policy, complete-answer approval,
normal generation and guarded selective consultation. It does not need a second
implementation of that path. Two preceding reference checks still disagreed
with the existing bounded grammar:

| Reproduction | Cause | Repair |
| --- | --- | --- |
| `Repeat Project Blue Rowan recorded answer verbatim.` was recognized as a literal request but lost its resolved identity and subsequent focus. | The unknown-project check inspected only the first whitespace-delimited name token. | Check the resolver's complete, boundary-checked alias span after each explicit project marker. A marker inside a declared alias belongs to that alias; a separate following unknown project remains unknown. |
| `Repeat  that recorded answer verbatim.`, a tab-separated form, or a fullwidth English form lost the corrected source's focus. | The literal grammar normalized the request, but a separate raw-text continuation regex required single ASCII spaces. | Reuse the grammar's normalization only for recognition. Keep the original query bytes in support/cache bindings. |

Regression coverage includes `Blue Rowan`, `Amber 31`, `青 山`, names containing
an internal `Project`, actual correction followed by a Japanese pronoun, English
continuations, Unicode name extensions, explicit known-plus-unknown projects,
and extra verification/explanation clauses. The first added regression batch
reproduced ten failures before repair. No semantic intent classifier was added.

Current/complete/scope checks, unique source identity before ranking, exact
query/source/policy approval, source completion and manual System2 authority
remain in the production path. An unverified remembered answer remains
unverified. Generated or recalled answers do not become new source evidence.

## The final-answer contract is now independently inspectable

V2 already explains quotation semantics, but still emits only
`task / grounding / clarification`. The additive `answer_assessment_v3` API
leaves that contract, its callers and historical scores unchanged.

| Axis | How it is obtained | What it cannot establish |
| --- | --- | --- |
| `exact_match` | Deterministic UTF-8 equality against an explicit diagnostic target, with no stripping or Unicode normalization. | Availability, source validity or truth. |
| `source_validity` | Actual exposed source metadata, explicit guard outcomes, exact target binding for replay, and consistency checks. | Independent verification of recorded claims. |
| `attribution` | Its own strictly validated semantic judgment about faithful source use and verification claims. | Exact byte equality or source admission. |
| `information_sufficiency` | Its own semantic judgment about handling supplied, absent or ambiguous evidence. | Access to a hidden target or comparison reference. |

Known failed guards remain failures; missing/malformed inputs remain unknown.
Explicit superseded, truncated, derived, empty or expired source metadata cannot
be overridden by passing caller flags. Expiry uses the supplied snapshot time.
Ordinary multi-source fact questions use per-source guards and do not inherit
replay's single-record requirement. A recorded quote can match exactly while its
source is unavailable, and a faithful attribution prefix can fail exact equality.
There is no combined success score that erases these distinctions.

The semantic evaluator receives actual available context and the candidate
answer. It does not receive the equality oracle, comparison reference,
expectations or legacy grades; nested metadata is also filtered. Original stored
bytes remain the equality target even when sanitation withholds the source.
Malformed, refused, truncated or ambiguous assessment completions remain
unavailable rather than silently becoming failed judgments.

`legacy_model_grade` is copied unchanged. `equality_task_disagreement` explicitly
records its relationship to equality, without declaring every difference a
judge error. A task failure may be appropriate even when invalid source text
was copied exactly. Public diagnostics retain verdicts and disagreement while
omitting source/answer text, evidence quotes and receipt cache contents.

## Fixed Fake Provider comparison

This is a development control, **not a new live-model experiment or judge
calibration**. The provider's support choices, response behavior and semantic
verdicts are scripted in fresh fixtures. It copies only text actually exposed
to generation, never the diagnostic answer target. The runner has no live mode,
does not load provider credentials, and rejects uninstrumented HTTP locally.

Eleven fixed cases cover identity, corrected continuation, distractors,
lifecycle, completeness, denial, partial approval, extra clauses and ordinary
questions. Two arms receive three fresh support preparations each. Runtime uses
only the first exact bound receipt for its own scenario and arm, including a
denial; later approvals cannot replace it. Each arm generates its own answer,
so this remains a path ablation, not a pure consultation comparison.

| Observed control | Result |
| --- | --- |
| Planned support preparations | 66: 45 ready, 18 locally empty, 3 withheld partial-quote decisions. |
| Runtime answers and V3 packets | 22 available answers; 22 assessment packets. |
| Four intended positive replay scenarios | Complete stored answer reproduced exactly in all four typed-replay sessions. |
| Cross-project control | Only the declared target source was exposed. |
| Expired, superseded and incomplete sources | No source exposure; all 18 support preparations made zero transport calls. |
| Initial denial followed by approval | Later preparations approved, but runtime retained preparation zero's denial in both arms. |
| Partial quotation approval | All three replay checks withheld it; runtime received no source. |
| Extra-clause and semantic questions | No typed replay intent; ordinary support policy retained. |
| Predeclared expectation mismatches | 0. |

The comparison deliberately includes a **scripted** failing legacy task verdict.
Its four disagreements with exact replay remain explicit. These are checks of
the diagnostic contract, not four new observations of a model judge failing.

## Calls, timing and exact replay

The frozen execution used **111/192 scripted transport attempts** and **zero
real provider/network calls**:

| Role | Scripted attempts |
| --- | ---: |
| Generic support, all preparations | 24 |
| Replay support, all preparations | 24 |
| Left generation | 22 |
| Right critique | 19 |
| V3 semantic assessment | 22 |

The fake receipts contain 1,110 input / 555 output tokens. These are synthetic
fixed values, not billed tokens. No transport failed and no usage was unknown in
this successful matrix; the three partial support responses were rejected by
validation and remain visible. Dedicated failure tests separately retain one
failed attempt, unknown usage, all planned unavailable cells and their replay.

Observed local totals were 228.10 ms for support, 5,347.06 ms for runtime
handlers, and 137.83 ms for assessment. Generic/replay runtime totals were
3,058.60 / 2,288.46 ms with 22 / 19 scripted calls. These handlers reuse support
receipts; their timings are local fake-provider observations and cannot support
a production latency or cost-saving claim.

The second execution removed transport and consumed all **111 recorded
occurrences with zero new transport calls**. Canonical support outcomes and
bindings, answers, exposure, routes, all four dimensions, unchanged legacy
grades and usage matched. Comparison excludes local elapsed times and the
inherited replay layer's local exception-wrapper type. Failed original provider
categories are retained from immutable receipts; the wrapper type is separately
reported rather than replacing the failure. Both successful and failed paths
are covered by exact-replay tests.

## Verification and boundaries

Targeted production regression: **256 passed**. New V3 assessment contract:
**54 passed**. New comparison/failure-replay tests: **7 passed**. The broader
Python run during the repair passed 1,373 checks and skipped 16; the final four
internal-marker cases were subsequently covered by the 256-check production run.
Node trace UI tests: **13 passed**. Local Gateway checks: **2 passed / 17 skipped**.
PostgreSQL, Torch and .NET integration were not available in this local runtime;
repository CI exercises those configured environments separately. Test counts
from overlapping commands must not be added together. Whitespace validation is
clean, and independent reviews covered production references, assessment and
the comparison harness.

The #85 report's three byte-identical answers with failed model task grades stay
as originally recorded. This followup does not regrade them or claim to have
recomputed equality from their unavailable private journal. The #88 measured
reports, fixtures, V2 grader, benchmark and historical scores are unchanged.

Missing-parameter clarification remains the separate scope of #89; empty-critic
availability remains #87. No AUTO default, primary-write policy or truth status
was promoted. No live-model calibration, natural API latency, durable-write or
restart result is claimed here.

Private reproducible run artifacts remain under ignored
`samples/module-gating-record-replay-development-2026-11-05-frozen-{fake,replay}/`.
Their hashes and the source/fixture/contract bindings are in the safe results.

| Artifact | SHA-256 |
| --- | --- |
| Frozen fake report | `f1e19bfa0e1b03a2d9fa8f7a75a1ac648baffacbca93cf273eeebcf13c0fa5ad` |
| Journal | `512368807cb1f3599276d12b16c34799aaa09ecace825eb674d5ad5fcd3e18e3` |
| Frozen replay report | `8aeaa5d56303dcd5eca899e38d4041235c9077316a8997ffdbdd94a470ce9e66` |
