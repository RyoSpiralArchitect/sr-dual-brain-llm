# Live conversation-memory retest — 2026-10-22 (Asia/Tokyo)

The new candidate-growth suite scored **16/16 for both whole-batch and incremental
conversation memory**. Incremental checking used **12,546 support input tokens
versus 16,594 (24.4% fewer)**, with the same fourteen checker calls. This measures
quality preservation and lower retransmission, not a demonstrated score increase.
The historical suite confirmed 15 passing answers for each conversation arm;
one of its sixteen questions remains ungraded because the grader returned an
invalid four-answer JSON structure. It is not reported as 16/16.

The [protocol](live-memory-2026-10-22-protocol.md), runner and new fixture were
committed at `d673d33` before paid calls. The [public metrics](live-memory-2026-10-22.metrics.json)
contain exact usage, per-probe decisions, source/fixture hashes, artifact hashes,
grader diagnostics and replay verification. No runtime or rubric was tuned after
either suite's live answers.

## Final-answer results

All 96 final probe answers were generated successfully. The two suites each use
two authored Japanese dialogue episodes; growth includes an English source policy.
Grades require both task completion and grounded assertions. Correct admissions
of missing information and clarification after ambiguous references count as
successful tasks when requested. Grading uses the same model family as generation.

| Suite / arm | Confirmed passes / probes | Graded | Ungraded | Grounding failures among graded answers |
| --- | ---: | ---: | ---: | ---: |
| Historical / memory off | 4/16 | 15 | 1 | 0/15 |
| Historical / current non-conversation checker | 5/16 | 15 | 1 | 7/15 |
| Historical / conversation, whole-batch | 15/16 | 15 | 1 | 0/15 |
| Historical / conversation, incremental | 15/16 | 15 | 1 | 0/15 |
| New growth / conversation, whole-batch | 16/16 | 16 | 0 | 0/16 |
| New growth / conversation, incremental | 16/16 | 16 | 0 | 0/16 |

The historical questions are unchanged from the earlier 16/16 conversation pilot.
Fresh sampling and incomplete grading prevent a claim of either improvement or
regression against that historical score. The matched whole-batch/incremental
comparison preserves all fifteen gradeable historical tasks and all sixteen new
tasks. Uncertainty about an ungraded answer is not converted into a model failure
or a successful result.

The new suite interleaves five writes with eight probes per episode. Repeated
questions survive an additional long policy, a distractor about another relation,
and an explicit correction. Compound questions retain two-condition exceptions:
European individual versus corporate retention, and urgent versus normal Asian
transfers. Corrected values, project comparisons and ambiguous follow-ups also
pass. The long source answers contain 835 and 1,476 characters; their conditions
are retained within the existing complete-source limits.

## Measured support usage

| Suite / conversation arm | Checker calls | Records sent | Input tokens | Completion tokens |
| --- | ---: | ---: | ---: | ---: |
| Historical / whole-batch | 10 | 12 | 10,199 | 817 |
| Historical / incremental | 10 | 12 | 10,199 | 803 |
| Growth / whole-batch | 14 | 38 | 16,594 | 1,801 |
| Growth / incremental | 14 | 28 | 12,546 | 1,621 |

The growth suite saves **4,048 input tokens**, and transmitted records fall
**38 → 28 (26.3%)**. Savings occur only where an exact question recurs after the
candidate set changes. These input totals sum both episodes:

| Probe stage | Whole-batch input | Incremental input |
| --- | ---: | ---: |
| After policy addition | 2,459 | 2,347 |
| After unrelated fact addition | 2,645 | 714 |
| After explicit correction | 2,679 | 674 |

Initial and new questions still check every eligible candidate; their input usage
matches. The historical suite writes all sources before probing and therefore has
no mixed-cache retransmission to remove: its support input is identical between
conversation arms. These observations explain where the optimization helps.
The current non-conversation checker uses 7,426 input tokens on the historical
suite, but misses long policies and admits unsupported answers. Complete-source
checking still has a first-check cost; the optimization does not remove that cost.

Across both suites, actual provider usage is **208 calls, 132,915 input tokens and
24,682 completion tokens**, including the invalid grader response. Requested and
served model names are `gpt-6-luna`. There were no provider failures, embeddings,
automatic retries or continuations. The hard allocations were 144 + 112 = 256
calls; actual usage was 122 + 86. Source generation, answer generation, support
and grading are separated in the metrics. Currency savings are not calculated.
Latency is descriptive only: arm order is fixed, and the independent suites ran
concurrently; this is not a controlled latency experiment.

## Weaknesses exposed

One historical grading request, `petal-solstice:b-exception`, required labels
A–D. The response placed C and D inside B, leaving only A and B at the required
level. Exact label/criterion validation rejected that entire packet. All four
answers for the question remain ungraded; valid-looking nested portions were not
salvaged, and no paid regrading or favorable rerun was performed. The next grader
improvement should enforce the complete output schema at the provider boundary,
while preserving invalid attempts and their cost. That improvement is not applied
to these scores.

A post-hoc qualitative inspection also found unnecessary clarification: both
incremental unknown-fee answers admit missing fee information but then ask which
project is meant, despite the question naming a project explicitly. The fixed
rubric accepts their grounded uncertainty, so this conversational friction is not
reflected in the 16/16 score. It also occurs in one whole-batch answer; this small
sample does not attribute it to caching. A future disjoint evaluation should
distinguish an unknown fact from an ambiguous project reference. No criteria or
scores were changed to penalize these answers after inspection.

## Validation and limits

Both complete suites replay from exact private HTTP journals with **zero provider
calls**, preserving every answer, stored source status/origin, grade, aggregate
quality result and the invalid-grader outcome. Shared source answer hashes match
across arms (six historical sources and ten growth sources). The reference adapter
also matches the actual pre-#60 gate at `74d552b` on the model-free request-volume
fixture; failure regressions retain earlier cache decisions until a complete
replacement batch succeeds. Local model-free Python checks: **643 passed**, with
ten isolated PostgreSQL cases skipped locally. CI runs those database cases, Node
and gateway checks separately.

The configuration is the existing controlled controller/left-model path:
System2, remote right consultation, executive models and other historical factual
hints are disabled. PostgreSQL persistence is not measured. Scope/write/correction
metadata is caller supplied. These are two authored episodes per suite with
correlated probes and same-family model grading, not a full default multi-brain
quality test, independent human assessment or production reliability estimate.
The six-candidate/three-delivered-source capacity boundary is not stressed. No
statistically established quality increase follows from these results.

Paid commands are in the frozen protocol. For replay, replace each `--model
gpt-6-luna` with its matching `--journal .../journal.partial.json` and use a new
output directory. Raw answers and journals remain ignored and are not distributed.
