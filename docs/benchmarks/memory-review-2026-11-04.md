# Memory → ACC → System2: first measurement

2026-11-04 (Asia/Tokyo), following #84. The bounded new memory cue reaches ACC
and improves the required clarification on the competing-record turn. The
experiment does **not** establish general score improvement or lower cost.
It also exposes unstable support decisions for a literal followup, two empty
critic completions and a separate final-answer grading disagreement.

Frozen source: `a600e702b847f442a8c739b9474eb97fc73fc0fb`.
[Protocol](memory-review-protocol.md),
[safe metrics](memory-review-2026-11-04.metrics.json).
All fixtures, expectations, schemas, label/arm order and the 128-attempt cap were
committed before paid calls. No retry, replacement, paid replay, rescue model or
post-result rubric/expectation change was used. Older studies remain unchanged.

## #84 obligation boundary

Four independent cases, twelve authored variants, two reversed presentations and
two rubrics use all sixteen planned calibration attempts. Both receive identical
atomic obligations and fields. V3 explicitly distinguishes task-related false
observer-access claims from assuming a different actor role, and audits each
declared obligation without inventing extra operational detail.

| Fixed-expectation agreement | V2 | V3 |
| --- | ---: | ---: |
| Relevance | 23/24 | 24/24 |
| Grounding | 24/24 | 24/24 |
| Potential use | 24/24 | 24/24 |
| Critical coverage | 24/24 | 24/24 |
| All criteria | 95/96 | 96/96 |
| Variants matching all criteria in both presentations | 11/12 | 12/12 |
| Criterion/variant order disagreements | 1 | 0 |

There is one paired correction (relevance of the false observer-access claim in
reversed order) and no paired regression. All 60 V3 atomic obligation checks agree
with the fixed expectations. The twelve predeclared focus presentations match:
coverage of qualified unknown/omitted check/missing units, and relevance of false
access/unrelated topic/actor change. All packets are valid, with no missing or
uncertain judgments. This is a successful bounded boundary check, not a claim of
general judge reliability or order robustness from twelve variants.

V2 uses 18,534 input / 5,285 output tokens; V3 uses 21,794 / 5,632.
The explicit audit increases input demand **17.6%** and output demand **6.6%** in
this sample. V3 is an explicit future API; original rubrics/callers/scores remain
unchanged. This private-advice calibration does not validate the separate
task/grounding/clarification judge used below.

## Retained-session handoff

One authored six-turn trajectory uses real EngineSession processing and production
LLM clients. A third retained session generates each shared natural initial draft;
baseline and treatment consume that same draft. Their approved-evidence hashes
match on **all six paired turns**. Treatment adds only the new context/ACC cue;
source lifecycle, query support and existing AUTO gates remain authoritative.
The initial-draft session does not receive the new context cue, so this comparison
measures the post-draft handoff rather than natural drafting with the cue.

| Turn | Observed result |
| --- | --- |
| Literal replay | All three answers equal the complete stored answer byte for byte. Both paired AUTO routes skip consultation; zero right calls. |
| Competing records | Both paired sessions receive both current scoped records. Baseline skips review. Treatment identifies one competition group, supplies ACC `memory` provenance and enables System2. Task/grounding pass in both; required clarification fails baseline and passes treatment. |
| Correction + pronoun | Both old records are superseded. The initial-draft support check approves the new source; the two paired checks reject the **identical HTTP payload**. Both paired outputs ask for the already stored text and fail task/clarification. No old source is revived. |
| Other project | Both paired sessions retrieve only Larch and pass all three criteria. One right call per arm. |
| Cross-project comparison | Both retrieve the correct two current sources. Each critic returns empty content with `finish_reason=stop`, no refusal, and a known three-token completion receipt. The strict study rule marks both cells unavailable despite subsequent handler fallback work. They are not graded or replaced. |
| Expired source | No expired retention source reaches any session. Paired task/grounding pass; clarification fails in all three outputs. Original judgments are retained. |

The text-difference cue is deliberately narrow. Exact source question and declared
scope must match; unique current approved IDs and complete displayed records are
required. It does not decide semantic incompatibility or which answer is true.
Equivalent wording/partial conditions can trigger review, while differently worded
or multilingual contradictions may be missed. Both promotion defaults stay OFF;
the measured profile opts into AUTO. No temperature, basal promotion, source
verification, correction transaction or deadline behavior is expanded by this cue.

| Final-answer model grades | Natural draft | Baseline | Memory bridge |
| --- | ---: | ---: | ---: |
| Planned / available / graded | 6 / 6 / 6 | 6 / 5 / 5 | 6 / 5 / 5 |
| Task pass | 5/6 | 3/5 | 3/5 |
| Grounding pass | 6/6 | 5/5 | 5/5 |
| Clarification pass | 4/6 | 2/5 | 3/5 |
| All criteria pass | 3/6 | 1/5 | 2/5 |
| Right HTTP attempts | 0 | 4 | 6 |

These denominators exclude unavailable answers, whose counts remain explicit.
The single conflict-turn clarification improvement is the supported positive
result. Neither the initial-draft arm nor the trajectory's six turns are
independent repeated trials. In particular, the correction's initial draft and
paired sessions had different support approvals, so their quality difference
cannot be attributed solely to later consultation.

There is also a **final-answer judge disagreement**: it fails `task` on all three
literal-replay answers although direct byte equality confirms the required stored
answer. The reference includes an attribution caveat alongside an exact-replay
requirement, which may confuse the judge; that causal explanation is unproven.
The scores stay unchanged. Correct-looking quotation evidence/schema validity is
not enough to accept a semantic verdict. Clarification failures on expired-data
answers likewise need a separately calibrated boundary before strong conclusions.

## Cost and latency

All **65/128** allocated HTTP attempts are retained: 16 calibration, 18 support,
6 initial drafting, 10 right-critic, 9 left integration/fallback and 6 final-judge
attempts. Production/support use `gpt-6-luna`; judges use `gpt-6-sol`.
There are two returned invalid completions, no transport outage/cancellation and
no unknown usage receipts. Total: **67,960 input / 17,539 output tokens**.
Unused allocation is not spent on replacing failures.

| Production slice | Calls | Input / output tokens | Handler total / median |
| --- | ---: | ---: | ---: |
| Shared initial drafts + their support | 12 | 4,014 / 1,168 | 24.05 s / 3.75 s |
| Baseline after shared drafts | 14 | 8,079 / 1,463 | 32.13 s / 4.65 s |
| Memory bridge after shared drafts | 17 | 11,640 / 1,666 | 36.90 s / 5.61 s |

The conflict turn costs 4 versus 1 calls and 10.41 versus 1.56 seconds in the
paired handlers. Optional-lookup right attempts remain two per arm. This is an
observed quality/cost tradeoff, **not** an aggregate cost or latency win. Shared
draft cost and judging/calibration remain in the complete ledger, rather than
being hidden from the experiment total. Timings are single observations, including
failed cells, and do not estimate a population latency effect.

## Replay and verification

At the frozen source, exact replay with transport removed consumes all 65 recorded
occurrences and makes **zero new provider calls**. Calibration scores with private
full-field context, summaries, answers/status, routes, source IDs/evidence hashes,
final grades and usage match exactly. Real handler timing is excluded; replay
does not establish latency reliability or independent judging accuracy.

Local frozen-source checks: **1,326 Python tests**, Gateway **17 passed / 2 expected
skips**, Node **13 passed**. New tests reproduce the missing memory-to-ACC signal
and cover correction/expiry, denied/derived/duplicate sources, explicit OFF,
missing context/capability, configured thresholds and temperature/basal bypasses.
Existing PostgreSQL tests run against the private local PostgreSQL 16 instance.
The live pilot seeds authored in-process records; it does not measure durable
primary writes, PostgreSQL transactions or process-restart retrieval.

Final review also reproduces a mixed-signal boundary: a memory-boosted ACC level
could raise otherwise ineligible micro evidence above the temperature threshold.
The final implementation retains the independent micro-only conflict level for
temperature admission and magnitude; two additional deterministic cases cover
weak evidence remaining ineligible and eligible micro evidence keeping its
original strength. No paid requests or old scores are replaced. The measured
competition turn has no micro-critic result, so the mixed-signal correction is
validated by regression tests rather than claimed as a separately measured gain.
The frozen commit above remains the reference for the exact live replay.

Private artifacts remain under ignored
`samples/module-gating-memory-review-2026-11-04-{live,replay}/`.

| Artifact | SHA-256 |
| --- | --- |
| Live report | `622386620d87f54c41888815e4ae86de54fc413da0bbe56dd89f132691dd91bf` |
| HTTP journal | `aa690453ffa00b3ba48706b7ff767b9a3df5131beea84c4788150c66a8938ef9` |
| Safe original summary | `ab18d39c86daaa776b59c22d15604a4459e59000cc7bce774a8ed636ce67c121` |

Next priority is explicit literal-replay intent through reference resolution and
support checking, paired with a calibrated exact-source equality criterion in
future final-answer grading. Retain mandatory support/lifecycle/source guards and
freeze independent cases before new measurements. Empty critic response/fallback
availability needs a separate bounded reliability followup, without certifying
missing review or silently adding paid retries.
