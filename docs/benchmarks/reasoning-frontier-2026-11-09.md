# Reasoning frontier — 2026-11-09 (Asia/Tokyo)

**Cap erratum (2026-11-09, #111):** native draft/integration requests actually
serialized **2048**, despite configured/requested generation caps and ordinary
reservations of4096. An inherited benchmark adapter silently clamped them.
The original holdout journal contains16 drafts and7 integrations with this
2048 cap; self-review's8 initial generations actually requested4096. The scores,
raw tapes and frozen metrics below remain unchanged, but this comparison did
not offer matched actual generation request caps. Native logical failures
occurred at2048, so the earlier claim of native failure at4096 is withdrawn.
The observed6/8 versus5/8 counts remain descriptive results under these unequal
caps and do not establish a fair per-request resource comparison. #111 fixes
the adapter, asserts actual serialized caps and uses never-sent tasks in a
[separate experiment](replanning-corrected-protocol.md).

**Manual dual ON returned 6/8 exact answers, self-review 5/8, and dual AUTO 5/8.**
ON gained one event-state task. AUTO gained that task but lost a SQL task; its
mean item score was slightly lower. Both logical-entailment tasks exhausted
generation caps in every condition. The initial one-shot reference scored 4/8.
These are observations on eight authored tasks, not statistical superiority or
an isolated causal demonstration of collaboration.

The original prospective calibration **did not meet its selection rule**. The
comparison below is a separately declared follow-up, amended after baseline
calibration and frozen before any holdout or dual output. Both studies and all
failures remain unchanged. All **152 request occurrences / 304 transport events**
replayed exactly with zero provider calls and zero blocked HTTP attempts.

See the [original protocol](reasoning-frontier-protocol.md),
[follow-up protocol](reasoning-frontier-followup-protocol.md),
[locked calibration manifest](reasoning-frontier-followup-manifest.json),
[52-task fixture](../../sr-dual-brain-llm/examples/reasoning_frontier.json) and
[filtered metrics](reasoning-frontier-2026-11-09.metrics.json). Raw answers,
reviews, quotes and original/partial journals remain in private ignored samples.

## What was frozen and compared

The model requested and reported by the provider was `gpt-6-luna`, using provider-
default reasoning effort and no continuation. The self-review condition makes
an initial answer, a structured self-critique, a revision and a final audit/
answer, with output caps 4096/2048/4096/2048. Its initial generation supplies the
auxiliary one-shot reference; that is the same paid sample, not another arm.

Dual AUTO and ON use the unchanged native RightBrainPolicy, explicit left lead,
balanced AUTO gates, goals AUTO, and Director/executive OFF. Sessions are fresh,
with matched per-case IDs/RNG/clock between native conditions. No draft, action,
critic, answer or oracle is substituted. The initial native draft is observed
without changing its return value and graded only after the turn.

All three conditions have the same **12,288 offered output-ceiling allowance**
per task, including every provider request and failed/cancelled work, with no
refunds. Configured generation cap/ordinary reservations were4096, but actual
native generation requests were2048 (see erratum); critic cap2048 and
ordinary-left allocation8192;
native bounded schema repair/fallback remains active and counted. Input tokens,
actual work, prompts, request counts and deadlines are not matched. Self-review
has a 60s client limit; native critic/consultation owner deadlines stay native.
Each condition independently samples its candidates. No production core/default
or original benchmark task was changed to improve these results.

Exact graders enumerate Boolean models, evaluate a fixed authored SQL query in
in-memory SQLite, replay explicit event-state rules, and evaluate source validity/
authority rules. Tests independently check SQL with a Python relational oracle,
small Boolean systems by brute force, and event/evidence boundaries and invariants.
Only evaluator-authored SQL is executed; model-generated code/SQL is not executed.
There is no paid model judge and no expected answer in model prompts.

## Calibration and the declared amendment

The original runner/fixture/protocol/graders were frozen at
`5afd59bb16c408b84f288940f7e7edefa8be5a19` before 80 paid calibration attempts.
Four domains were evaluated at every level, in ascending order. The original
rule required at least three eligible workflows and two exact semantic errors
within a level. Eligibility requires a complete answer, valid answer schema and
all stages valid. No level qualified, so its recorded outcome remains
`no_eligible_frontier_within_frozen_levels`, with **zero holdout rows**.

| Calibration level | Final exact / 4 | Complete / 4 | Valid workflow / 4 | Mean item score |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3 | 4 | 3 | 95.83% |
| 2 | 3 | 4 | 3 | 81.25% |
| 3 | 3 | 3 | 3 | 75.00% |
| 4 | 3 | 3 | 3 | 75.00% |
| 5 | 2 | 3 | 3 | 70.83% |
| 6 | 3 | 3 | 2 | 75.00% |

Logic at levels 1/2 returned real semantic errors but its self-critique exhausted
2048 tokens. At levels 3–6 its initial generation exhausted 4096 tokens. Those
workflow failures cannot select a semantic frontier. The eligible semantic error
was level-5 SQL: a complete final audit omitted one eligible account (5/6 rows).
Its initial answer and revision were exact; the account's annotation had reason
`note`, while the query excludes only active `hold` bans. Calibration also includes
one evidence answer repaired at level 3. Initial and final exact counts were both
17/24, with one repair and one regression; final mean item score was 78.82%.

After replaying the original unchanged source, the separate follow-up selected
the **first level with >=3 original eligible workflows and >=1 semantic error**:
level 5. The original report's bytes/canonical hash, admission count, fixture,
protocol, config and original core/scripts manifest are locked and checked. Only the
new follow-up runner is excluded from the original source-manifest digest.

The amendment, manifest, new runner and its ten offline tests were committed at
`b52f3658859cf8d8ab2fe812aeeb181d66670497` before any confirmation output. No
calibration was rerun and no dual result informed selection. All four domains
at untouched holdout levels 5 and 6 were used, with the original fixed cyclic
arm order and unchanged scoring/profile. This is post-calibration exploratory
selection, not retrospective success under the original rule.

## All eight heldout tasks

Scores below are correct-item fractions; `U` is an unconfirmed/unavailable answer
and contributes zero to all-case utility. Whole valid JSON, exact field types and
all requested non-SQL item IDs are required. SQL scores compare row bags, including
multiplicity. Correct complete fallbacks retain their actual score even when an
earlier generation/review failed; workflow validity is reported separately.

| Level / domain | One-shot reference | Self-review | Dual AUTO | Dual ON |
| --- | ---: | ---: | ---: | ---: |
| 5 / Boolean entailment | U | U | U | U |
| 5 / SQL | 1 | 1 | 6/7 | 1 |
| 5 / Event state | 1 | 1 | 1 | 1 |
| 5 / Evidence authority | 1 | 1 | 1 | 1 |
| 6 / Boolean entailment | U | U | U | U |
| 6 / SQL | 6/7 | 1 | 1 | 1 |
| 6 / Event state | 7/8 | 7/8 | 1 | 1 |
| 6 / Evidence authority | 1 | 1 | 1 | 1 |

| Condition | Exact / 8 | Complete / 8 | Schema valid / 8 | Valid workflow / 8 | Mean item score |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-shot reference | 4 | 6 | 6 | — | 71.65% |
| Self-review | 5 | 6 | 6 | 4 | 73.44% |
| Dual AUTO | 5 | 6 | 6 | 3 | 73.21% |
| Dual ON | 6 | 6 | 6 | 3 | 75.00% |

Each task has equal weight, regardless of its item count. Compared with self-
review, ON is **+1.56 percentage points**, with **1 higher / 7 same / 0 lower**
task scores. AUTO is **−0.22 points**, with **1 higher / 6 same / 1 lower**. Its
SQL loss is 1/7 of a task; its event gain is 1/8, explaining the slight mean
decline despite equal exact counts. All three pairs with both workflows valid
(level-5 evidence, level-6 SQL/evidence) tie in both native conditions. The useful
differing pairs remain in the primary population and fail that stricter subset.

The observed level curves, with four independently authored tasks per level:

| Holdout level | One-shot mean | Self-review mean | AUTO mean | ON mean |
| --- | ---: | ---: | ---: | ---: |
| 5 | 75.00% | 75.00% | 71.43% | 75.00% |
| 6 | 68.30% | 71.88% | 75.00% | 75.00% |

Two sparse levels with different seeds do not establish a monotonic difficulty
scale or scaling law. Evidence remains at a ceiling; broad semantic headroom
across all four domains has not been demonstrated.

## What helped, failed or remained unproven

The level-6 event error concerns ownership: after transfer from `birch` to
`coral`, a finish from `birch` has no authority. Self-review returned committed;
the rule requires running with owner `coral`. Both native final answers were
correct. AUTO's initial draft was already correct and its critic timed out. ON
recovered from an incomplete initial draft through integration. Neither is an
observed correction of a confirmed wrong native answer by a successful critique.

AUTO's level-5 SQL answer selected event 50 instead of event 47 as the latest
eligible non-void row for one account; every other field/row matched. That wrong
complete draft survived a critic timeout. ON and self-review were exact. These
are independent samples, so the difference cannot be attributed to the AUTO gate
alone. The self-review condition itself repaired its level-6 SQL answer (6/7 to
7/7); no confirmed-complete semantic regression occurred on its heldouts.

Native AUTO recovered one incomplete initial answer to an exact final answer
(level-5 events); ON recovered two (events at both levels). Among pairs with both
native snapshots complete, all final item scores were unchanged: no semantic
repair or regression was observed. Logical drafts and integrations exhausted
their actual2048-token request ceilings in both native modes. Their critics reported
issues, but the subsequent generation still failed to finish. Additional native
work did not solve those two tasks within this envelope.

Both event self-audits exhausted 2048 tokens and retained complete revisions.
There were six native owner cancellations: level-5 SQL in each mode (ON during
schema repair), level-5 event verification in each mode, and level-6 event review
in each mode. Three event cancellations were followed by existing bounded direct
fallback reviews; the failed attempts remain visible. A correct final answer,
a valid review and a workflow with no failed stages are different observations.

## Actual work and replay

| Scope | HTTP attempts | Known input subtotal | Known output subtotal | Unknown usage | Summed handler seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | 80 | 108,486 | 77,842 | 0 | — |
| Holdout self-review | 26 | 48,722 | 34,567 | 0 | 303.2 |
| Holdout AUTO | 21 | 45,321 | 22,793 | 3 | 252.1 |
| Holdout ON | 25 | 57,760 | 27,602 | 3 | 290.9 |
| Whole campaign | **152** | **260,289** | **162,804** | **6** | — |

The original global limit remained 192 attempts and confirmation limit 96;
confirmation spent 72. There were no denied admissions or real transport errors.
Every trial reserved <=12,288 output tokens, including failed/cancelled requests.
Six missing receipts mean full token totals/billing cost are unknown; native
subtotals are not a measured cost saving. Durations are descriptive independent
samples with different native/self-review deadlines, not a causal speed estimate.

Original replay covered 80/80 occurrences and 160/160 transport events. Follow-up
replay covered 72/72 and 144/144. Canonical reports, exact grades, difficulty/
allocation, paired outcomes, reservations, denials and usage matched. Both provider
guards reported zero blocked attempts and both replay runs made zero provider
calls. Native cancellation came from its actual owner, not tape injection. An
earlier replay bootstrap stopped with `ValueError` before creating a journal;
its failure artifact was retained separately and is not counted as verification.

| Identity | SHA-256 |
| --- | --- |
| Fixture, canonical JSON | `0739e09c07cdc5ba5c0558466da31e15f92cb882fe430127d954941a3eb676d7` |
| Original core/scripts manifest | `a83d40fa9fe31ff1d5ae28efd45b5b35acf48b82e820d135ea77d399523f8154` |
| Follow-up core/scripts manifest | `f7fe1c738cbfefa5db1b190eabaf8f9cfd877b2dc979870a851d65fe090c53c8` |
| Original canonical report | `42c496ef209eee9f3084a3805211880414553ead4f64b3f417c7a83c10770e10` |
| Follow-up canonical report | `a602029c455a6fc69bdec7c0e30435f6b23eb1c020f9423d7aeb28a2cfc5f482` |

The filtered metrics include protocol hashes, original byte hashes, transport
hashes and every private JSON artifact's byte hash. They omit answers, reviews,
questions, goal quotes and request payloads; none of the raw tapes is published.

## Next bounded work

First make orchestration respond to completion state and remaining allowance:
distinguish missing candidates, complete provisional answers and failed reviews;
decompose unresolved work before repeating a generation that already exhausted
its ceiling. Preserve completion/evidence guards, native cancellation ownership
and unresolved status. The current gain supports investigating completion
recovery, not promoting a critic to semantic authority.

In parallel scope, diagnose critic availability using deterministic completion,
schema-repair and owner-cancellation cases (related [#87](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/issues/87)).
Keep a confirmed candidate through review failure without counting that failure
as verification. Do not silently retry, widen deadlines or add case-specific
answer branches to improve this study's scores.

For the next quality measurement, freeze new cross-domain tasks that increase
interacting conditions at bounded input size, with more repetitions around the
semantic-error boundary. These outputs are now development data. Test genuine
repair, completion recovery, regressions and actual work separately; retain
one-shot/self-review comparators and the same provenance discipline.
