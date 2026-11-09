# Reasoning frontier — prospective protocol, 2026-11-09 (Asia/Tokyo)

The #108 baseline was another dual-brain runtime with its local detector ablated.
This study asks a different question: where a strong single-model baseline stops
solving every task, does dual-brain orchestration help, tie or harm? Freeze this
protocol, generator, materialized fixture, exact graders, runner and tests in a
clean commit before any paid output. No task from #108 is reused as evaluation.
No production code/default is changed. All tasks are synthetic, authored by the
implementer and governed by explicit closed-world rules.

## Prospective difficulty selection and untouched confirmation

Materialize 52 tasks before calls: six calibration levels and seven holdout
levels, each with logic, SQL, async event state and source-authority reasoning.
Seeds depend only on partition/level/domain. Input is <=10,000 characters and
requested outputs stay small; difficulty grows through composed conditions,
interleavings, bag/NULL interactions, Boolean constraints and stale authority.

Run ONLY the single-model self-review baseline on four calibration tasks at each
level, in ascending order. Choose the first level with >=3 eligible answers and
>=2 exact semantic errors among those eligible. Eligibility requires a confirmed
complete answer, valid requested JSON schema and a fully valid stage workflow.
Length/empty completion, malformed format, unavailable review, transport error
or output-cap denial must not select the frontier as if it were a reasoning error.
Retain all calibration rows and individual item scores. Stop at level 6; if no
eligible frontier is found, report that limit and do not claim a comparison win.

After selection, evaluate all eight untouched holdout tasks at that level and
the next: all four domains at both levels, all three independent conditions.
Keep their fixed six-permutation cyclic order; with eight tasks this is a declared
near-balance (not exact equal order counts). Do not replace hard/failed holdouts,
select only a favorable domain, or continue paid search based on dual outcomes.
Calibration outputs are development data and never enter holdout prompts.

## Conditions and offered resource allowance

- `self_review`: same `gpt-6-luna` makes an initial answer, critiques it, revises,
  then independently audits/reissues a final answer. Four fixed calls have output
  ceilings 4096/2048/4096/2048. Reviews are provisional; they receive only original
  task data and preceding candidates/review, never expected answers.
- `dual_auto`: the unchanged production controller/RightBrainPolicy with explicit
  left lead and System2 AUTO. Other balanced AUTO gates and goals AUTO stay active;
  Director/executive are OFF. No draft, response, action or critic is substituted.
- `dual_on`: identical controller with manual System2 ON, separating an AUTO
  admission/effort issue from behavior when collaboration is explicitly requested.

The initial answer of `self_review` is also scored as an auxiliary one-shot
reference; it is the SAME paid generation, not a fourth independent arm or an
extra paid call. Conditions sample their own initial answers; this is a system-
configuration comparison, not an isolated causal proof about role prompts.
Observe the native initial draft without changing its return value. Grade that
snapshot only after the turn, separately from the final output, to count observed
within-turn repairs/regressions when both answers have confirmed completion. No
oracle value or score is supplied to the controller, reviewer or answering model.

Every condition gets an all-provider offered output-ceiling allowance of 12,288
per task. Generation cap 4096, critic cap 2048, ordinary-left allocation 8192,
existing ordinary detail caps 450/900, provider-default reasoning effort and no
continuation. Reserve every actual requested ceiling before transport, including
failed, invalid or cancelled calls, with no refunds; deny a request that would
exceed the allowance without shrinking it or spending again. Output allowance is
matched, not actual input/output tokens, number of requests, deadlines or compute.
Report actual work separately. Native consultation/owner deadlines remain native;
self-model HTTP calls have the same client ceiling 60s. Do not silently widen the
native critic deadline to make the dual condition look available.

The complete campaign has a hard maximum of 192 actual provider HTTP attempts:
96 reserved for calibration and96 for confirmation. Each phase has its own
ScheduledJournal and a shared global admission counter/latch. No paid health or
judge request. A real transport/auth/quota error closes all subsequent campaign
paid admission, including a later phase. A native cancellation is retained with
unknown usage when no receipt arrives; it is not a zero-cost success. No paid
resampling, rescue, cap increase or post-output profile/criterion change.

Use fresh in-memory native sessions and matched per-case IDs/RNG/clock for both
dual conditions. No primary seed, external persistence or model-generated code
execution. Only the evaluator executes its OWN fixed SQL with bound data in an
isolated in-memory SQLite connection. Direct self-review prompts and native role
prompts differ by design; expected answers/case oracle data stay evaluator-only.

## Objective grading and interpretation

Logic is checked by enumerating satisfying Boolean assignments and classifying
entailed true/false/undetermined variables. SQL result bags are evaluated by SQLite
and independently verified by a Python relational calculation. Async job records
and source-authority queries use explicit deterministic state/rule evaluators,
with hand cases, boundary tests and metamorphic checks before paid output.

Require whole valid JSON with duplicate keys rejected. Integer fields require
JSON integers, not booleans/strings. A semantic item is a variable classification,
SQL output row (bag multiplicity matters), complete job state or complete source
query result. Source-ID list order is immaterial; duplicates are invalid. SQL row
order is immaterial. Per-task accuracy is the proportion of correct items (SQL
denominator=max(expected rows,returned rows,1)); each task has equal aggregate
weight. Require complete requested item IDs for non-SQL schemas. Missing/extra root
fields, invalid types and malformed schemas are format failures, not a hidden
semantic success. References and graders never appear in answering prompts.

Report confirmed exact task successes, schema/completion/workflow availability,
all-case mean item accuracy and paired task deltas. For all-case utility, an
unconfirmed/unavailable/non-scoreable answer gets 0; a confirmed complete fallback
answer is scored as actually returned even if its review failed. Preserve review
failure separately. Report both-workflows-valid pairs as a secondary slice with
the full population retained. Calibration eligibility is deliberately stricter
than returned-answer scoring so transport/format failures cannot set difficulty.

For each dual condition, a positive/zero/negative paired item delta is respectively
an observed higher/same/lower result. Show wins/ties/losses, exact counts and mean
deltas by level and domain; disagreement between metrics is mixed evidence, not a
forced winner. Do not claim statistical superiority/equivalence from eight tasks.
Show one-shot, self-review, AUTO and ON curves; a decline with level and the gap
to baseline are distinct observations. All schema/availability/cap failures and
unknown usage stay visible. No model judge or post-output semantic rescue grade.

## Replay and publication

Replay both complete phase journals under the offline provider guard. The same
calibration answers must select the same difficulty and holdout allocation.
Require exact request/transport-event coverage and matching non-timing rows,
grades, budget denials, selected level, comparisons and usage. Native cancellation
must come from its owner, not a tape injection; replay waits remain 30s. Retain
divergence without tuning or relabeling. Freeze another fresh study before paid
confirmation of any resulting implementation change.

Keep raw answers/reviews/requests and original/partial journals private in ignored
`samples/task-goals-frontier-*`; publish filtered status/score/cap/usage receipts
and original artifact, fixture, code, protocol and canonical hashes. Known token
subtotals are not full totals when a usage receipt is unknown. Actual time/cost
differences are descriptive and not proof of an isolated architectural cause.
