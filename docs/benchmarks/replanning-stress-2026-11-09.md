# Hard replanning stress — 2026-11-09 (UTC)

**On eight fresh tasks, self-review and native replanning AUTO each returned6/8
exact answers; native replanning OFF returned5/8. AUTO did not outperform
self-review.** Mean item scores were75.00%,75.00% and73.44%, respectively. Both
logic tasks were unavailable in every condition. Every complete initial answer
kept its original score: no semantic repair, regression or completion recovery
was observed. The one-shot reference already scored6/8.

This work found and corrected two concrete defects: native benchmark requests
were silently clamped to2048 despite configured4096, and HTTP duplicated the
consultation owner's deadline timer. The first study remains stopped, its exact
replay remains failed, and its outputs were never rerun or regraded. The corrected
study is a separately frozen post-failure experiment on never-sent questions.
It supports descriptive bottleneck findings, not statistical superiority or an
isolated causal effect of replanning.

See the [original protocol](replanning-stress-protocol.md),
[corrected protocol](replanning-corrected-protocol.md),
[locked lineage/selection manifest](replanning-corrected-manifest.json),
[original12-case fixture](../../sr-dual-brain-llm/examples/replanning_stress.json),
[derived8-case fixture](../../sr-dual-brain-llm/examples/replanning_corrected.json)
and [filtered metrics](replanning-stress-2026-11-09.metrics.json). Raw provider
messages, reviews, goal quotes, payloads and original/partial journals remain
private in ignored samples. No paid judge, health call or resampling was used.

## The failed original study and cap erratum

The original experiment was frozen at
`f88437b69dae50967737fd59c7603068e9db815a`. It stopped after12 admissions and
five trial rows with `transport_blocked`. Only `main:0:logic` and `main:0:sql`
were exposed. Logic was unavailable under all three arms. SQL had two complete
native answers: OFF exact and AUTO5/6, with no self-review trial. These unequal
populations cannot rank a complete eight-task comparison; the SQL difference
was between independent initial samples, with no actual replanning exposure.

One SQL critic was cancelled by its owner at approximately14.25s. Another critic
returned unstructured text around12.49s; its repair raised generic `TimeoutError`
approximately1.75s later. That unclassified failure correctly closed the frozen
study's sticky transport latch. There was no recorded HTTP auth/quota/provider
status error; this absence does not prove provider health or retroactively
reclassify the timeout. Known usage was16,756 input /16,063 output tokens with
two unknown receipts. Full totals and billing remain unknown.

The single strict duration-aware replay at that original source failed with
`ValueError`: the final generic timeout became actual-owner cancellation at
the competing boundary, so event coverage diverged. The original and partial
failure artifacts remain unchanged. No retry, timing adjustment, tape fitting
or successful-replay claim is made for that experiment. Its failed-run guard
counter was not persisted and is not claimed as a verification receipt.

Inspecting actual serialized requests exposed an inherited `RecordedClient`
clamp: native drafts/integrations requested2048 although model configuration
and ordinary allocation reserved4096. The original stress tape contains four
such drafts and two integrations. The earlier #109 heldout tape contains16
drafts and7 integrations at2048, while all8 self-review initial requests were4096.
The [earlier report now carries an erratum](reasoning-frontier-2026-11-09.md).
Its observed scores, raw tapes and frozen metrics remain unchanged; matching
actual generation caps and native failure at4096 were incorrect interpretations.

## Bounded corrections and the new freeze

The modern benchmark `StageClient` now forwards the explicit positive request
cap or configured fallback; the legacy adapter retains its historical2048
default. A regression checks serialized native draft and integration payloads
at4096. The corrected study checks every role/cap before provider admission and
closes on configuration mismatch without rewriting the request.

Critic admission, retries, repair and late-result checks still consult the
monotonic local review window and actual RPC/callosum owners. HTTP uses the
remaining local/client window; the real owner cancels at its unchanged deadline,
without a second HTTP timer competing at the same boundary. Local review-window
expiry has a typed `ReviewBudgetExhausted` receipt with HTTP-attempt provenance.
It leaves review unavailable and never clears an existing transport latch.
Generic timeout, network, HTTP, auth and quota failures remain sticky stop causes.
Reservations/admissions are never refunded; proven pre-HTTP expiry may have
known-zero HTTP usage, while attempted work without receipts remains unknown.
No real deadline, output ceiling, retry count or production module default was
automatically enlarged. Completion replanning remains default OFF.

Corrections, tests, new protocol, manifest and selected fixture were frozen at
`4a1bdb5ebbe9dad45347d02bd7cd64ed19e84dfe` before any corrected output. Source,
HEAD, questions, profile, graders and criteria stayed unchanged through live
work and its single replay. The actual cap/timer correction was not separately
randomized, and the old exposed tasks were not rerun: no pre/post improvement
claim is supported.

Selection was the first two never-sent cases per domain in the original fixture
order, using exposure membership only. Case objects and original logical-clock
indices are unchanged. Logic/SQL now use two independent higher-tier cases;
events/evidence retain both tiers. The old `confirmation` IDs record provenance,
not successful confirmation under the original protocol. This is not a common
two-tier monotonic difficulty curve.

## Conditions and all eight results

Requested/reported model: `gpt-6-luna`; provider-default effort, no ordinary
continuation. Self-review uses initial/critique/revision/audit caps4096/2048/
4096/2048. Its initial answer supplies the auxiliary one-shot reference. Both
native arms use the production RightBrainPolicy, explicit left lead, manual
System2 ON, goals AUTO and balanced AUTO gates, Director/executive OFF. Native
completion replanning is OFF versus AUTO. Sessions use matching per-case IDs,
RNG and logical clocks; initial candidates are independently sampled. No model
answer, review or action is substituted with an oracle or script.

Every trial offers the same all-provider output-ceiling allowance12,288; native
ordinary-left allocation8192. Actual serialized native generation caps are4096
and critic caps2048. Prompts, input tokens, actual compute, request counts and
native/self-review deadlines differ. Exact graders are unchanged: exhaustive
Boolean models, a fixed evaluator-authored SQLite query plus independent Python
oracle, event-state replay and scoped evidence validity/authority. Model code
is never executed. Whole JSON, strict field types, duplicate-key rejection and
SQL row-bag semantics are retained. Complete fallback answers keep their true
score despite review failure; unavailable answers contribute zero utility.

`U` means no confirmed complete answer. Fractions are exact correct-item scores.

| Case | One-shot reference | Self-review | Native replan OFF | Native replan AUTO |
| --- | ---: | ---: | ---: | ---: |
| main0 / events | 1 | 1 | 1 | 1 |
| main0 / evidence | 1 | 1 | 1 | 1 |
| main1 / logic | U | U | U | U |
| main1 / SQL | 1 | 1 | 7/8 | 1 |
| main1 / events | 1 | 1 | 1 | 1 |
| main1 / evidence | 1 | 1 | 1 | 1 |
| confirmation1 / logic | U | U | U | U |
| confirmation1 / SQL | 1 | 1 | 1 | 1 |

| Condition | Exact /8 | Complete/schema valid /8 | Valid workflow /8 | Mean item score |
| --- | ---: | ---: | ---: | ---: |
| One-shot reference | 6 | 6 | — | 75.00% |
| Self-review | 6 | 6 | 5 | 75.00% |
| Native replan OFF | 5 | 6 | 2 | 73.44% |
| Native replan AUTO | 6 | 6 | 3 | 75.00% |

Each case has equal weight. AUTO versus self-review is0 higher /8 same /0 lower;
OFF versus self-review is0 higher /7 same /1 lower; AUTO versus OFF is1 higher /
7 same /0 lower (+1.5625 percentage points). The differing SQL case had no
replanning plan in either arm. OFF's complete initial answer overcounted one
`joined_events` bag-multiplicity field, and its critic timed out. AUTO's initial
answer was already exact. This is no observed semantic correction or isolated
planning effect. All three cases with valid workflows in both self-review/AUTO
(both evidence cases and main1 SQL) tie; this secondary subset does not erase
the failures in the primary population.

## Completion and critic bottlenecks

All conditions had six complete initial/final pairs with unchanged scores and
two unavailable initial/final pairs. There were zero semantic repairs, semantic
regressions or completion recoveries. Self-review's higher-tier event critique
exhausted2048 and retained its correct initial answer; that remains a failed
workflow. The one-shot reference was already6/8, so added review did not improve
the measured score in this sample.

Both native modes exhausted4096 on each logical draft and again on each
integration. AUTO produced two actual replan records and its two corresponding
integration prompts carried the recovery plan. Neither finished. The prototype
changes instructions inside one existing integration, but it does not preserve
a completed intermediate derivation or subdivide the remaining allocation into
new work products. Asking for a dependency-ordered reconstruction was
insufficient within this fixed envelope; it is not evidence that every form of
decomposition fails.

Native critics returned9/16 valid external reviews, but four reviewed missing
logical candidates. Among the12 complete native initial answers, only5/12 had
a valid external review (OFF2/6; AUTO3/6). Seven owner cancellations account for
the rest: all four event trials, both confirmation SQL trials and OFF main1 SQL.
An available critic on a missing candidate is not a verified final solution.
Goal telemetry is provisional: OFF recorded3 valid /1 invalid /4 unavailable
goal reviews, AUTO5 valid /3 unavailable; none means verified correctness.

The only wrong complete native candidate had no completed critic, so this sample
does not test successful semantic correction of a confirmed wrong candidate by
a valid review. Evidence and events remained at a ceiling; the new difficulty
was largely a completion floor in logic, not a broad semantic frontier. The
results do not establish cross-domain superiority or a statistical scaling law.

## Actual work and replay

| Scope | Conservative admissions / attempted HTTP | Known input | Known output | Unknown receipts | Summed handler seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original stopped study | 12 /12 | 16,756 | 16,063 | 2 | — |
| Corrected self-review | 24 /24 | 39,498 | 30,624 | 0 | 320.7 |
| Corrected native OFF | 18 /18 | 30,132 | 27,088 | 4 | 344.5 |
| Corrected native AUTO | 18 /18 | 33,973 | 28,853 | 3 | 344.2 |
| Combined | **72 /72** | **120,359** | **102,628** | **9** | — |

Corrected admissions were60, with no generic transport failures or denied
admissions. Every trial reserved at most12,288. Seven cancelled HTTP attempts
have unknown usage. No typed local-window expiry or schema repair occurred in
this live phase; deterministic tests cover those branches. Thus attempted HTTP
and conservative admissions happen to match. Combined
unknowns include eight cancellations and the original generic timeout. Full
token totals and billing cost remain unknown; these subtotals are not cost
savings. Handler times are descriptive, not matched-compute causal speed results.

The corrected phase was capped at96 admissions, initialized with the original12:
the separately frozen combined maximum was108, below the announced128. Actual
combined spend stopped at72. No optional32-call confirmation, health probe,
judge, paid retry/rescue or further tuning was performed after these results.

The single corrected replay reproduced **60/60 requests and120/120 transport
events**, the canonical report, all24 non-handler-timing rows, exact grades,
reservations, denials, known/unknown usage, planning/critic outcomes and paired
comparisons. It made **zero provider calls and zero blocked HTTP attempts**.
Recorded durations and event order were preserved; actual owners generated
cancellation. The raw journal was identical, and source remained at the frozen
candidate throughout. This verifies the corrected experiment only: the original
failed replay remains failed and separately retained.

| Identity | SHA-256 |
| --- | --- |
| Original canonical report | `980449312aad4ed99fd53063a0ef04e15e5c9ad6def34c51840e25f440817b41` |
| Corrected canonical report | `25f63cc60cc3d7c16071d920e52c5432fec0cf245692fd7b441d1310ba358fd8` |
| Corrected transport events | `0ff37ce20a63e6592c0928d147f3a971a4511f5ac7abb0e37177211a9902494d` |
| Corrected candidate core/scripts | `5359d9350cff494d6c55c2e58e09642c575a15c514780a95daeb1c28f40b04e2` |
| Derived fixture, canonical JSON | `c533f2b8831c26f33ddf53605ade955c691f3b4d28b2a9371fe88b8c84b0b4a7` |

The manifest locks the original report/journal byte and JSON identities, fixture,
grader, exclusions, selected case IDs and candidate digest. Filtered metrics add
protocol/runtime metadata, actual cap audits, replay receipts and byte hashes
of every private JSON artifact, including the failed original replay/partials.

Before corrected paid work,337 focused regression tests passed with zero blocked
provider HTTP. The frozen candidate passed all8 CI checks, including minimum
Python3.10 ownership, PostgreSQL persistence and gateway contracts.

## Next bounded work

Prioritize explicit intermediate work products and handoff according to candidate
completion state and remaining allowance. Keep user-facing answer completion,
unverified internal work and verified evidence separate. The present full-task
reconstruction repeated the same completion failure without carrying a finished
intermediate result; a new design should make that progress observable within
the existing resource ledger, across SQL/event/evidence as well as logic.

Then improve critic availability by matching review scope to complete candidates
and remaining time, while retaining actual owner cancellation and correct-answer
preservation. Do not count a missing-candidate report or timeout fallback as
semantic verification. Measure useful review and actual initial-to-final repair,
not just returned JSON. Develop these mechanisms against the observed failures,
then freeze new questions and criteria before another paid comparison. This
experiment's outputs are development evidence, not future confirmation data.
