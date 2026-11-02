# Advisor session sequence — 2026-11-03 (Asia/Tokyo)

Two persistent production sessions completed six live turns each, with no
advisor-history seeding between turns. Whole-turn HTTP attempts, including
natural primary generation, were **38 ON versus 30 AUTO**. After a real
**61.06-second** idle, both histories expired and AUTO probed again. All twelve
Executive tasks completed usefully; every cell drained its advisor tasks.

The study also exposed a cancellation bug, repaired before live measurement,
and a conservative-policy limitation: a recently slow advisor can stay inhibited
even after it becomes fast. Raw blind grades were 4/6 ON and 5/6 AUTO, with a
shared rubric defect described below. These scores do not establish a quality
gain. Compatibility defaults and admission thresholds remain unchanged.

The [prospective protocol](advisor-session-protocol.md), fixture, production fix
and runner were frozen at `27d887ce6b10b097a85a8e90eb1e77415ba90d14` before paid
calls. Safe role counts, original grades, estimates and lifecycle outcomes are
in the [metrics projection](advisor-sessions-2026-11-03.metrics.json). A later
runner-only validation latch rejects missing replay facts even when the budget
producer catches a ledger error; it changes neither production policy nor the
recorded study. Use the frozen revision for the original journal replay.

## A real RPC cancellation defect

The initial network-free sequence used the real TurnDispatcher and exposed a
Python 3.11+ interaction: `asyncio.wait_for` can cancel/uncancel its current task
on a local timeout. That consumed `_TurnTask`'s cancel-once guard, so a subsequent
advisor wait or RPC cancellation could be ignored. In the failing controlled
run, metrics completed after approximately 2,701 ms despite a 2,500 ms wait,
following the Director's earlier timeout.

Advisor waiting now uses `asyncio.wait` with a timeout, then explicitly cancels
and drains the owned advisor. The parent turn's guard remains available for
actual abandonment and cleanup. Timeouts are not raised globally, and repeated
cleanup cancellation remains guarded. Regression tests verify successive local
timeouts and an RPC deadline after a local timeout. The complete local suite
passed 1,175 Python tests, including real PostgreSQL; Gateway passed 17 with two
expected skips, and UI passed 13.

## Live sequence and state

Create one empty EngineSession per profile and retain it for every turn. Compare
three advisors ON versus AUTO v2; other nine gates remain AUTO in both. Caller
modes are left lead, System2 auto/balanced, evaluative dial, Executive observe and
observer both. Initial state, questions and per-turn policy seeds match; normal
source, unconscious and learner state subsequently evolves independently.
Each arm generates its own natural primary draft. No separate warmup is excluded.

All turns use a real RPC: 120-second caps for the first five steps and 20 seconds
for the last. Conversation record context is disabled and no external records
are seeded. Ordinary source/state updates still produce five retrieval-support
requests per arm; those are counted. Primary, consultation, integration and
support work are not assumed identical just because the inputs match.

| Phase | ON calls | AUTO calls | Actual AUTO Director / metrics behavior |
| --- | --- | --- | --- |
| `cold` | 6 | 6 | Unknown estimates; both probe and time out |
| `warm` | 8 | 4 | One censored sample each; both inhibited |
| `warm_again` | 7 | 5 | Same retained slow bounds; both inhibited |
| `expired` | 5 | 5 | After 61.06 seconds, estimates unknown; both probe and time out |
| `recovered_warm` | 6 | 4 | New censored bounds; both inhibited |
| `bounded_rpc` | 6 | 6 | Both inhibited; Executive admitted under the real outer budget |
| Total | **38** | **30** | Eight advisor omissions |

Source counts progress from one through six in both live sessions. No state or
ledger is replaced at the idle boundary. The actual read facts confirm cold,
warm and expired history; no planned phase required forced reseeding. In the
last AUTO turn, approximately 18.54 seconds remained at Executive admission,
after request preparation. Executive completed; this live case exposes a bounded
RPC window, not a demonstrated live Executive inhibition. The stricter causal
deadline case is in the controlled experiment below.

## Quality and rubric limits

All twelve finals and twelve natural primary drafts were blind graded using
`gpt-6-sol`, shuffled labels, three independent criteria and literal quote
validation. No grade was malformed. Production roles used `gpt-6-luna`, 2,048
output tokens, 60-second provider timeout and no continuation. Six serial pairs
within one session per arm are correlated, not six independent replications.

| Phase | ON final, raw grade | AUTO final, raw grade |
| --- | --- | --- |
| `cold` | Task/clarification pass; grounding fail | All pass |
| `warm` | All pass | All pass |
| `warm_again` | All pass | All pass |
| `expired` | All pass | All pass |
| `recovered_warm` | Task fail; grounding/clarification pass | Task fail; grounding/clarification pass |
| `bounded_rpc` | All pass | All pass |

No ON-pass/AUTO-fail criterion occurred. The raw aggregate is ON 4/6 and AUTO
5/6, and natural primary drafts already have the same respective aggregates.
Different generation and critic responses, evolving state and a model judge
prevent assigning the difference to advisor admission.

The shared `recovered_warm` failure is an authored **rubric/question mismatch**.
The question asks, in exactly two sentences, why a timed-out request may still
have succeeded. Both answers correctly explain that client timeout does not stop
server execution and that a response can be delayed/lost. The task criterion
additionally demands a next check that the question did not request. Preserve
the original fail grades, but treat this as an evaluation-design defect rather
than evidence of a production answer defect. Do not rescore or change the frozen
fixture after seeing results; correct alignment on independent future cases.

The cold ON final's grounding failure quotes “allowing traffic through in an
orderly turn”, an implication of ordering that semaphores do not generally
guarantee. The judge also penalized its single-permit draft analogy; binary
semaphores are valid examples, so that wording is interpretation-sensitive.
Raw model grades remain descriptive and are not authoritative ground truth.

## Completion and attempt accounting

| Advisor role, across six cells per arm | ON admitted / usable / timeout | AUTO admitted / usable / timeout |
| --- | --- | --- |
| Director | 6 / 0 / 6 | 2 / 0 / 2 |
| Executive advice | 6 / 6 / 0 | 6 / 6 / 0 |
| Executive metrics | 6 / 1 / 5 | 2 / 0 / 2 |

Every admitted live task started. There were no other failed/cancelled task
outcomes or transport failures. Structural usability is not factual correctness
or proof that guidance improves the answer. Eight AUTO advisor tasks were
inhibited before creation. ON had one fast usable metrics completion on the last
turn while AUTO retained the earlier slow bound: this is a recovery warning
signal, not proof that the unattempted AUTO call would have completed equally.

| Whole-turn role | ON | AUTO |
| --- | --- | --- |
| Primary generation | 6 | 6 |
| Retrieval support from retained session sources | 5 | 5 |
| Right consultation/verification | 6 | 6 |
| Left integration/repair | 3 | 3 |
| Executive advice | 6 | 6 |
| Director | 6 | 2 |
| Executive metrics | 6 | 2 |
| Total | **38** | **30** |

Non-advisor aggregate roles match. Per-turn work differs: ON used an extra
System2 round on `warm`; AUTO used an extra round on `bounded_rpc`. Those
opposite variations cancel in aggregate. There is no aggregate replacement
increase, but the sequence does not isolate a causal effect on individual
generation, consultation, quality or end-to-end latency.

| Live phase | Attempts | Unknown cancellation receipts | Known input / output tokens |
| --- | --- | --- | --- |
| ON turns, including primary generation | 38 | 11 | 17,845 / 3,372 |
| AUTO turns, including primary generation | 30 | 4 | 16,579 / 3,177 |
| Shared blind judging | 6 | 0 | 5,311 / 3,338 |
| Whole study | **74** | **15** | **39,735 / 9,887** |

The prospective cap was 96 attempts; no cell was resampled. Only 59/74 attempts
have complete token receipts. Known-token columns exclude unknown cancellation
usage; both arm totals and the full billed token total remain unknown. This is
eight fewer requests on this sequence, not an exact monetary saving or a natural
traffic-frequency estimate. The real idle is an explicit experimental condition,
and each turn has one observation, so elapsed times are descriptive only.

## Controlled recovery and tight deadlines

The separate network-free run consumed **52 synthetic transport attempts**:
ON 28, AUTO 24; paid provider calls zero. Synthetic token receipts and simulated
idle are not live billing evidence. For causal isolation it uses System2 off,
skip consultation and the fixed answer `Controlled answer.`; memory/source state
and the same timing ledger still advance naturally through all six turns.

Director/metrics take 1.8/2.7 seconds initially, exceeding unchanged 1.6/2.5-second
waits. AUTO probes both on `cold`, inhibits both on `warm`, and **still inhibits
both on `warm_again` after they become fast (5 ms)**. Advancing the history clock
by 61 seconds at `expired` permits both fast probes to complete usefully. Both
continue to be admitted with completed-history estimates on `recovered_warm`.
The first five pairs return identical answers and create no replacement consults.

Executive takes 1.35 seconds and primary generation 1.45 seconds. In the real
3-second `recovered_warm` RPC, Executive completes before primary generation,
with usable advice and no lost overlap. In the last 400 ms RPC, primary generation
is shortened to 20 ms: AUTO inhibits the known-slow Executive and returns its
answer; ON admits it, returns `DeadlineExceeded`, does not commit, and retains
five sources. AUTO retains six. Every cell has zero pending advisors. This tests
actual timer/ownership behavior with controlled delays, not live provider speed.

The recovery limitation is therefore measurable without a quality claim: recent
maxima plus no probes until expiry can hide a newly fast role. The next policy
work should compare bounded deterministic recovery probes or aging of censored
bounds on independent sequences, counting extra attempts and useful output.
Do not infer a useful-completion benefit merely from a resumed task.

## Replay and remaining scope

At the frozen revision, network-free replay consumed all **74** original events
with **zero** new provider attempts. All twelve statuses, answers, primary drafts,
gate decisions, recorded timing reads, source counts and pending-task counts
matched; all grades matched. Fifteen live timeout labels become immediate
recorded cancellations, so elapsed times, stop reasons and real-time lifecycle
ordering are not reproduced. Replay substitutes the original estimate reads;
it validates request/answer/admission paths, not natural timing adaptation.

Raw artifacts are ignored under
`sr-dual-brain-llm/samples/module-gating-advisor-sessions-2026-11-03-{controlled,live}/`;
the live folder contains `replay/`. Use the frozen revision above:

```bash
python sr-dual-brain-llm/scripts/benchmark_advisor_sessions.py --controlled \
  --max-calls 96 --output-dir /tmp/advisor-sessions-controlled
python sr-dual-brain-llm/scripts/benchmark_advisor_sessions.py --model gpt-6-luna \
  --judge-model gpt-6-sol --max-calls 96 --output-dir /tmp/advisor-sessions-live
python sr-dual-brain-llm/scripts/benchmark_advisor_sessions.py \
  --journal /path/to/original/journal.json --judge-model gpt-6-sol \
  --max-calls 96 --output-dir /tmp/advisor-sessions-replay
```

There was no positive fast Director/metrics AUTO completion in the live sequence;
the controlled run covers it. More sessions, real provider recovery, stricter
live Executive windows, aligned independent rubrics and utility beyond structural
output remain needed. No broad AUTO default promotion follows from this study.
