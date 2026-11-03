# Confirmed advisor recovery — prospective protocol (2026-11-03, Asia/Tokyo)

Freeze code, complete independent fixture, estimator, schedules, rubric and call
cap before paid calls. Preserve #74's original study and grades. Do not choose a
cutoff from its 2,260/2,503 ms pair or change parameters after new outputs exist.

## Resource policy and invariants

Compare v3's single-success reset with v4 confirmation. Hypothalamus keeps timing,
one candidate and opaque leases per retained role/model profile; TRN remains a
pure deterministic gate. Existing 15-second probe quiet time, one probe start per
rolling 60 seconds, native waits, 50 ms reserve, caller modes and defaults stay.
Probe/confirmation windows must allow at least 1,600 ms Director, 2,500 ms metrics
or 2,000 ms Executive, capped by the actual RPC deadline. Executive's later wait
alone is not its launch window. No runtime classifier request or new user flag.

A fast usable probe stays provisional. At the next requested eligible phase,
permit **at most one confirmation** for that probe, claimed only at actual work
start. No confirmation is created as a background call. OFF, cancellation before
start, model changes and an inadequate window spend no confirmation allowance.
Competing/duplicate claims cannot multiply work. Manual ON is ordinary execution.

Promotion needs two fresh usable completions and a prediction
`max(first, second) + abs(first - second) + 50 ms < claimed_window`.
Newer concurrent evidence also has to fit. Confirmed profiles thereafter estimate
the maximum completed latency plus observed completed-latency range, compared
with all censored lower bounds. This is an uncertainty allowance, not a statistical
confidence interval or a guarantee. No distribution parameter is fitted here.
Timeouts and late/empty/error outputs cannot supply a successful observation.

Exploratory evidence retains its actual observation timestamp but its expiry is
capped by the **pre-existing blocking witnesses**. A failed probe/confirmation
therefore cannot postpone the original cold opportunity. A separate provisional
hold prevents ring-buffer eviction from bypassing confirmation; it ends at that
same anchor. Expiry still cannot erase an unexpired rolling probe allowance.
Newer independent ordinary observations retain their own lifetime. Successful
promotion supersedes only older evidence and keeps newer concurrent data.

An unusable ordinary output after promotion suspends further ordinary AUTO work
until the prior valid evidence expires or bounded recovery succeeds; it does not
invent a fast latency from an error. Manual ON remains authoritative. Confirmation
bounds unconfirmed induced work, not all useful ordinary work forever: after
promotion each normal turn is still separately gated and its full cost counted.
The bounded LRU retains at most 32 profiles with 16 samples plus one candidate;
confirmation requires capacity for at least two samples. Discarded profiles have
normal unknown/cold behavior. Persistence/security/source contracts stay outside
optional admission.

## Independent natural sequence

Retain one initially empty production EngineSession per arm, `single_success_v3`
and `confirmed_v4`, for five new questions. All twelve module modes are AUTO;
use left lead, System2 auto/balanced, evaluative dial, Executive observe, observer
both and real 120-second TurnDispatcher caps. The v3 baseline disables confirmation
only through an internal ledger constructor, preserving its eager reset and normal
negative-observation TTL. No deployment switch is added.

Generate primary drafts normally per arm/turn, with matched questions, initial
state, caller modes and per-turn policy seeds. Source, learner and unconscious
state then evolves independently; never seed, clone or replace history. No external
records/PostgreSQL are seeded; conversation record context is disabled but normal
source updates and support calls count. After both initial turns drain, idle for
16 real seconds. Alternate arm order and otherwise proceed immediately. Planned
phase names do not assert that provider latency or history matches an intention.

Question/task obligations are aligned before generation:

| Task | Only requested task obligations |
| --- | --- |
| Wall-clock adjustments | Two sentences; explain elapsed-time failure and name an alternative clock |
| Bulkheads and circuit breakers | Three bullets; one benefit each and one shared limitation |
| Cache miss versus invalidation | Two sentences; distinguish concepts and give one cause each |
| Pacing requests | Two sentences; two apt analogies, one each |
| Retry amplification | Two sentences; explain multiplication and name one mitigation |

Blind grade every available final and primary draft using the established task,
grounding and clarification rubric, shuffled labels and literal quote validation.
Omissions and false asserted claims are distinct; no unrequested next step is
required. Retain all planned cells, failures, malformed grades and regressions.

## Separate advice assessment and paid budget

Benchmark-only wrappers capture completed external advice before any cancellation
request. Keep raw memo/control text private. Canonicalize role/text order before
shuffling so cached completion order cannot alter the judge packet. Exclude profile,
latency, confidence and admission labels from assessment packets. Include question,
reference, advisor role and the corresponding final answer as grounding data.

Use a separate blinded assessment of relevance, grounding and **potential_use**.
Each pass/fail must quote the advice text literally. Generic tone/privacy/direct-
answer boilerplate alone fails potential_use; concrete task-specific guidance,
controls or accurate monitoring observations can pass. Metrics is post-answer:
its observation cannot improve that already returned answer. These grades measure
potential usefulness, not a causal downstream effect or authoritative truth.
Report available, graded and structurally usable outputs and criterion outcomes
by role, including missing exposure and grader errors. Exclude error/late fallback
text; never count it as useful external advice.

Production uses `gpt-6-luna`, both blind assessments `gpt-6-sol`, 2,048 output tokens,
60-second provider timeout, no continuations. Hard cap: **96 HTTP attempts total**,
including ten full turns, all probes/confirmations/support/repair, five answer-judge
packets and up to five advice-judge packets. No free warmup, resampling or paid
replay. Transport/auth/quota failure blocks new paid requests. Unknown cancellation
receipts remain unknown; do not infer exact monetary savings.

## Controlled logical-time coverage

Use a separate network-free model of service completion in real retained production
sessions. Only metrics is caller-selected; Executive is off, observer metrics,
System2 off and consultation skipped. Primary output is fixed `Controlled answer.`;
primary/support advance each session's logical clock by 30/2 ms. Metrics fractions
below multiply its unchanged 2.5-second window. Responses at or above the window
are explicitly censored at 2.5 seconds by the transport. This models cancellation;
it is not a wall-clock timeout measurement. Independent tests use real owned-task
timeouts/RPC cancellation, including the confirmation path.

| Regime | Five service fractions | Scheduled logical starts |
| --- | --- | --- |
| Steady recovery | over, .20, .22, .21, .20 | 0, 19, 23, 27, 63 s |
| Near-window jitter | over, .70, .90, over, .20 | 0, 19, 23, 27, 63 s |
| Lucky fast result | over, .10, over, over, .20 | 0, 19, 23, 27, 63 s |
| Missed useful opportunity | over, .70, .90, .50, .20 | 0, 19, 23, 27, 63 s |
| Recovery after failed probe | over, over, over, .20, .22 | 0, 19, 23, 63, 80 s |

Initial observation ends near 2.53 s, so 19 s allows the existing 15-second quiet
period after completion. Clocks only advance; no samples are seeded. Predict equal
usable outputs/calls for steady recovery, fewer speculative failures for jitter,
an earlier original cold opportunity after failed confirmation/probing, and a
temporary lost fast output in the deliberately unfavorable missed-opportunity
regime. Report that loss and the cold resumption rather than selecting only wins.
Count both exploratory and induced ordinary work, time/turns to usable output,
missed calls, subsequent cancellations, support/repair and drained tasks. Synthetic
receipts/fixed answers are not live billing or answer-quality evidence.

## Replay, publication and limits

Record actual resource reads, probe/confirmation claims and finishes. Replay only
substitutes recorded observations/capabilities; instantaneous cached responses
must not retrain latency. Require exact HTTP occurrences and matching final/draft
answers, canonical advice outputs/grades, gate decisions, normalized resource
occurrences and source/pending counts with zero fallback transport. Preserve any
timer/cancellation/order mismatch; replay does not reproduce real timing.

Raw artifacts stay in ignored `samples/module-gating-advisor-confidence-*/`.
Publish safe aggregate projections without prompts, advice text or credentials.
One correlated sequence per arm, two timing observations, synthetic controls and
model-based potential-use grades cannot justify broad AUTO default promotion.
