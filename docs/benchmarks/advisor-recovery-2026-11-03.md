# Bounded advisor recovery — 2026-11-03 (Asia/Tokyo)

Bounded recovery resumed one post-turn metrics observer **before its slow history
expired**. Both profiles passed 5/5 independent live tasks on every criterion.
Whole-turn attempts were 26 TTL-only versus 25 recovery, but structurally usable
advisor completions were **8 versus 6**, and cancellations **3 versus 4**. The
result demonstrates bounded re-evaluation, not a general utility or savings gain.
The recovered observer's next ordinary attempt timed out and was inhibited again.

Production source, parameters, the complete fixture and
[prospective protocol](advisor-recovery-protocol.md) were frozen at
`34cb1ce44c25a3f44e546edd4af1ef022e94f132` before paid calls. No threshold, fixture,
grade or answer was changed afterward. See the
[safe metrics projection](advisor-recovery-2026-11-03.metrics.json) for original
grades, role counts, readiness, claims and outcomes. Prior studies, including
the #72 rubric defect and original scores, remain unchanged.

## What changed

Hypothalamus owns an opaque lease per retained role/model profile. After 15
seconds without a new observation, AUTO v3 can try known-unaffordable work once
per rolling 60 seconds if its window is sufficiently large: 1,600 ms Director,
2,500 ms metrics or 2,000 ms Executive. Native waits and RPC deadlines are not
extended. TRN remains a pure admission function, with no classifier or recursive
gate. The canonical module modes and compatibility defaults stay unchanged.

Claim at actual work start after rechecking profile, deadline and allowance.
OFF, missing caller modes and pre-start cancellation spend nothing. Manual ON
remains ordinary execution. A fast structurally usable external completion plus
the existing 50 ms reserve must fit its claimed window before older evidence is
replaced; newer concurrent observations survive. Slow/empty/error/late probes
spend the allowance without removing slow history. Sample expiry does not reset
a recent probe limit. The bounded 32-profile LRU can discard a profile, which
subsequently has ordinary unknown/cold behavior.

The benchmark compares `ttl_only_auto`, using the new ledger with recovery
disabled internally (v2-equivalent recent-max/TTL behavior), against
`recovery_auto`. All twelve canonical modes are AUTO in both. The constructor
injection is an evaluation baseline, not another user deployment switch.

## Controlled causal coverage

Two separate network-free regimes retain one production EngineSession per arm
across initial, opportunity, followup, relapse and capped turns. No history is
seeded. For causal isolation only, use System2 off, skip consultation, fixed
`Controlled answer.` output, synthetic receipts and a 16-second history-clock
advance at opportunity. Real native advisor waits and TurnDispatcher ownership
still govern every task. Executive/primary take 10/30 ms and support takes 2 ms.

| Controlled regime | TTL-only calls | Recovery calls | Useful structural recovery and subsequent work |
| --- | --- | --- | --- |
| Slow → fast → slow | 16 | 22 | Two successful probes; two subsequent ordinary fast completions; two ordinary relapse timeouts |
| Always slow | 16 | 18 | Two unsuccessful probes; no recovered output and no further attempt before the limit |

On initial, Director/metrics take 1.8/2.7 seconds, exceeding their unchanged
1.6/2.5-second waits. In the recovery regime, both become 5 ms on opportunity and
followup. AUTO's one probe per role restores ordinary admission before TTL expiry;
both become slow again on relapse, then remain inhibited on capped. In the slow
regime, the two failed opportunity probes are the only additional calls. Each
profile starts exactly one recovery probe within the observed rolling window.

The six extra calls in the recovering regime include **ordinary work induced by
recovery**, not six recovery probes: two probes, two fast ordinary completions,
two failed ordinary calls on relapse. A probe bound alone does not bound all
future admitted work. Both regimes together use 72 synthetic attempts and **zero
paid provider calls**. All twenty controlled finals are equal, every cell drains
its tasks, and no replacement consultation occurs. These fixed outputs and
synthetic token counts do not establish live quality, downstream guidance value
or monetary savings.

Offline tests additionally cover failed/empty/late output, cleanup time exclusion,
competing/duplicate claims, model changes and eviction, newer concurrent evidence,
sample expiry with a still-active limit, ON/OFF, unknown windows and tight/ample
Executive RPC budgets. Local validation passed 1,196 Python tests including real
PostgreSQL; Gateway passed 17 with two expected skips; UI passed 13.

## Natural live sequence

Keep one initially empty EngineSession per arm for five turns. Initial state,
questions, caller modes and per-turn policy seeds match, while ordinary source,
learner and unconscious state evolves independently. Primary drafts are generated
normally per arm; no warmup, shared draft or history transplantation is excluded
from cost. After both initial turns drain, idle for **16.016 seconds** in real
monotonic time, then alternate arm order at each phase. Phase names are intended
opportunities, not assumed provider-speed regimes.

Use left lead, System2 auto/balanced, evaluative dial, Executive observe and
observer both, with a real 120-second TurnDispatcher cap. No external records or
PostgreSQL are seeded; conversation record context is disabled. Ordinary source
updates still generate four support calls per arm, all included below.

| Phase / independent task | TTL-only calls | Recovery calls | Recovery-arm observation |
| --- | --- | --- | --- |
| Initial: LRU eviction and recency limitation | 6 | 6 | Director and metrics cold probes both time out |
| Opportunity: jitter, retry limit, idempotency benefits | 5 | 6 | Director recovery times out; metrics recovery succeeds |
| Followup: lease expiry and stale-write rejection | 5 | 5 | Ordinary metrics admission times out again |
| Relapse: concurrent-query limit analogies | 6 | 4 | Both slow roles inhibited under the rolling limit |
| Capped: retry budget and a stopping signal | 4 | 4 | Both slow roles remain inhibited |
| Total, including natural primary generation | **26** | **25** | Two recovery starts: one usable, one cancelled |

The metrics recovery replaces a still-current censored **2,502.9 ms** observation
with a usable **2,259.8 ms** completion. Its next gate sees that completed estimate
and admits ordinary work, which is cancelled at **2,502.8 ms**; its true completion
time and billed usage are unknown. The new censored bound then inhibits later
turns. Director's recovery is cancelled at 1,602.3 ms and likewise spends its
allowance. No second recovery starts for either profile in this sequence.

This demonstrates earlier re-evaluation within the recovery arm's own history.
The TTL-only arm already had a fast metrics completion on initial, whereas the
recovery arm did not. Its metrics subsequently complete three times versus once
in the recovery arm. Independently sampled provider latency and differing history
prevent claiming earlier recovery than the TTL-only arm in this live comparison.
TTL-only Director history naturally expires by relapse and triggers a cold retry;
recovery's newer failed probe remains recent. No intended phase is forced by
resetting or seeding observations.

## Quality, utility and full accounting

All ten finals and ten natural primary drafts are available and blind graded
with shuffled labels and literal quote validation. **Both profiles pass 5/5** on
task, grounding and clarification, with no malformed grade or paired criterion
regression. The new task rubric requires only explicitly requested content and
format; it does not repeat #72's unrequested-next-check criterion. Primary drafts
already pass 5/5, so this study shows no measured final-answer improvement.

Production roles use `gpt-6-luna`; the blind judge uses `gpt-6-sol`, with 2,048
output tokens, no continuation and a 60-second provider timeout. These are five
serial turns in one session per arm, not five independent replications. Model
grades and nonempty guidance do not certify factual correctness or guidance's
downstream effect. No broad quality or automatic-default claim follows.

| Live advisor role | TTL-only starts / usable / timeout | Recovery starts / usable / timeout | Recovery probe starts |
| --- | --- | --- | --- |
| Director | 2 / 0 / 2 | 2 / 0 / 2 | 1 |
| Executive advice | 5 / 5 / 0 | 5 / 5 / 0 | 0 |
| Executive metrics | 4 / 3 / 1 | 3 / 1 / 2 | 1 |
| Total | **11 / 8 / 3** | **10 / 6 / 4** | **2** |

Every admitted live task starts and every cell has zero pending advisors.
Source counts advance one through five in both sessions. All ten Executive tasks
complete usefully; the live study exposes no Executive recovery claim. No other
provider failure or cancellation occurs. Useful output here means structural
usability, not answer correctness or net utility.

| Whole-turn HTTP role | TTL-only | Recovery |
| --- | --- | --- |
| Primary generation | 5 | 5 |
| Support from retained session sources | 4 | 4 |
| Right consultation/verification | 5 | 5 |
| Left integration/repair | 1 | 1 |
| Executive advice | 5 | 5 |
| Director | 2 | 2 |
| Executive metrics | 4 | 3 |
| Total | **26** | **25** |

There is no aggregate increase in primary/support/right/left work. The one-call
difference is a metrics attempt, not a proven causal saving from probes; baseline
cold retries and independently varying response times change the paths.

| Live phase | Attempts | Unknown cancellation receipts | Known input / output tokens |
| --- | --- | --- | --- |
| TTL-only complete turns | 26 | 3 | 16,182 / 3,454 |
| Recovery complete turns | 25 | 4 | 15,263 / 3,277 |
| Shared blind judging | 5 | 0 | 4,672 / 2,896 |
| Whole study | **56** | **7** | **36,117 / 9,627** |

The prospective maximum was 96 attempts. There was no resampling, output editing
or paid replay. Only 49/56 attempts have complete token receipts; known-token
columns exclude cancelled usage. Both billed arm totals and the study total
remain unknown. Extra probes have a cost even when aggregate attempts happen to
decrease. No exact monetary or latency-significance claim is supported.

## Replay and next evidence

At the frozen revision, replay consumes all **56** events with **zero** new
provider requests. All ten statuses, answers, drafts, source/pending counts and
gates match; all grades and normalized resource read/claim/finish occurrences
match. Seven live timeouts replay as immediate recorded cancellations, so real
elapsed time, stop reasons and async completion order are not reproduced.
Recorded recovery capabilities are used only during replay; this validates
request/answer/admission paths, not fresh latency learning from cached responses.

Raw artifacts remain private under ignored
`sr-dual-brain-llm/samples/module-gating-advisor-recovery-2026-11-03-{controlled-recovery,controlled-slow,live}/`.
The live folder contains `replay/`. Use the frozen revision for original replay:

```bash
python sr-dual-brain-llm/scripts/benchmark_advisor_recovery.py --controlled recovery \
  --max-calls 96 --output-dir /tmp/advisor-recovery-controlled
python sr-dual-brain-llm/scripts/benchmark_advisor_recovery.py --controlled slow \
  --max-calls 96 --output-dir /tmp/advisor-recovery-always-slow
python sr-dual-brain-llm/scripts/benchmark_advisor_recovery.py --model gpt-6-luna \
  --judge-model gpt-6-sol --max-calls 96 --output-dir /tmp/advisor-recovery-live
python sr-dual-brain-llm/scripts/benchmark_advisor_recovery.py \
  --journal /path/to/original/journal.json --judge-model gpt-6-sol \
  --max-calls 96 --output-dir /tmp/advisor-recovery-replay
```

One successful near-window completion is not stable recovery evidence: the next
metrics request missed its window. A failed probe also renews negative evidence,
potentially delaying a later cold opportunity versus old TTL-only history.
Further work should compare recovery confidence, censored-history aging and
bounded confirmation on new jitter/late-recovery sequences, counting induced
ordinary calls and actual guidance utility. Keep this study and its parameters
frozen; compatibility defaults remain unchanged.
The next evaluation is tracked in
[#76](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/issues/76).
