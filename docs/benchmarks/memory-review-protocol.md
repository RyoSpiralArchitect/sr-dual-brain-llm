# Memory → ACC → System2 pilot — prospective protocol

Date: 2026-11-04 (Asia/Tokyo). Freeze this protocol, code, both new fixtures,
expectations, schemas, rubric wording, order and budget in a clean source commit
before any paid request. Preserve all earlier studies and their disagreements.

## Bounded implementation

Conversation memory already excludes expired, superseded and derived answers and
requires exact query/source answerability approval. Approval establishes requested
fact support, not consistency between records. ACC previously consumed the
deterministic micro-critic but not competing historical answers; an offline
regression reproduces a high-confidence selection with no ACC promotion.

`SharedMemory.competing_answer_groups` examines only the three complete approved
records exposed in the normal memory context. Two unique current source IDs must
have the same declared scope and **exact source question**, with different answer
text after stripping outer whitespace. This is a review cue, not a semantic
contradiction verdict, trust ranking or inferred supersession. Equivalent wording
can trigger it; different questions/languages can hide a real contradiction.
Partial and conditional answers can coexist, so the cue requests reconciliation
and preserves alternatives rather than declaring either false.

The historical-evidence context carries that qualification. Existing ACC consult
and System2 promotion gates receive a typed `memory_competition` fact. ON/AUTO
remain opt-in (both defaults stay OFF); no new organ/module/default is needed.
Their policy provenance becomes `acc_memory_review_v2`. The new signal applies
only to System2 AUTO, capable left-led turns with supplied approved context.
Explicit OFF, right lead, memory drop, missing capability and individual gate OFF
retain authority. It does not change temperature, source verification, correction
transactions, deadlines, record authority or persistence.

## #84: new obligation calibration first

`advisor_assessment_v3` is an explicit future evaluation API. V1/V2 and historical
callers remain unchanged. Both comparison arms receive the same finite mapping
of atomic role obligations. V3 requires a pass/fail/uncertain check for every ID,
with coverage fail if any fails, pass if all pass, otherwise uncertain. Validation
checks the audit's completeness/internal consistency and canonical quote/context;
it does not determine the semantic verdict or consult expected results.

An unknown outcome plus the specifically requested proposed check can be fully
sufficient; do not invent operational detail. A false claim about seeing hidden
task-related text fails grounding/coverage but does not alone change topic or
actor role. Directing a monitor to operate a service changes actor scope and can
also violate operational authority. These are explicit conventions for this
rubric, not a claim that every plausible alternative convention is objectively
wrong. Ordinary scope/count restrictions still apply independently.

Four fresh cases × three authored variants × two rubric arms × two reversed label
orders yield **16 packets**, 24 presentations/96 criterion checks per rubric, but
only **12 distinct variants in four cases**. The cases contrast qualified unknown
versus omitted check, missing units, false observer-access attestation versus
unrelated topic, and monitor versus service operator. These are calibration
outputs, not production adviser replies or evidence of causal advice benefit.
Both arms use `gpt-6-sol`, 4096 output tokens and 60-second provider timeouts,
without continuation. Labels/order are inherited from the previous boundary
runner; expected values/variant IDs are never sent. Rubric order alternates by
case index plus label order. No rescue judge, adjudication or selective repeat.

Report criterion agreement, paired corrections/regressions, order disagreement,
obligation agreement and missing/invalid packets separately. The fixed focus is
V3 coverage of qualified unknown, omitted check and missing units, and relevance
of task-aligned false access, unrelated topic and actor change, in both orders.
Failure is retained; a structured audit does not guarantee correctness/stability.

## Six-turn retained-session dialogue

Run three retained production EngineSessions: one natural initial-draft session,
and paired baseline/treatment sessions consuming the **same initial draft** at
every turn. Draft generation uses System2 OFF and the historical left-only harness
so there is exactly one initial draft. Baseline removes only the new competition
helper from context/ACC; treatment keeps it. All other optional modules use AUTO,
except Director/executive/metrics advice, explicitly OFF to isolate this handoff.
Final-answer judging uses the existing task/grounding/clarification contract with
explicit per-turn requirements; the private-advice calibration is separate and
must not be presented as an answer-grader reliability estimate.

The single fresh trajectory in `memory_review_dialogue.json` has: literal Cedar
replay; a second active answer to the same question; explicit correction of both;
pronoun carryover; switch to Larch; cross-project comparison; expired Moss source
(the correction and pronoun are one turn, for six total). Source creation and
correction use declared **authored fixture records** through normal SharedMemory
operations, not generated primary writes. All derived process answers stay
non-sources. No generated reply is promoted to a trusted fact. This measures
in-process dialogue and routing, not PostgreSQL commits/restart durability; the
existing real-PostgreSQL regression suite checks those separate contracts.

Use a fixed logical wall clock (90-second increments), matched Python/NumPy/Torch
and UUID seeds per turn, identical initial source states, and alternating paired
arm order. Source IDs/hash, route/ACC provenance, actual roles/calls, missing
answers, literal-quote grades, usage and handler duration are retained. Stochastic
provider outputs remain stochastic; the draft generator does not see the new cue.
Thus this pilot measures the post-draft handoff, not the new cue's effect on natural
initial drafting. Later trajectory effects and six turns are not independent
replications. No population significance or general score improvement is implied.

Consultation on optional lookup turns is a candidate for optimization, not
automatically "unnecessary": a nonliteral/incomplete draft can need review.
Record exact replay coverage and route decisions before making that claim.
Compare the two required-review turns, explicit correction, scope bindings and
missing-evidence clarification separately, and preserve regressions/unknowns.

## All-attempt budget, failures and replay

Use `gpt-6-luna` production/support, `gpt-6-sol` judges, production client output
caps of 2048, support's existing 4096 cap, no continuation. The **128 HTTP-attempt
cap is shared across the whole run**, including the 16 calibration attempts,
initial drafts, three sessions' support checks, critic/integration/repair calls,
and six final-answer judge packets. No warmup, paid replay, extra models, retries,
resampling, budget reset or post-result expectation/rubric edits. Source/endpoint
and public config hashes bind the journal. Transport/auth/quota failure blocks
subsequent paid requests monotonically. Retain every planned missing cell and
unknown usage; do not equate an unknown receipt with zero cost.

Replay consumes exact recorded payload occurrences with `journal.transport=None`
and requires `finish()` to verify no missing or unused request. Compare grades,
private field contexts, calibrated summaries, answers/status, evidence IDs/hash,
route and usage exactly. Real handler duration is excluded from equality: replay
does not validate network latency or repeated-measure reliability. Any divergence
is reported, not silently repaired with paid calls. Raw packets/answers/quotes
remain under ignored `samples/module-gating-memory-review-*/`; publish only safe
aggregates and artifact hashes. The experiment's clean source commit is its
reproduction reference even if later review improves unrelated validation code.

```bash
python sr-dual-brain-llm/scripts/benchmark_memory_review.py \
  --model gpt-6-luna --judge-model gpt-6-sol --max-calls 128 \
  --output-dir sr-dual-brain-llm/samples/module-gating-memory-review-2026-11-04-live
python sr-dual-brain-llm/scripts/benchmark_memory_review.py \
  --journal sr-dual-brain-llm/samples/module-gating-memory-review-2026-11-04-live/journal.json \
  --judge-model gpt-6-sol --max-calls 128 \
  --output-dir sr-dual-brain-llm/samples/module-gating-memory-review-2026-11-04-replay
```
