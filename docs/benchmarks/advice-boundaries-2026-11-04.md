# Advice criterion boundaries — 2026-11-04 (Asia/Tokyo)

The explicit v2 contract matches **142/144** fixed criterion expectations versus
**136/144** for v1. Grounding agreement is **36/36 versus 32/36**, including
**18/18** predeclared focus checks versus 14/18. All 24 packets are valid. This
satisfies the prospectively defined bounded boundary criterion, not a guarantee
of universal grader correctness. There are **eight paired corrections and two
regressions relative to fixed expectations**, and each rubric still has two
label-order disagreements.

Both rubrics, schemas, all six new cases/18 authored variants, expectations,
presentation schedule and 24-attempt cap were frozen at
`1eb8aaf881f2c590025caf07b98285dd6876e6af` before paid calls. See the
[prospective protocol](advice-boundaries-protocol.md) and
[safe metrics projection](advice-boundaries-2026-11-04.metrics.json). Nothing was
resampled, rescored or tuned after new outputs existed. The prior #78 advice
failures and #80's 29/30 calibration remain unchanged.

## Explicit criterion contracts

The remaining #81 ambiguity concerned whether an irrelevant proposed control
should fail grounding as well as relevance. V2 defines the distinction forward:

- Relevance measures topic, role, requested scope and step count. A task-aligned
  factual error need not automatically fail relevance.
- Grounding measures supported assertions, observed operations, operational
  authority and observer access. Harmless extra styling or an unexecuted permitted
  check does not become an unsupported operation merely because it is unrequested.
  A write against an explicit read-only grant still fails grounding.
- Potential use asks for at least one concrete useful element. It is not net
  endorsement or operational safety; valid partial guidance can coexist with
  another failure.
- Critical coverage measures required facts, qualifications and constraints.
  Omitting a unit or contradicting a one-check requirement can fail coverage
  without inventing a factual assertion.

Failures may overlap; these are not mutually exclusive bins. A prohibition is
not automatically true or safe: a positive explanatory rationale in a negated
instruction can still fabricate a fact. Metrics cannot borrow omitted content
from full-answer comparison data, or deny a fact visible in its actual input.

`advisor_assessment_v2.py` retains structured fields, canonical quote paths,
literal quotes, full selected-field context and polarity interpretation. It adds
a criterion-specific `basis` enum; task scope cannot be supplied as a grounding
basis. The model still chooses semantic verdicts and bases. Validation checks
structure/provenance, not truth, and never reads expected results. The experiment
compares changed rubric wording **and** schema together, not an isolated wording
effect.

The original evaluator and every historical runner keep their original contract.
V2 exposes `assess` and `grade_report` for explicit use in new studies. A future
benchmark can reuse v1's `structured_output` capture, opt into v2 grading and
record its version/rubric hash before new calls. No production request, runtime
classifier, enable/default policy or original score changes here.

## New matched calibration and results

Six newly authored cases cover unobserved barrier position, invoice
decoration, rate-limiter check count, inventory read/write authority, sensor units/
calibration and metrics snippet access. They are synthetic calibration examples,
not generated production answers or actual operations. Each has three variants
with complete fixed expected verdicts on all four criteria.

For each case, opaque A/B/C labels are deterministically shuffled then reversed.
Both rubrics receive identical packet content/labels within a comparison, with
counterbalanced rubric order. Expected vectors, case/variant IDs, arm labels and
order flags are withheld. Each grader necessarily receives its own instructions
and schema. Two label orders are a stability probe, not independent replications:
there are **18 distinct variants in six cases**, 36 presentations per rubric,
and 144 correlated criterion checks per rubric.

| Fixed-expectation agreement | Structured v1 | Criterion v2 |
| --- | --- | --- |
| Relevance | 34/36 | 35/36 |
| Grounding | 32/36 | **36/36** |
| Potential use | 34/36 | **36/36** |
| Critical coverage | **36/36** | 35/36 |
| All criterion checks | 136/144 | 142/144 |
| Presentations matching all four criteria | 28/36 | 34/36 |
| Distinct variants matching all criteria in both orders | 13/18 | 16/18 |
| Criterion/variant pairs whose verdict changes with label order | 2 | 2 |

All 12 packets per rubric are valid, with zero malformed/uncertain judgments or
unattempted cells. V1 fails grounding on the harmless invoice decoration and extra
rate-limiter inspection in both orders: four mismatches relative to the declared
scope/grounding boundary. V2 passes grounding for those while retaining their
relevance/coverage failures. Both retain grounding failures for unauthorized
writes, fabricated results, a false calibration rationale and borrowed observer
content. V2 also preserves grounding for correct negation, legitimate uncertainty
and an omitted unit while retaining the omission's coverage failure.

The primary focus is nine predefined variants in both orders: harmless decoration,
extra check, omitted unit, unauthorized write, false rationale, borrowed observer
content and the three barrier contrasts. V2 matches **18/18**, has complete valid
packets and exceeds v1 overall grounding agreement, meeting the protocol's focus
criterion. This does not erase secondary failures or establish population accuracy.

## Preserved regressions and limits

Relative to fixed expectations, v2 corrects four grounding, two relevance and two
potential-use checks. It also introduces two mismatches where v1 agreed:

- The correct-uncertainty barrier variant fails v2 critical coverage in reversed
  order only, with `omission` as its basis. The whole field contains unknown
  position and a proposed fresh sensor check. Its expected pass and model failure
  both remain recorded; no new required detail is retroactively added.
- The observer variant that claims unseen full-answer coverage fails v2 relevance
  in reversed order only, with `task_scope` as its basis. Its grounding/coverage
  failures are correctly preserved. Expected relevance pass treats this as a
  task-aligned factual/access defect, while the model's failure can reflect an
  overlapping role-scope interpretation. This is a remaining contract/expectation
  boundary, not proof that all additional failures are objectively wrong.

V1's order disagreements concern relevance of fabricated limiter results and
invented inventory execution. V2's concern the two cases above. The disagreement
count is unchanged: better fixed-expectation agreement does **not** demonstrate
general label-order robustness. Full contexts remain available privately, so an
isolated short quotation is not being substituted for the complete field.

These results support the bounded #81 distinction on fresh examples. Future work
should define coverage sufficiency and observer role-scope overlap using new
independent examples, then examine order stability without rescoring this study.
No grader is promoted into production admission or treated as authoritative truth.

## Usage, tests and exact replay

Both rubrics use `gpt-6-sol`, 4,096 output-token caps, 60-second provider timeouts,
no continuations and the provider's supported request options. All **24 allowed
HTTP attempts** are included. There is no warmup, retry, rescue judge, production
model turn or paid replay. Every response has complete usage; no cancellation,
transport/auth/quota failure or unknown receipt occurs.

| Rubric | Attempts | Input tokens | Output tokens |
| --- | --- | --- | --- |
| Structured v1 | 12 | 22,122 | 8,240 |
| Criterion v2 | 12 | 28,374 | 8,073 |
| Whole study | **24** | **50,496** | **16,313** |

The longer v2 contract/schema adds **6,252 input tokens (28.3%)** in this sequence;
output is 167 tokens lower. This is a token tradeoff, not a cost-saving result or
an exact monetary claim. Actual billing prices/cache treatment are not measured.

Replay at the frozen source consumes all **24 recorded HTTP occurrences with
zero new requests**. Original normalized scores, full private contexts, expected
vectors, packet errors/usage and aggregate summaries match exactly. Replay checks
recorded request/grade paths; it does not reproduce independent judge reasoning,
real service latency or repeated-measure reliability.

Local validation passes **1,307 Python tests**, plus **17 Gateway tests** with two
expected skips and **13 Node UI tests**. The 12 new focused cases cover criterion-
specific schema isolation, prohibited text and full-context provenance, invalid
bases/quotes, fixture blindness/completeness, preserved mismatches, exact replay,
incomplete responses, blocked transport and opt-in v2 report grading. Those focused
cases were rechecked after normalizing recorded transport-error names for replay;
legacy rubric defaults remain unchanged.

Raw requests, scores, quotes and contexts remain in ignored
`sr-dual-brain-llm/samples/module-gating-advice-boundaries-2026-11-04-live/`, including
`replay/`. Use the frozen source for the original contracts:

```bash
python sr-dual-brain-llm/scripts/benchmark_advice_boundaries.py --model gpt-6-sol \
  --max-calls 24 --output-dir /tmp/advice-boundaries-live
python sr-dual-brain-llm/scripts/benchmark_advice_boundaries.py \
  --journal /path/to/original/journal.json --max-calls 24 \
  --output-dir /tmp/advice-boundaries-replay
```

The live CLI requires a clean committed tree. The safe projection publishes
verdict/basis/path aggregates and hashes, excluding raw advice, quote contexts,
credentials and HTTP payloads. Original #78 and #80 studies stay intact.
