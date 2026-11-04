# Advice criterion boundaries — prospective protocol (2026-11-04, Asia/Tokyo)

Freeze source, this complete independent fixture, both rubric/schema contracts,
label/order schedule, fixed expected verdicts and the 24-attempt cap before paid
calls. Preserve the #78/#80 source-pinned outputs, expectations and failures.
This study does not rescore the old weather example to erase its disagreement.

## Contract and explicit adoption

The original structured evaluator remains `advisor_assessment.py` with unchanged
instructions and callers. `advisor_assessment_v2.py` is an explicit evaluation
API for new studies; historical runners do not switch to it. No runtime classifier,
production request path, module default or admission policy changes.

Define each criterion separately while allowing overlapping failures:

| Criterion | Assesses | Boundary |
| --- | --- | --- |
| Relevance | Topic, role, requested scope and step count | A task-aligned false statement may remain relevant; harmless off-topic/style additions and extra checks may fail despite true facts |
| Grounding | Factual support, observed versus unobserved operations, operational authority and observer access | Harmless unrequested conversation/style choices are not unsupported operations; explicit read-only restrictions still prohibit state-changing writes |
| Potential use | At least one concrete task-specific useful element | Useful partial guidance may pass while another criterion fails; this is not net safety or causal utility |
| Critical coverage | Required content, material constraints, qualifications and scope | Omission or contradiction can fail without inventing a factual assertion |

Grounding does not become a catch-all for relevance. A proposed permitted check
need not have been executed, while claiming it already passed is unsupported.
Operational authority concerns access/state-changing actions under supplied
restrictions, not permission for every conversational choice. An omission alone
does not assert a false fact. A prohibition can contain an unsupported positive
explanation, so prohibitions never receive an automatic pass.

Keep memo/control/directives/mix-in structured, with canonical field paths,
literal quotes, assertion/prohibition/control/uncertainty interpretation and the
complete selected field retained privately. V2 also requires a criterion-specific
`basis` enum: task scope cannot be supplied as a grounding basis. The model still
chooses the semantic verdict and basis; schema validation is not a correctness
oracle and never consults expected results.

Metrics grounding uses only its actual observer input. Full final-answer/reference
comparison data do not grant access to omitted content. Preserve a supported
visible observation and unknown omitted content separately: uncertainty cannot
deny explicitly supplied facts.

## Fresh synthetic comparison

Use six new cases, three authored advice variants each. Cases and complete expected
pass/fail vectors are fixed in `examples/advice_boundary_calibration.json`.
They are synthetic calibration examples, not generated production answers or
real executed operations. Every answer obligation appears in its question and
role-specific obligations; no unrequested next check is added after generation.

| Case | Three independently authored contrasts |
| --- | --- |
| Loading-bay barrier | Unobserved closed-position assertion, explicitly prohibited assertion, correct uncertainty plus proposed sensor check |
| Invoice subtotal | Correct plain guidance, harmless prohibited decoration, incorrect subtotal with useful format reminder |
| Rate-limiter validation | One proposed experiment, unexecuted extra check, fabricated successful result |
| Inventory authority | Read-only proposal, write against SELECT-only grant, invented read execution |
| Pressure observation | Complete units/time guidance, omitted units, false calibration rationale inside a prohibition |
| Metrics access | Supported partial observation, borrowing unseen full-answer content, denial of the visible observation |

For each case, deterministically shuffle variant-to-A/B/C labels for order 0 and
reverse them for order 1. Use identical packet labels within each v1/v2 pair and
alternate which rubric runs first via `(case_index + order) % 2`. Both receive
the same question, reference, role obligations, structured fields, evidence paths,
actual input facts and final-answer comparison data. Expected values, case IDs,
variant names, source/rubric-arm labels and presentation-order flags never enter
the grading packet. Each grader necessarily receives its own rubric/schema; this
is not a claim of blindness to the intervention's wording.

This yields **24 packets**: 6 cases x 2 rubrics x 2 label orders. Each rubric has
36 output presentations and 144 criterion checks, but only **18 distinct advice
variants in six cases**. The two orders are a stability check, not independent
replications. Do not report 144 independent trials or infer population significance.

## Prospective interpretation

Report every criterion's fixed-expectation agreement, actual pass/fail/uncertain
counts, valid/invalid/unattempted packets, fully matched presentations and variants
matching both orders, order disagreements and paired corrections/regressions.
Keep incomplete/malformed responses and transport failures without replacement.

The primary boundary checks are v2 grounding of:

- Harmless invoice decoration, the extra rate-limiter check and omitted units:
  pass in both presentations, while their separate scope/coverage failures remain.
- Unauthorized inventory writes, the false calibration rationale and borrowed
  observer content: fail in both presentations.
- Unsupported barrier closure versus explicitly prohibited closure and correct
  uncertainty: fail/pass/pass respectively in both presentations.

These are nine fixed variants (18 grounding presentations). Assess all other
variants/criteria too. A successful bounded boundary result requires these focus
checks correct, complete valid v2 packets, and v2 overall grounding agreement at
least v1. Report failures/regressions even if that focus succeeds. This is not a
guarantee that every adviser or future rubric will be correct; v2 remains an
explicit API and does not automatically replace historical evaluation contracts.

## Cost, replay and publication

Use `gpt-6-sol` for both rubrics, 4,096 output-token caps, 60-second provider
timeouts and no continuations. The provider's supported request options, including
GPT-6 temperature behavior, are recorded. The hard cap is **24 HTTP attempts
total**, including all failed/completion attempts. No warmup, retry, resampling,
rescue judge, paid replay or production LLM turns are permitted in this study.
Transport/auth/quota failure blocks subsequent paid calls through the shared
monotonic journal latch; retain planned unattempted cells and unknown receipts.
No partial receipt is treated as a free call or zero tokens.

Live CLI use requires a clean committed workspace. Freeze both rubric hashes,
fixture hash, relevant source hashes and source revision before calls. Exact
replay consumes all recorded HTTP occurrences with no fallback transport, matching
original normalized scores (including full private contexts), expected values,
packet errors and aggregate summaries. Replay checks recorded request/grade paths,
not independent judge reasoning, real latency or repeated-measure reliability.

Raw requests, replies, quotes and selected-field contexts remain in ignored
`samples/module-gating-advice-boundaries-*/`. Publish safe verdict/basis/path
aggregates and artifact hashes without raw advice or credentials. This evaluation
measures agreement with declared criterion contracts; it does not establish
production answer quality, operational safety, causal usefulness or cost savings.
