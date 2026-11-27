# Coordinated field verification and integration

Measure a coordination rule for scalar JSON answers. A critic proposes
specific edits. A fresh verifier derives values for every answer field
from the question. An integration then decides whether to adopt the
supported proposals. The candidate can return a corroborated draft,
commit only agreed changes, or use the matched verified final.

## Frozen cohorts

- Fixed screen: 12 mixed drafts across Boolean logic, SQLite, event
  state and source authority, plus four initially exact controls.
  Mixed drafts have known wrong fields and known correct fields under
  attack by a proposed change. The fixed critic proposes both correct
  and damaging edits. Their labels are evaluator-only.
- Natural holdout: 12 separately authored questions across the same
  four domains. Luna generates each draft and its critique through the
  native roles. The critic uses the existing one-issue-per-field format.
  Exact answer keys and oracle derivations are computed independently
  before provider calls. The verifier sees the question and the field
  names that the question already requests.
- Two repetitions per task and two trials per wave. Within each cohort
  use seed 2811, randomized task order and opposite final launch order
  on the two repetitions for each task. Freeze both cohorts, design,
  scoring and code before the fixed run. Run both cohorts with no
  output-derived changes to the protocol or fixtures.
- Use standard gpt-6-luna throughout. Use 24 seconds and 4,096
  completion tokens per call; disable retries, continuation, local
  critic fallback, work products and controller intervention. A missing
  or malformed draft blocks the structured route. A transient failure
  or deadline remains in the denominator. A request-contract, model or
  authentication error stops the run after its current wave and keeps
  attempted receipts.

## Matched policies

All live final generations receive the same original question and draft.
The unfiltered arm receives all well-formed critic proposals. The verified
regeneration arm receives those whose typed values match the independent
verifier and whose verifier evidence passes the existing exact quote check.
Both use a neutral note if their list is empty.

The coordinated policy uses the same verifier and samples. A usable check
must exist for every original field. If every checked value matches the
draft, return it unchanged without a final-generation admission in this
route. If checked values differ, commit only when each difference has a
matching critic proposal and the verified final agrees on every such
change. Apply those changes to the original JSON object. Preserve every
other field. An invalid shape, incomplete verification, discrepancy
without a supported proposal, or integration disagreement uses the
matched verified final. This policy is limited to the task's known
scalar JSON field schema. Agreement and quotations do not prove a
derivation, and the roles use the same model.

The experiment runs both live finals even when a route would return the
draft. The coordinated policy uses the verified output where needed
and makes no third final call. Its accounted final-generation requests
exclude the verified counterfactual only on direct return. The cap is 96
admissions for the fixed screen and 120 for the natural holdout. Any
final-call saving is relative to always regenerating after the same
full-field verifier. The counterfactual comparison does not establish
overall cost savings relative to unfiltered critique without that verifier.

## Readout

Score exact typed JSON per case and retain all matched outcomes. Report
incorrect drafts repaired and exact drafts degraded, alongside the paired
wins/losses against both comparators, route counts, field-level changes,
call counts and usage. For a given complete cohort with both exact and
wrong drafts, predeclare these comparisons against verified regeneration.

- Quality advantage requires more exact finals, at least as many wrong
  drafts repaired, and no increase in losses of initially exact drafts.
- Efficiency advantage requires the same exact final count, no paired
  loss of an exact final, at least as many repairs, no increase in losses
  of initially exact drafts, and fewer logical final-generation calls.

Report the fixed and natural results separately. A fixed-only qualification
describes the controlled screen. A natural qualification describes these
12 held-out questions and two samples each. Report any comparison against
unfiltered critique separately. A single cohort and a single evaluated
model provide descriptive effect estimates. There is no criterion-based
resampling.

Keep raw journals local and recompute the metrics with network access blocked.

    python sr-dual-brain-llm/scripts/benchmark_coordinated_critic.py fixed --live /private/path/fixed
    python sr-dual-brain-llm/scripts/benchmark_coordinated_critic.py natural --live /private/path/natural
    python sr-dual-brain-llm/scripts/benchmark_coordinated_critic.py natural --score /private/path/natural/journal.json
