# Fixed mixed critique and draft retention

Measure whether the existing claim gate keeps correct fixes and blocks
damaging suggestions when both occur in one critique. Measure the behavior
of returning the original draft without another generation when no
suggestion passes verification.

## Fixed design

- Use 12 new mixed cases across Boolean logic, SQLite, event state and
  source authority, three per domain. Each has one wrong draft field
  and one correct fix for it, plus one damaging proposal for a correct
  field. Add one correct-draft control per domain with one damaging
  proposal only. The fixture contains the manually fixed drafts and
  critiques. Oracle values were computed by enumeration, SQLite or
  deterministic simulation. The oracle and critique labels stay in
  evaluation code.
- Sample verification and integration twice per case. Keep all 32
  trials, including unavailable verification. Freeze the code,
  fixture, ordering seed 2714 and scoring before any provider call.
  In each block, launch each order of the two live final arms eight
  times. Use two trials at a time.
- Use the existing verifier, with standard gpt-6-luna effort,
  4,096 completion tokens and a 24-second owner deadline. The
  verification prompt contains the original question with line
  numbers and requested field IDs. A proposed value passes only
  when the fresh model independently derives the same typed value
  and supplies a nonempty derivation and citations that match exact
  question excerpts. Matching values and citations can still share
  a mistaken derivation.
- Make one verifier call and two live final generations per trial,
  for at most 96 admissions. The unfiltered final gets the entire
  frozen critique in the field-patch format. The verified regeneration
  gets accepted patches in the same format, or a neutral note if
  there are no accepted patches. Both finals have the same question,
  draft, system prompt and output limits. There is no draft-generation
  or critic-generation call in this experiment.
- The retention condition returns the exact original draft with zero
  final-generation admissions when no claims pass. This includes
  fail-closed verification. When any claim passes, it shares the
  verified regeneration result and its generation receipt. Sharing
  the sample keeps the conditions identical where the policy is
  identical, and never adds a third final generation. The benchmark
  still makes the verified regeneration call as the counterfactual
  on zero-pass trials.

## Readout

Report correct fixes admitted out of all known correct fixes and damaging
suggestions blocked out of all known damaging suggestions. Report the number
of mixed trials where both decisions are right. Include decisions missing
usable derivations and disagreements separately. Grade the two generated
finals and retention output by strict typed JSON against the frozen oracle.

Report actual zero-pass trials split by correct and incorrect draft, with
matched exact scores and generation calls avoided under retention. The
correct-draft controls provide a direct opportunity to see whether
rewriting introduces an error after complete rejection. Retaining a wrong
draft can forgo a repair from neutral regeneration. Report this separately
if an actual mixed trial has zero accepted proposals. Synthetic tests
exercise both paths regardless of the live outcome.

Raw journals are private. Recompute the scored outcomes from the journal
with network access blocked. This is a descriptive fixed-corpus screen of
the verifier and integrated output policies. It does not measure how well
a live critic proposes repairs.

    python sr-dual-brain-llm/scripts/benchmark_fixed_claims.py --live /private/path/new-run
    python sr-dual-brain-llm/scripts/benchmark_fixed_claims.py --score /private/path/new-run/journal.json
