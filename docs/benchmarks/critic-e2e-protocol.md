# Paired Luna draft-to-final benchmark

Measure whether supplying a critic's issues helps repair incorrect drafts
or damages correct drafts. Compare two final generations from the same
natural draft. Use the repo's native draft, critic and integration methods
with the controller policy held out of this experiment.

## Freeze

- Use 12 new authored tasks, three each in Boolean logic, SQLite, event state
  and evidence authority. Task inputs contain every rule and datum needed.
  Oracles come from Boolean enumeration, actual SQLite execution and rule
  simulation. The fixture records the oracle and its derivation. Grade only
  with that fixture and never include its answers in provider prompts.
- Sample two fresh drafts per task, in two shuffled blocks. Keep every
  attempted draft, including incorrect, invalid and empty drafts. Use one
  shared draft per pair. Run the critic on that draft with standard provider
  effort. Both finals receive its exact text and the same question and system
  instructions. The control receives the neutral note "No collaborator review
  was supplied." The feedback arm receives the native critique summary when
  a valid external response reports issues. An ok or unavailable review
  sends the same neutral note in both arms and stays in the denominator.
- Use gpt-6-luna through OpenAI Chat Completions throughout. Omit the effort
  parameter for draft, critic and both finals. Use 4,096 output tokens
  including reasoning and a 24-second owner deadline per call. Use identical
  settings for both final generations. The feedback arm incurs one additional
  critic call. Do not run preview, micro fallback, JSON repair, retries,
  continuation, verification rounds or work products. The native integration
  can preserve the original draft if its own generation fails. Track stage
  completion and delivered answer completion independently.
- Freeze the runner, oracle and grading before calls. Use two sequential
  repetition blocks with shuffled task order from seed 2712, reversing the
  two final calls' launch order per task in the second block. Execute two
  paired trials per wave, with the two finals launched together after each
  trial's critic call. Up to 96 admissions, one per stage. A request contract,
  authentication or model error stops the run at the end of that wave.
  Keep raw request/response and terminal receipts locally with owner-only
  permissions. No rerolls.

## Scoring

Use terminally complete answers that contain just one JSON object. The keys
must match exactly without duplicates and the values must be integers, strings
or null. Compare each field by exact type and value. For example, an integer
in place of SQL null is a wrong answer. Count answers with wrong fields,
malformed answers and incomplete answers separately. For each arm report:

- Valid wrong draft to exact final, out of all valid wrong drafts.
- Exact draft to valid wrong final and exact draft to any nonexact final, out
  of all exact drafts. Report format and completion losses separately.
- Recovery from invalid or incomplete drafts, along with exact final grade
  across all attempted drafts.
- Paired wins, losses and ties by exact final score, including the subset
  where critique issues reached the final call. Preserve all graded rows,
  usage missingness and critic validity. Recompute scores from private
  receipts without provider access.

The second generation can fix a draft on its own. The incremental contrast
is the paired gap between the two finals. Cases in which both finals receive
the neutral note still capture independent sampling variation. Shared drafts
and matched final budgets control these parts of the setup. A critic call
changes offered total work. Two samples per authored task support descriptive
case-level findings.

Run the experiment from a clean frozen commit and recompute scores:

    python sr-dual-brain-llm/scripts/benchmark_critic_e2e.py --live /private/path/new-run
    python sr-dual-brain-llm/scripts/benchmark_critic_e2e.py --score /private/path/new-run/journal.json
