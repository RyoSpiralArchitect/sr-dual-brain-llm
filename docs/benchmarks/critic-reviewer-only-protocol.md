# Reviewer-only Luna benchmark

Measure whether the native external critic detects a single known error in
a draft and leaves a correct draft alone. This study tests the critic without
running draft generation, repair, integration, or deterministic micro checks.

## Freeze

- Freeze 12 authored tasks and their expected answers in
  [critic_reviewer_only.json](../../sr-dual-brain-llm/examples/critic_reviewer_only.json).
  The four domains are Boolean logic, SQLite, event state and evidence authority,
  with three tasks per domain. Each task has three answer fields. The paired
  drafts differ in one field, with four mutations per field position. Truth
  tables, SQLite execution and deterministic rule tracing verify the oracle.
- Send one draft at a time in a fresh prompt. Pass only that draft, the task,
  and the native critic instructions with the same field-witness addendum
  across all arms. Do not include the other draft or oracle metadata.
- Use OpenAI chat completions and requested model gpt-6-luna. Compare provider
  default (omitted effort), low and medium, all with 4096 completion tokens
  including reasoning and a 24-second owner window. Disable retry, repair,
  continuation and micro fallback. Allow one HTTP admission per cell and
  72 admissions total. Run the three efforts concurrently for each draft,
  rotating launch order and shuffling drafts with seed 2711. Complete one
  batch before the next. An auth, model, or request contract error stops
  the study with the attempted batches retained.
- Freeze the runner and fixture in Git before starting. Keep request and
  response receipts in a private directory. Report missing provider usage
  explicitly, including cancelled requests. Preserve every cell, including
  transport errors and deadline outcomes.

## Scoring

Each arm has 12 correct drafts and 12 single-error drafts. A hit requires a
valid issues verdict and a witness naming the mutated field, quoting its
supplied value, and giving the exact corrected value. Field labels are
case-insensitive and surrounding whitespace is ignored. Track unmatched
issue strings on wrong drafts separately. A wrong draft with a valid ok
is a false ok. A wrong draft with valid issues and no matching witness
is off target. Any valid issues verdict on a correct draft is a false
issue. An unavailable review contributes to the all-attempted miss count
for wrong drafts and is reported separately from both types of false verdict.
Successful detection does not score the quality of the fix explanation.

Report per-arm counts and all rows, including completion/transport status,
elapsed time, and available usage. The exact witness requirement can miss
semantically correct reviews that violate the requested format. A separate
manual audit may report those as such while preserving the original score.

This is a bounded development screen with one sample per cell. It is
descriptive and uses no automatic winner or default change. Recompute
metrics from the stored terminal decisions and critic outcomes without
provider access. This verifies scoring rather than the earlier full-turn
replay at the deadline boundary.

Run the frozen benchmark and recompute the metrics with:

```sh
python sr-dual-brain-llm/scripts/benchmark_critic_reviewer_only.py --live /private/path/new-run
python sr-dual-brain-llm/scripts/benchmark_critic_reviewer_only.py --score /private/path/new-run/journal.json
```
