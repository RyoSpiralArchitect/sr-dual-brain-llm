# Controlled conversation-memory pilot — 2026-10-21

Caller-assisted conversation memory completed **16/16 evaluation probes**, versus
**5/16** with the previous support checker and **4/16** with memory off. It retains
explicit corrections and late conditions, resolves a narrow class of follow-ups,
and keeps generated answers from becoming additional primary sources. This is a
small authored final-answer pilot with model grading, not a production reliability
estimate. The feature remains opt-in: [usage](../conversation-memory.md),
[frozen protocol](conversation-memory-protocol.md),
[measurements and artifact hashes](conversation-memory-2026-10-21.metrics.json).

## Observed weaknesses and repairs

In development, the previous checker initially answered Aurora's corrected deadline
as seven days, then returned the old fourteen-day value on a follow-up and an exact
repeat. Checking whether an answer supports a requested relation does not decide
which source an explicit correction replaces. Ordinary controller writes also had
no declared project metadata, and generated answers became fresh candidates.

The new optional `memory_context` supplies a trusted scope, aliases, explicit write
intent and exact prior source IDs to supersede. Source answers stay unverified.
Corrections replace only uniquely identified records within that scope; the new
answer must be stored first. Answer-only turns remain in history but are excluded
from factual retrieval. A unique registered focus can bind conservative follow-up
phrases. Comparisons, unknown projects and unrelated turns clear that focus.
This is caller-assisted ingestion, not automatic natural-language fact extraction.

The long source answers were 815–822 Japanese characters; their source questions
also exceeded the old 240-character limit. The old checker withheld them in full.
Conversation mode checks complete source questions/answers up to 2,048 characters
each. Both right-worker routes preserve complete bounded packets, including the
late regional exception. Larger or sanitizer-modified units are still withheld.
No document chunking or arbitrary long-document recall is claimed.

## Final-answer results

Four authored Japanese episodes each contain three real model source turns and
eight probes. Development uses deadlines and parcel weights; evaluation uses
retention and concurrency. Names and values differ, but dialogue templates repeat.
Evaluation therefore has two episodes and four underlying requested factual values,
with correlated repeats/follow-ups plus unknown and ambiguous questions.

All arms run the actual controller/LeftBrainModel generation and storage path with
a fixed left lead. Working-memory, hippocampal factual hints, System2, executive
models and remote right consultation are disabled. Source writes receive no recall;
identical source requests share frozen real completions, with matching source QA
asserted. This controls source rewrites while comparing the bundled memory changes.
It does not isolate the causal contribution of each change or test the full default
multi-brain configuration. Probe answers are independently generated per arm.

| Split / arm | Task + grounding pass | Grader grounding failures | Available / graded | Support calls |
| --- | ---: | ---: | ---: | ---: |
| Development / off | 4/16 | 0/16 | 16/16 | 0 |
| Development / previous checking | 4/16 | 8/16 | 16/16 | 16 |
| Development / conversation | 16/16 | 0/16 | 16/16 | 10 |
| Evaluation / off | 4/16 | 0/16 | 16/16 | 0 |
| Evaluation / previous checking | 5/16 | 7/16 | 16/16 | 16 |
| Evaluation / conversation | 16/16 | 0/16 | 16/16 | 10 |

Correct unknown-fee admissions and clarification requests count as successful tasks
when requested by the rubric. Missing known values fail task without necessarily
failing grounding. The conversation arm handles corrections, anaphoric follow-ups,
exact repeats, late exceptions, missing fees, comparisons and post-comparison
ambiguity in both episodes. No provider or final grader parsing failures occurred.

Generation, checking and the arm-blinded shuffled-label rubric grader all used
`gpt-6-luna`, served under that name. This is same-family model grading, without
independent human labels. The first development grader conflated missing recall
with unsupported factual assertions and rejected one valid uncertainty statement.
Its original scores/journal are retained as a diagnostic in the measurements.
The clarified rubric regraded the same saved answers once; no answers were
regenerated. Both grading costs are included. Implementation and rubric were then
frozen before any evaluation response; no evaluation-driven tuning was performed.

## Cost and latency

The complete experiment used **208 chat calls, 113,180 input tokens and 30,885
completion tokens**, within the 256-call ceiling, including the initial flawed
development grading and its replacement. There were no embedding calls, automatic
retries or continuations. Requests allowed at most 4,096 completion tokens.

Support calls fall **16 → 10 per split (37.5%)**: four exact-repeat probes reuse
approvals and two ambiguous probes have no candidates. Repeats alone fall 4 → 0.
This is exact source/query/model/policy caching, not approximate semantic skipping.
The evaluation conversation arm's four cached preparations had a median **0.44 ms**.
Its ten uncached preparations had median **1.590 s**, nearest-rank p95 **2.417 s**;
the previous checker's sixteen had median **2.554 s**, p95 **7.264 s**.
These small sequential measurements include local overhead and differ in candidate
content; they are not a controlled service-latency or speedup guarantee.

Fewer calls did **not** reduce support input tokens: development rose **7,925 →
10,122 (+27.7%)**, evaluation **8,173 → 10,229 (+25.2%)**, because complete long
records now reach the checker. No currency savings are claimed. Source generation,
probe generation and grading costs are broken down in the measurements.

## Validation and provenance

Baseline generation, repaired development trajectories, calibrated development
grading and complete evaluation generation/grading replayed from private exact HTTP
journals with **zero provider calls**. Answers, stored source metadata and grades
match; replay latency and actual network-call counters naturally differ. The
development trajectory replay uses the archived initial grader instructions only
to reproduce its diagnostic scores; final quality uses the calibrated journal.
Private request/response journals and answers remain ignored; their hashes are
published. Previous shortlist, expanded, evaluation and 45-query precision fixtures
also replayed with unchanged source selections and aggregate summaries, without
chat or embedding calls.

The fixture/protocol were committed before paid generation (`fac969d`), the baseline
runner at `64a6007`, conversation generation at `7d5e1d7`, and frozen evaluation
implementation/rubric at `4a2954d`. The later focus/grade validation changes were
confirmed by exact development replay. Final local validation: **502 Python unit
tests, 5 gateway contract tests and 13 Node tests passed**. Coverage includes normal
and streaming RPC metadata, failed persistence, atomic historical status changes,
rolling-buffer writes, legacy origin defaults and actual prompts in both workers.

PR review found two production-path gaps beyond the controlled benchmark: pending
correction targets could still enter a source-write draft, and working/hippocampal
summaries could reintroduce excluded QA. Pending targets are now hidden before
drafting without changing persisted status early. Conversation mode delivers
historical facts only through the SharedMemory source route; raw working-memory,
hippocampal/replay, schema and pitfall text is suppressed, including director `keep`
overrides. Components retain indexing/control roles. Cortical line filtering also
preserves full approved QA units. Three additional regressions include real
EngineSession components with a simulated model endpoint, pending/cancelled writes,
and late-condition preservation. Frozen answer/grading journals are replayed after
these review repairs without more paid evaluation or rubric changes. Full engine
answer quality remains unmeasured; the engine regression tests prompt delivery.
Follow-up review also added rejection of stale/expired correction targets and
scoped the flag to the metadata-capable stdio engine. The interactive server
retains its existing memory policy. Regression tests cover both cases.
Benchmark generation failures now preserve an unavailable row/checkpoint and abort
before any subsequent turn can consume a locally stored fallback. Source/probe
failure regressions cover this path. The live measurements had no such failures.

Paid generation commands (run from repository root, each output directory new):

```bash
python sr-dual-brain-llm/scripts/benchmark_conversation_memory.py \
  --phase development --arms off checked --model gpt-6-luna \
  --output-dir target/benchmarks/conversation-baseline-2026-10-21
python sr-dual-brain-llm/scripts/benchmark_conversation_memory.py \
  --phase development --arms conversation --model gpt-6-luna --grade \
  --seed-cache target/benchmarks/conversation-baseline-2026-10-21/journal.partial.json \
  --baseline-report target/benchmarks/conversation-baseline-2026-10-21/report.json \
  --output-dir target/benchmarks/conversation-development-2026-10-21
python sr-dual-brain-llm/scripts/benchmark_conversation_memory.py \
  --phase development --grade-only --grade --model gpt-6-luna \
  --baseline-report target/benchmarks/conversation-development-2026-10-21/report.json \
  --output-dir target/benchmarks/conversation-development-regraded-2026-10-21
python sr-dual-brain-llm/scripts/benchmark_conversation_memory.py \
  --phase evaluation --arms off checked conversation --grade --model gpt-6-luna \
  --output-dir target/benchmarks/conversation-evaluation-2026-10-21
```

To replay evaluation, replace `--model gpt-6-luna` with
`--journal target/benchmarks/conversation-evaluation-2026-10-21/journal.partial.json`
and select a fresh output directory. The grader amendment is already included in
the current runner; historical diagnostic grading requires its archived prompt.

The six-candidate/three-delivered-source limits remain. Legacy rows without origin
metadata remain primary by default; enabling this mode on historical data needs
curation or a fresh session. Session focus is not restored after restart. Tests use
simulated PostgreSQL transactions; no live database behavior, adversarial source
robustness, unrestricted pronoun understanding or factual truth is established.
