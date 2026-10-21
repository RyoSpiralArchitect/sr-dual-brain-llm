# Live memory retest protocol — 2026-10-22 (Asia/Tokyo)

Freeze this protocol, runner, reference adapter and new fixture before any paid
calls. Rerun final-answer quality and measure actual support input/output tokens
after PRs #58–#60. Do not tune implementation, prompts, criteria or grading after
seeing either suite's live answers. Preserve failures and incomplete artifacts.

## Suites and comparison

1. **Historical evaluation retest:** the unchanged two evaluation episodes in
   `conversation_memory_benchmark.json`, with 3 source writes and 8 probes each.
   Compare memory off, the current non-conversation checker, conversation memory
   with the previous whole-candidate batching rule, and current incremental
   conversation memory. Previous conversation evaluation was already 16/16;
   this suite tests retention rather than providing score headroom. Differences
   from the October 21 run are not controlled causal effects: fresh sampling and
   possible service/model changes can contribute.
2. **New candidate-growth evaluation:** two prospectively authored episodes in
   `conversation_memory_cache_growth.json`, with 5 writes interleaved with 8 probes
   each. Compare whole-batch and incremental conversation memory. Exact queries
   recur after a long policy, an unrelated same-scope fact and an explicit source
   correction. Other probes combine facts/conditional exceptions, ask an unknown
   fee, compare two projects and require clarification after ambiguity. Domains
   are API-key lifetime and concurrency; exceptions require two simultaneous
   conditions. One policy is English with Japanese questions. This authored set
   is unseen by live generation, but its mechanisms deliberately target the cache
   change; it is not an independent generalization benchmark.

Each suite uses actual `DualBrainController`/`LeftBrainModel` generation and stores
the resulting QA. Source requests receive no recall and identical requests share
their real completions within a suite; every stored source answer hash must match
across arms. The reference adapter uses the production parser/limits but rechecks
the full candidate batch on a miss. It preserves old cache decisions on failure
and replaces them only after a successful complete batch, reproducing the rule
before #60. It makes real support requests, not synthetic subset responses.
All other runtime code is current main; this isolates the batching rule, not a
complete historical binary. The current non-conversation checker is also current
code, not a pinned October 21 baseline.

Keep the previous controlled configuration: fixed left lead, no System2, remote
right consultation, executive models, working-memory or hippocampal factual hints.
Conversation scope/remember/supersession metadata is caller supplied. This is not
a full default multi-brain quality test or automatic fact extraction. PostgreSQL
is not involved. Runs use fixed arm order, with whole-batch before incremental;
do not claim controlled latency or broad statistical significance from two episodes.

## Scoring and costs

Use the existing calibrated `JUDGE` and `TASK` verbatim with `gpt-6-luna` for
source generation, final answers, support checking and grading. Shuffle labeled
answers using the existing deterministic seed; graders see no arm identity or
retrieval diagnostics. Report task+grounding passes, grounding failures,
unavailable answers and grader failures with all denominators. This is exploratory
same-family model grading, not independent human grading. Candidate-growth probes
provide per-turn reference snapshots, so an early answer is graded against facts
known then, rather than a future correction. These references never enter source
or answer generation.

Record actual prompt/completion tokens by role and support usage by arm/turn,
records transmitted, cache hits, requested/served models, source hashes, latency,
fixture/policy hashes and clean code revision. Missing usage stays unknown, not
zero. Compare support input tokens directly; currency cost is not inferred from
token counts and no approximate byte/token conversion is used. Overall generation
and grading overhead is separate from support preparation.

Hard allocation: **144 live calls for historical retest + 112 for candidate-growth
= at most 256**, with at most 4,096 completion tokens per call. No continuations,
automatic retries, embeddings or favorable reruns. Frozen replay makes no calls.
Generation failure checkpoints the unavailable turn and aborts before later turns
consume a fallback; partial suites are reported as incomplete, not discarded.

```bash
python sr-dual-brain-llm/scripts/benchmark_conversation_memory.py \
  --phase evaluation --arms off checked conversation_whole_batch conversation \
  --grade --model gpt-6-luna --max-calls 144 \
  --output-dir target/benchmarks/live-memory-historical-2026-10-22
python sr-dual-brain-llm/scripts/benchmark_conversation_memory.py \
  --fixture sr-dual-brain-llm/examples/conversation_memory_cache_growth.json \
  --phase evaluation --arms conversation_whole_batch conversation \
  --grade --model gpt-6-luna --max-calls 112 \
  --output-dir target/benchmarks/live-memory-growth-2026-10-22
```

Keep raw source/answer/HTTP journals private and ignored. Publish aggregate and
per-probe decisions, usage, limitations and artifact hashes. Verify complete exact
HTTP replay without paid calls before finalizing the report. Any weakness found
becomes a separately measured follow-up; these evaluation responses are not used
to tune the scored implementation.
