# Optional question/evidence support checking

Enable this explicitly for the stdio engine or `run_server.py`:

```bash
export DUALBRAIN_MEMORY_ANSWERABILITY_MODEL=gpt-6-luna
```

This sends the current question and bounded historical Q&A to an OpenAI-compatible
chat endpoint. It uses `MEMORY_ANSWERABILITY_*` provider/key/base settings with the
existing `LLM_*` fallback. The flag selects the actual model; it does not reuse a
chat model silently. Without the flag, retrieval behavior and provider cost are
unchanged. No new dependencies or environment-wide activation are required.

Preparation runs asynchronously before controller perception, subject to the
existing recall policy. One request checks up to six candidates independently.
A record may support one explicitly requested part of a compound question; it
does not have to answer every part by itself. Supporting records must quote an
exact nonempty span of their own answer. Conflicting historical records remain
separate. Explicit absence can refute a question's premise; missing information
cannot supply an answer.

For trusted declared scopes, the pool expands to at most six available compatible
records, ranked by the existing lexical/optional semantic scores. Threshold and
relative-score rejection do not prune that pool: a second requested fact may
have a lower score than the first. The checker makes the support decision. For
unannotated memories the existing candidate policy still applies before checking.
Embeddings are optional and still receive questions only; the support checker
necessarily receives answers as well. Embedding threshold defaults are unchanged.

Synchronous retrieval reads approvals without network calls and returns up to
the requested limit (normally three). It does not apply the relative-score cutoff
again to approved records. Exact query, source text and identities, status,
timestamp, expiry, scope, backend endpoint/model identity and policy bind each
approval. The session-local LRU holds at most 512 decisions; it is not persisted.
Changed sources/models/policies require new approval. Expiry, supersession and
scope ambiguity are checked again independently of approvals.

Oversized source questions (over 240 characters) or answers (over 640 characters)
are withheld in full, rather than approving an excerpt with omitted conditions.
Current questions are limited to 6,000 UTF-8 bytes. This conservative source limit
is a practical recall limitation for long answers. The provider timeout is at
most 20 seconds and output is capped at 4,096 completion tokens, without automatic
continuations or retries. Missing/duplicate/malformed decisions, nonboolean flags
and ungrounded quotes create no approvals. Provider failures expose only their
exception type. Previously approved unchanged data can still be read from cache;
unchecked data cannot replace a failed check via lexical fallback.

Both direct context retrieval and seed-to-QA evidence resolution require approval.
Controller latent recall also receives an exact seed/source allowlist, preventing
a question hash from bypassing the check. Query-free `get_context()` and
`evidence_for_seeds()` return no QA when the feature is enabled. Ordinary dialogue
flow storage is separate. Existing downstream memory gates still suppress recall.
Source verification status is preserved: support checking neither verifies facts
nor establishes current applicability, final-answer correctness, or complete
coverage of all parts of a question. A quote is mechanically grounded text, not
a proof that the model interpreted it correctly.

The [experiment](memory-answerability-2026-10-19.md) measures this tradeoff. From
the repository root, explicitly paid development measurement is:

```bash
python sr-dual-brain-llm/scripts/benchmark_memory_answerability.py \
  --phase development --support-model gpt-6-luna \
  --embedding-model text-embedding-3-small \
  --output-dir target/benchmarks/support-development
```

Use `--phase evaluation` only after freezing the design. `--candidate-mode
shortlist` replays the first development design without scope expansion. Complete
private checkpoints enable zero-provider-call replay:

```bash
python sr-dual-brain-llm/scripts/benchmark_memory_answerability.py \
  --phase evaluation \
  --support-cache target/benchmarks/answerability-evaluation-2026-10-19/support.partial.json \
  --embedding-cache target/benchmarks/answerability-shortlist-dev-2026-10-19/embeddings.partial.json \
  --output-dir target/benchmarks/support-evaluation-replay
```

Output directories must be new. Private checkpoints are ignored and are not
distributed with the repository. Replay requires the complete matching fixture,
question views, policy, phase and exact support request set; a partial or corrupt
checkpoint cannot become a complete run. Raw provider errors are never printed.
