# Question-centered recall and memory evidence

Recall now treats the question as the retrieval cue. The current draft remains an
input to archetype mapping, affect and symbolic processing. Newly created latent
seeds retain a question vector separately from their legacy question/draft vector.
Changing a draft therefore does not change an exact-question recall decision.
Legacy seed objects without a question vector retain the previous payload path;
`UnconsciousField(recall_mode="legacy_payload")` provides an explicit comparator.

This separation is an indexing analogy: the index finds a source; `SharedMemory`
owns the QA contents. An emerged seed never promotes its unresolved draft into a
fact. Exact seed IDs and source qids resolve an unambiguous stored final QA, and a
bounded evidence packet is included in the actual right-consult context. Both the
stdio and standalone right workers consume this packet for detail and critic
requests. The controller also renders it for its own fallback/integration path.
Packet inclusion demonstrates data flow, not internal model reliance.

## Evidence and policy

`MemoryTrace` adds `evidence_status` (`unverified`, `verified`, `superseded`) and
optional `valid_until` (epoch seconds). Existing/model-produced answers default to
`unverified`. Verification is supplied explicitly by a trusted caller, never
inferred from model confidence, a successful turn or text saying “verified”.
`timestamp` is the recording time, not the date when the content became valid.
Superseded and expired traces are excluded from SharedMemory retrieval and seed
resolution. Ambiguous seed IDs or source qids are not attributed.

Evidence includes at most three entries and 2,800 JSON characters, with each
question/answer capped at 240/640 characters. Truncation makes this a bounded
snippet, not a complete original document. Context labels entries as historical
data, marks truncated excerpts, and warns against assuming omitted qualifications.
It labels unverified past answers explicitly;
they are not confirmed current facts.
Explicit director memory drops, thalamic gating and prefrontal recall policy are
honored; an explicit director keep follows the existing director override order.
Seed identity alone does not override these gates.

PostgreSQL's idempotent schema setup adds an `evidence_metadata` JSONB column.
Seed IDs, status and expiry survive QA save/load; historical rows default to no
seed ID and unverified status. SQL adapter tests cover serialization and legacy
defaults; an unavailable optional pgvector extension is rolled back to a savepoint
so the required schema migration can commit. A real PostgreSQL server is not
exercised by these local unit tests.
Latent cache contents and learned vectors remain session-local. Independent
hippocampal/schema memories retain their existing retrieval/lifecycle behavior;
the new QA status filters do not constitute global deletion of their copies.

## Optional learned question index

The default remains local and requires no embedding API. To opt into OpenAI
question embeddings, use the existing shared LLM credentials or `MEMORY_*` role
configuration and set:

```bash
export DUALBRAIN_MEMORY_EMBEDDING_MODEL=text-embedding-3-small
export DUALBRAIN_MEMORY_MIN_SIMILARITY=0.70
```

This enables additional paid embedding requests in newly created stdio engine
sessions. The threshold is configurable and model dependent; 0.70 is a conservative
default, not a universally calibrated probability. Chat model settings and the
symbolic/archetype embedder remain separate. The standalone server consumes
evidence packets but does not automatically create the optional semantic index;
custom controllers can inject `QuestionSemanticIndex` into SharedMemory.
The [2026-10-16 pilot](question-memory-2026-10-16.md) shows the tradeoff explicitly:
the 0.70 default retrieves fewer relevant sources than lexical search on that
fixture, while a calibration-selected 0.45 improves recall but admits one
unsupported-source selection. The calibrated setting is not automatically enabled.

Before perception, the controller prepares the current question, latent-seed
questions and up to 256 recent available QA questions. Drafts and answers are not
sent to the embedding API. The cache is per session, limited to 512 questions,
and network work is asynchronous. A prepare call submits at most 64 missing
questions/48,000 UTF-8 bytes; individual questions over 6,000 bytes are skipped.
Unprepared or oversized queries fall back to lexical retrieval. Batch failures,
model/endpoint identity changes and vector-dimension drift do not silently mix
embedding spaces. Provider error logs contain exception types, not raw messages.

For prepared pairs, a semantic score meeting the threshold can add candidates;
rejected semantic pairs need a strong lexical score (0.75) to remain eligible.
Ranking uses the larger of the lexical score and `0.65 * semantic + 0.35 * lexical`.
The selected candidates must be within 80% of the best score, up to three items.
Missing vectors use the lexical path. This is a fixed experimental selection rule;
semantic similarity is not evidence that a memory is correct or current.

`memory_semantic_preparation` telemetry records preparation state, cache coverage,
embedding call/token deltas and elapsed time. `memory_evidence_routed` records the
source IDs provided to the consult stage. These extra costs belong in full-turn
latency/token accounting; a warmed lookup benchmark excludes network preparation.

## Competing-memory benchmark

```bash
python sr-dual-brain-llm/scripts/benchmark_memory_retrieval.py \
  --output-dir target/benchmarks/question-memory-offline
python sr-dual-brain-llm/scripts/benchmark_memory_retrieval.py \
  --semantic-model text-embedding-3-small \
  --output-dir target/benchmarks/question-memory-semantic
python sr-dual-brain-llm/scripts/benchmark_memory_retrieval.py \
  --embedding-cache target/benchmarks/question-memory-semantic/embeddings.partial.json \
  --output-dir target/benchmarks/question-memory-replay
```

The second command explicitly makes embedding API calls. Output directories must
be new; private reports/vector checkpoints are under ignored `target/`. The corpus
contains current/superseded/expired records, a similar log policy from another
project, same-project unrelated information, unsupported questions and conflicting
active records. English paraphrases and Japanese questions are included. Distinct
project scenes and relevant source IDs are separated between calibration and
evaluation; language variants within a scene are correlated.
Replay requires a complete cache matching both fixtures and the exact question
set. It makes no provider calls and is a reanalysis of the original vectors,
not an independent model run. Incomplete checkpoints are preserved but cannot
be presented as complete frozen experiments.

Comparators are the previous lexical selector, the current lexical selector with
status/function-word filtering, and optional hybrid selection. A fixed threshold
grid is selected solely by calibration macro set-F1, preferring the stricter value
on ties. Evaluation labels do not participate. The selected threshold is an
experimental result, not automatically applied to runtime settings.

The report keeps all queries and selected source IDs, top-1 hits, full-source
recall within three candidates, no-answer abstentions, unrelated selected slots
and macro set-F1. A separate three-scene ablation varies question wording and
drafts from an unchanged seed state. Those rewritten drafts were authored for
diagnosis and have not been independently judged semantically equivalent.

This benchmark measures retrieval/provenance, not final-answer factual coverage,
action quality or grounding. Those are distinct future evaluation axes; retrieval
hits must not be reported as repaired answers. No historical controlled-memory
ratings or fixture contents are changed by this workflow.

Run checks with injected model settings removed only from the child:

```bash
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python -m pytest -q --ignore=tests/test_gateway_contract.py
```
