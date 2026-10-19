# Scope and content in multilingual memory recall

Some questions share a project name but ask about unrelated facts. Other questions
refer to the same project using a Japanese name. Scope-aware recall separates a
declared identity from the question content; a name match alone is insufficient.
It is an indexing analogy, not a claim about biological memory.

Trusted callers can annotate a trace explicitly:

```python
memory.store(MemoryTrace(
    "How long does Project Helix retain audit logs?",
    "The recorded retention is 72 hours.",
    qid="original-record", seed_id="original-seed",
    retrieval_scope="helix", scope_aliases=("Helix", "ヘリックス"),
))
```

This record remains **unverified** unless a trusted caller separately verifies it.
Scope IDs/aliases do not certify answers. Model-produced QA records are not
automatically scoped or translated. No project-specific dictionary is built into
the selector. Existing records with no scope annotations keep their original
retrieval behavior when used without declared scopes.

Names use Unicode NFKC and case folding. Latin aliases have word boundaries;
Katakana aliases cannot match inside a longer Katakana name. Declared aliases
and generic project wrappers are removed from the content cue. Answers, drafts,
source IDs and verification labels do not enter embeddings.

A query must identify a declared scope to retrieve that scope's facts. A query
explicitly naming two unique scopes can retrieve both. An alias shared by multiple
scopes is ambiguous and cannot choose either. Missing or unknown scopes abstain
from scoped facts. A known-scope query does not borrow an unannotated global
record. This deliberate boundary can reduce recall; generic guidance and project
facts should not silently substitute for each other. Active metadata supplies the
alias registry: superseding, expiring, or evicting a trace removes its aliases if
no other available trace declares them. This is not a global deletion mechanism.

Local retrieval checks topic words after name removal. With the optional learned
index enabled as described in the [existing guide](question-memory-recall.md),
scoped pairs instead compare these content cues. The original raw-question default
stays 0.70. The new content threshold defaults to 0.50 and is configured separately:

```bash
export DUALBRAIN_MEMORY_SCOPE_MIN_SIMILARITY=0.50
```

The existing 0.65/0.35 semantic/lexical ranking, strong lexical override (0.75),
relative cutoff (0.8), cache/input limits and lexical outage fallback still apply.
Both defaults are experimental similarity cutoffs, not probabilities. A cache
stores question views keyed by their actual text; it does not attribute a clipped
or transformed vector to a different full question. Provider identity/dimension
checks still apply. Original inputs over 6,000 UTF-8 bytes cannot bypass the
embedding budget by removing aliases. Registry digests and the content threshold
are included in semantic/field state snapshots.

The controller shares the active registry before perception. Latent seed recall
checks scope even when its question hash would pass. Exact seed/source-QA IDs
identify the stored scope when question text is shared across projects; duplicate
identity bindings are rejected. Legacy seeds without a binding retain conservative
question-text resolution, which cannot choose between multiple possible scopes.
Hash-based latent triggers
still use the full question; they are not a complete topic-relevance gate. When
source IDs resolve to QA evidence, SharedMemory rechecks scope and content using
the retrieval selector. Both actual worker prompts receive the accepted scope ID
beside provenance/status. Scope metadata persists in the existing PostgreSQL JSONB
column, with empty defaults for legacy rows; no additional schema DDL is needed.
Independent hippocampal, working, and schema memory mechanisms retain their own
policies. These checks are not a global fact-validation or tenancy boundary.

The [fixed protocol](scoped-memory-protocol.md) defines the synthetic comparison:

```bash
python sr-dual-brain-llm/scripts/benchmark_scoped_memory.py \
  --output-dir target/benchmarks/scoped-memory-offline
python sr-dual-brain-llm/scripts/benchmark_scoped_memory.py \
  --semantic-model text-embedding-3-small \
  --output-dir target/benchmarks/scoped-memory-semantic
python sr-dual-brain-llm/scripts/benchmark_scoped_memory.py \
  --embedding-cache target/benchmarks/scoped-memory-semantic/embeddings.partial.json \
  --output-dir target/benchmarks/scoped-memory-replay
```

Only the second command makes paid requests. Its input includes both raw baseline
questions and content views, so its total is not a production per-turn cost.
Outputs are private under ignored `target/`. Frozen replay validates the complete
fixture/question-view set and makes zero provider calls. Baseline selectors ignore
the new scope annotations. Scoped selectors get additional declared information:
their gain is not attributable solely to the embedding model. Calibration and
evaluation scopes/sources differ, but some question-content templates repeat
between them; this tests scope handling, not generalization to unseen topics.
Retrieval is not a substitute for independently graded final-answer quality.
