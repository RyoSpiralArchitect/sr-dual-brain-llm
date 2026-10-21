# Caller-assisted conversation memory

The stdio engine supports opt-in, explicit project memory ingestion:

```bash
export DUALBRAIN_MEMORY_ANSWERABILITY_MODEL=gpt-6-luna
export DUALBRAIN_MEMORY_CONVERSATION_CONTEXT=1
```

The support flag sends historical Q&A to the configured provider and incurs cost.
The conversation flag changes which answers become retrievable primary records.
Both are disabled by default. No automatic scope or correction extraction is
provided. Callers using the controller directly can construct
`SharedMemory(conversation_context=True, answerability_gate=...)`.

Supply `memory_context` in `process` or `process_stream` params (or as a controller
keyword), together with a unique `qid` for each turn:

```json
{
  "qid": "deadline-update",
  "question": "Project Alpha's deadline is now seven days. Repeat and remember it.",
  "memory_context": {
    "id": "alpha",
    "aliases": ["Alpha", "アルファ"],
    "remember": true,
    "supersedes": ["prior-deadline-source"]
  }
}
```

Scope metadata is trusted application input, not model output. `remember: true`
declares a primary source write; ordinary answers remain in QA history but are
excluded from factual retrieval in this mode. Source answers remain unverified.
Only explicit, unique source IDs in the same scope can be superseded. No records
are superseded until the new answer is stored. PostgreSQL persists the new source
and old status changes in one transaction; legacy rows default to primary,
unverified records. The in-session focus is deliberately not restored on restart.
Tests simulate PostgreSQL transactions; this change has no live database result.
Existing rows without origin metadata retain their legacy primary-source default.
Use a fresh session or explicitly curate historical records when enabling this
mode; it cannot identify previously saved answers that were derived from recall.

An explicit project mention sets a single focus. Conservative follow-ups starting
with phrases such as `その`, `前の件`, `that` or `the previous` inherit that focus
only while its alias remains uniquely registered. A comparison, unrelated turn,
unknown project or ambiguous alias clears focus. This is a small phrase resolver,
not general pronoun understanding. The resolved cue is shared by support checking,
direct retrieval and latent seed relevance. The user question is preserved.

Complete source questions and answers up to 2,048 characters each can be checked.
Larger or sanitizer-modified records are withheld entirely. Both right-worker
paths accept an explicit `complete_v1` packet bounded to three entries and 16,000
characters; the default packet retains its previous limits. Conditions at the end
of a record are retained. Arbitrary document segmentation is not implemented.

Exact repeated query/source/model/policy combinations reuse the session support
cache. Generated probe answers cannot create fresh primary candidates and force
rechecking. Similar but different questions still require their own check.
These controls do not prove the factual correctness of stored model answers or
prevent every unsupported final answer.

See the [controlled evaluation protocol](benchmarks/conversation-memory-protocol.md).
