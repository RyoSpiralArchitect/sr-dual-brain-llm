# Complete context through attention and memory relay

Prefrontal attention selects complete context streams before the thalamus relays
them to drafting and consultation. It no longer edits factual content one line
at a time. The default stdio EngineSession uses this path; no additional flag,
dependency or model is required.

The previous gate could turn this retrieved record:

```text
Q:Oriole parcel limit? A:5kg.
Europe:2kg.
```

into its first line alone. At the default threshold its query overlap was enough
to activate filtering, but the late regional exception did not share query words.
The new attention operation either retains the complete stream or suppresses that
stream. Q/A lines and their qualifications stay together.

## Responsibilities

| Component | Behavior |
| --- | --- |
| Prefrontal working memory | Keep bounded recent traces. Mark entries whose question or answer exceeded the field limit and withhold the entire entry from prompt recall. |
| Prefrontal attention | Score each supplied stream by its strongest query anchor and select whole streams. Preserve the existing low-signal fallback. |
| Thalamus | In `working_only` mode, suppress other memory only when working memory has sufficient supplied relevance. Otherwise retain the long-term fallback. |
| Controller | Use the same context assembly/attention path initially, after relay and after director memory control. Send the resulting complete context to both workers. |

`gate_context_parts(parts, focus, protected=...)` accepts independently selectable
streams, such as working memory, SharedMemory and hippocampal summaries. A stream
may contain several records: this does not perform per-fact segmentation or
arbitrary document chunking. It preserves the text as received; upstream memory
sanitization and hippocampal summarization retain their existing behavior.
`gate_context(context, focus)` now retains opaque text intact. Callers needing
selection must supply real stream boundaries rather than infer them from newlines
or user-supplied `Q:`/`A:` text.

The strongest line overlap supplies a stream's selection score; it is not a
probability of factual correctness. The existing aggregate focus still controls
whether prefrontal selection runs. SharedMemory returned by an answerability
checker is protected from lexical reclassification, as is the dedicated source
stream in conversation mode. Explicit director drops, thalamic suppression,
source expiry/supersession, scope matching and support approvals retain their
roles. Conversation mode continues to exclude raw working, hippocampal, replay,
schema and pitfall facts; this change does not reopen those routes.

Thalamus accepts optional `context_relevance` scores. With scores, a working-memory
score below the configurable `working_relevance_threshold` (default 0.125) keeps
the long-term fallback available, even when working memory is nonempty. Explicit
`suppress_long_term` still suppresses it. Callers omitting scores keep the previous
relay behavior. Fallback notes identify `working_memory_low_relevance`.

Clipped working traces remain bounded bookkeeping but are not delivered as factual
prefixes. The requested recent-turn window is applied before filtering, so an
incomplete recent turn does not silently fetch more distant turns. Complete older
entries within that window can still be delivered. Long source facts may be
available through SharedMemory; otherwise recall can be incomplete. The existing
memory-text sanitizer is unchanged.

## Verification and practical limits

`tests/test_cortical_attention.py` covers late exceptions, whole-stream selection,
approved evidence without lexical overlap, incomplete working-memory fields,
director drops and relevance-aware relay decisions. An actual EngineSession with
its perception components and real right-worker queue captures both model prompts
using simulated clients; the same late qualification reaches both workers.
`prefrontal_context_attention` telemetry reports stream scores and retained stream
names without recording their text.

Local validation: **655 model-free Python tests and 5 gateway tests passed**;
the ten isolated PostgreSQL cases run separately in CI. This verifies prompt
delivery and control contracts, not improved live final-answer scores. The earlier
controlled memory quality benchmark disabled prefrontal attention and therefore
cannot establish this change's quality impact.

Retaining conditions can increase context size, and changed context/focus can
affect downstream consultation decisions. No token or latency saving is claimed.
The next quality measurement should enable these brain components, compare a
fixed previous implementation against this one, and freeze disjoint conditional
reasoning tasks before live generation. Model weights, general semantic recall,
source truth verification and long-document capacity are unchanged.
