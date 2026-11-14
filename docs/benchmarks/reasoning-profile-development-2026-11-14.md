# Ordinary reasoning profiles: stopped development screen (2026-11-14)

This is a **stopped, one-case observation**, not a five-case ranking or a
successful holdout. The source and prospective protocol were frozen at
`98f77d35bd4278c77952af0a57d3b470d8b4a53b`. The public five-case
fixture, exact graders, arm order, caps, 96-admission ceiling and stop rule are
in [the protocol](reasoning-profile-development-protocol.md) and
[manifest](reasoning-profile-development-manifest.json). The run requested
`gpt-6-luna` through the OpenAI-compatible provider with a verified local CA
bundle. No new provider calls followed the transport stop.

The first Boolean case completed all five arm rows, then the study stopped.
The first `self_review` request raised `ClientOSError` before a provider usage
receipt. The remaining four cases and 20 arm rows are **unmeasured**.

| Arm | Complete answer | Exact | Item score | Workflow valid | Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `default_off` | 0/1 | 0/1 | 0 | 0/1 | 3 |
| `default_aware` | 0/1 | 0/1 | 0 | 0/1 | 2 |
| `medium_aware` | 0/1 | 0/1 | 0 | 0/1 | 2 |
| `low_aware` | 1/1 | 0/1 | 2/3 | 0/1 | 2 |
| `self_review` | 0/1 | 0/1 | 0 | 0/1 | 1, failed |

The baseline and medium draft/integration stages returned no visible answer
within their 4096-token caps. The low draft returned a schema-valid answer,
but it was semantically wrong on one of three items. Its 2048-token external
critic returned an empty response, so no valid review or workflow resolution
occurred. One task cannot show that low effort is better; an earlier numeric
error under low effort is adverse context. No arm qualified for promotion or
for the protocol's untouched holdout gate.

The journal records ten admissions, 16,530 known input tokens and 28,190
known output tokens. One failed call lacks usage; full token totals and cost
are **unknown**. The recorded serialized efforts and caps matched their
assigned arms. The live canonical digest is
`327edbafa010b97c568e14f990eb04f327e6c65f20a542cf2057cd82c7f7aebb`.

One network-forbidden replay used the frozen journal and reproduced all ten
admissions, 20 transport events, phase receipt, usage and semantic grades.
It **failed exact replay**: its final row stored the mapped exception name
`OSError` where live stored `ClientOSError`. That one row field changed the
canonical digest. No second replay or tape adjustment was made. The earlier
frontload screen's replay also remains failed. The separate next screen will
use fresh questions and omit the self-review reference that caused both
transport stops; neither stopped screen supplies a complete ranking.

Private local receipts (provider payloads and outputs are not published):

| Artifact | SHA-256 |
| --- | --- |
| live `report.json` | `192d11b35aa9903bfe10f79f64992b8ccfca9a3d9fa21b32c0b1b6efd1d75d97` |
| live `journal.json` | `1c0150eec419385320f02babd02ca3711f0f13e4d8192cdfc7c2384507e85858` |
| failed replay `report.partial.json` | `ba873d557c84af0d35cbc548792cd505ad23fcdc234454e2f8c89025cbb213aa` |
| failed replay `failure.json` | `3c1222d95add4c543c9507d17c3fbe7fbc30fdae4a6f8b0ff36887b1626cd0cf` |
