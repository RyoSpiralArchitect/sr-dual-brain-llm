# Frontloaded generation: stopped development screen, 2026-11-14

This frozen prospective study at `d1b79f887d71d0e5418b69fb8d9c9959ba53cb11`
stopped after **7/80 provider admissions and four cells on only the first
Boolean task**. The self-review initial request raised an unclassified
`ClientOSError`. The sticky transport latch closed further paid work. There is
no five-task ranking, no candidate selected for holdout and no provider call
on the other four cases. The original inputs, caps and outcomes were not
rerun or tuned. See the [frozen protocol](frontload-development-protocol.md),
[fixture](../../sr-dual-brain-llm/examples/frontload_development.json) and
[manifest](frontload-development-manifest.json).

| First task arm | Terminal answer | Requests | Draft receipt | Other relevant receipt |
| --- | --- | ---: | --- | --- |
| Native 4096 baseline | Unavailable | 3 | 4096/4096 reasoning tokens; 0 visible characters | Integration also 4096/4096 reasoning, 0 visible characters |
| Frontload 8192 | Unavailable | 2 | 8192/8192 reasoning tokens; 0 visible characters | Ordinary integration had no remaining allowance |
| Frontload 8192 with completion-aware critic | Unavailable | 1 | 8192/8192 reasoning tokens; 0 visible characters | Incomplete draft was not sent to critic |
| Self-review reference | Unavailable | 1 | Transport failed before a response | No gradeable output |

The 8192-cap result is a negative observation for simple output-ceiling
frontloading on this task. Because the campaign stopped, it cannot establish
a complete comparative result, and the development selection rule was not
evaluated. All caps and the 12288 per-trial allowance were respected; no
admission was denied. Six requests returned envelopes and the seventh has
unknown usage. Known usage is at least 11,289 input and 25,380 output tokens;
full totals and billing are unknown.

The one strict network-free replay failed with `ValueError` before trial
execution: the frozen adapter did not recognize the recorded `ClientOSError`
class. This is retained as a failed replay, not a verified receipt. The
original live report SHA-256 is
`e486484288afc1f8fcf315fa8c795a8682123cfd612ebf61e4bd6c0962f3ffd0`,
the journal SHA-256 is
`8f36c332b3a0a3611d033dff00e4109c7471e66127746432f73556ba3cfcc30c`,
and the replay failure SHA-256 is
`3c1222d95add4c543c9507d17c3fbe7fbc30fdae4a6f8b0ff36887b1626cd0cf`.
The raw provider payloads and responses remain private in ignored samples;
the authored questions and evaluator references are tracked in the fixture.

A separate new-task screen may compare provider-default, medium and low
ordinary reasoning effort at the same 4096 stage cap. Previous low-effort
evidence includes a concrete numeric error, so an output-availability gain
alone would not justify changing defaults. The replay adapter can separately
gain a typed, recorded `ClientOSError` bridge for future studies; it cannot
retroactively turn this failed replay into a success.
