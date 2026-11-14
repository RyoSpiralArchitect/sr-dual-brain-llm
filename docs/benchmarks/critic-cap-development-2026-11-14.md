# Critic capacity: complete development screen (2026-11-14)

This is an **adaptive development result**, selected after the preceding
[stopped reasoning-profile screen](reasoning-profile-development-2026-11-14.md).
Source, fresh five-case fixture, exact graders, arm order, budgets and gate
were frozen at `f9507dde60672ba9c99b5d11336012cbd30ad22c` before this
run. See the [protocol](critic-cap-development-protocol.md) and
[manifest](critic-cap-development-manifest.json). The requested model was
`gpt-6-luna` through the OpenAI-compatible provider.

All **20/20 cells completed**. The two Boolean tasks remained difficult:
provider-default ordinary generation emitted no complete answer, while both
low-effort conditions emitted complete but incorrect answers. The other three
domains were scored with their unchanged exact graders.

| Arm | Exact | Complete | Mean item accuracy | Workflow valid | Valid external reviews | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `default_off` (primary baseline) | 2/5 | 3/5 | 0.575 | 2/5 | 4/5 | 12 |
| `default_aware` | 2/5 | 3/5 | 0.571 | 2/5 | 2/5 | 10 |
| `low_aware` (critic 2048) | 3/5 | 5/5 | 0.683 | 2/5 | 2/5 | 13 |
| `low_critic4096` | 3/5 | 5/5 | 0.683 | 3/5 | 3/5 | 10 |

The first Boolean cases show that increasing the critic cap does **not**
guarantee a review: the 4096-token critic was cancelled at about 24 seconds
on both. SQL, event replay and evidence selection were exact for
`low_critic4096`; the baseline was exact on SQL and evidence, but not the
event case. `low_aware` was also exact on all three. Thus the observed
one-case exact uplift over the primary baseline cannot be attributed to the
critic cap. The cap arm did gain one valid review and one workflow-valid case
over `low_aware` in this independent-sample screen.

The **precommitted development gate passed**: `low_critic4096` had one more
exact case than `default_off`, higher mean item accuracy, no item-score loss
on SQL/events/evidence, at least as many workflow-valid cases as `low_aware`,
and one more valid external review. This only authorizes a separately frozen,
untouched holdout. It does not change product defaults or establish a causal
or statistically reliable advantage.

The journal has 45 conservative admissions, 90 transport events and no
denials. Four requests were owner-cancelled without usage receipts. The
known subset used 105,151 input and 76,150 output tokens; full totals and
cost remain **unknown**. Serialized request caps and ordinary effort matched
the assigned arms. A single network-forbidden replay reproduced all 20
canonical rows, grades, admissions, usage, reservations, transport events and
actual owner cancellations. It made zero blocked HTTP attempts and matched
the live canonical digest
`1b3b9e62e153610d848c2a8805c5fcf1827460637f893b7a218a52ca32797a50`.

Private local receipts (provider payloads and outputs are not published):

| Artifact | SHA-256 |
| --- | --- |
| live `report.json` | `b3b5ee9135d1a56d1a799e0c56ec53d79c501302f72576a221d20f9e664fe3b0` |
| live `journal.json` | `8ca16cc3394246aaaeb47d5c24d59e13c0d6074ea15eb4157a33969ac74e209d` |
| replay `report.json` | `855736ede9a24ae514c94bb3d2837db95b14437659771bcc950532b7143f3cba` |
| replay `journal.json` | `8ca16cc3394246aaaeb47d5c24d59e13c0d6074ea15eb4157a33969ac74e209d` |
