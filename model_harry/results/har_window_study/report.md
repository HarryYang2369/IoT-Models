# Activity (HAR) window study

Official 10-fold CV on datasetJan2026, harry_cnn, final HAR recipe, seed 0. Latency = app replay (32-sample packets = 2.56 s, 0.5 s step): seconds from the change until the new class is shown and kept 4 s; *raw* = smoothing off, *smoothed* = the app default (majority of the last 3 results).

| candidate | window | static | dynamic | weakest dynamic | folds | raw median (Jan / ours) | smoothed median (Jan / ours) | smoothed mean (Jan / ours) | smoothed p90 (Jan / ours) | miss % (ours) |
|---|---|---|---|---|---|---|---|---|---|---|
| w50 | 4 s | 99.4 | 90.7 | 83.6 | 10 | 5.0 / 5.0 | 5.6 / 5.6 | 5.0 / 5.0 | 5.6 / 5.6 | 0 |
| w38 | 3 s | 99.4 | 89.8 | 80.3 | 10 | 4.3 / 4.3 | 5.0 / 5.0 | 4.6 / 4.5 | 5.0 / 5.0 | 0 |
| w25 | 2 s | 99.4 | 88.1 | 75.5 | 10 | 3.0 / 3.0 | 3.7 / 3.7 | 3.7 / 3.5 | 3.7 / 3.7 | 0 |
| w50rec | 4 s + recency | 99.4 | 89.0 | 74.4 | 10 | 3.0 / 3.0 | 3.0 / 3.0 | 3.4 / 3.3 | 4.3 / 3.7 | 0 |

Per-class recall (%):

| candidate | sittingStanding | lyingBack | lyingLeft | lyingRight | lyingStomach | normalWalking | shuffleWalking | ascending | descending | running | miscMovement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w50 | 97.4 | 100.0 | 100.0 | 100.0 | 99.8 | 88.9 | 87.6 | 93.8 | 92.4 | 97.7 | 83.6 |
| w38 | 97.3 | 100.0 | 100.0 | 100.0 | 99.8 | 88.5 | 87.1 | 93.1 | 92.5 | 97.2 | 80.3 |
| w25 | 97.3 | 100.0 | 100.0 | 100.0 | 99.6 | 87.3 | 87.4 | 91.2 | 91.2 | 95.9 | 75.5 |
| w50rec | 97.3 | 100.0 | 100.0 | 100.0 | 99.7 | 89.8 | 87.9 | 92.2 | 92.7 | 97.3 | 74.4 |

Smoothed median latency on our recordings by kind of change (s):

| candidate | dynamic->dynamic | dynamic->static | static->dynamic | static->static |
|---|---|---|---|---|
| w50 | 4.3 | 5.6 | 5.6 | 5.6 |
| w38 | 3.7 | 5.0 | 5.0 | 5.0 |
| w25 | 3.0 | 3.7 | 3.7 | 3.7 |
| w50rec | 3.7 | 3.0 | 3.0 | 3.0 |

Pass marks: static mean >= 90 %, dynamic mean >= 85 %. Our recordings were never used for training here, so their latency is an unseen-person test.
