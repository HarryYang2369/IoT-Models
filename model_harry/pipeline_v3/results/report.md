# Pipeline v3 – faster models (trained to favour the most recent seconds)

Generated 2026-11-13 10:37. Official 10-fold CV, per-class recall averaged over the folds (CW3 marking), mean over seeds (± = seed sd). **As marked** = datasetJan2026 test windows only. *v3* = the new models; *v2 design* = the deployed design (4 s activity / 6 s social, no recency) with the same code, seed 0 only, as the speed reference. Recipes: *mixed* = members' recordings in the training folds (how the final model is trained); *jan* = datasetJan2026 only (strict).

## Activity (CNN, 4 s, recency)

| config | seeds | folds | static | dynamic | pass (≥90 / ≥85) | sittingStanding | lyingBack | lyingLeft | lyingRight | lyingStomach | normalWalking | shuffleWalking | ascending | descending | running | miscMovement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 mixed | 2 | 10 | 99.4 | 89.5 | ✅ | 97.4 ± 0.1 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.7 ± 0.1 | 90.3 ± 0.6 | 88.5 ± 0.1 | 92.6 ± 0.7 | 92.9 ± 0.5 | 96.7 ± 0.1 | 76.1 ± 1.0 |
| v3 jan | 2 | 5 | 99.6 | 88.8 | ✅ | 98.6 ± 1.8 | 100.0 ± 0.0 | 99.9 ± 0.1 | 100.0 ± 0.0 | 99.3 ± 0.3 | 88.5 ± 1.5 | 89.6 ± 1.5 | 91.1 ± 2.0 | 93.3 ± 0.5 | 96.5 ± 0.6 | 74.1 ± 0.6 |
| v2 design mixed | 1 | 10 | 99.4 | 90.5 | ✅ | 97.2 | 100.0 | 100.0 | 100.0 | 99.7 | 89.3 | 87.9 | 93.5 | 92.6 | 96.9 | 82.7 |

Each member as an unseen person (their own fold; mean over seeds):

| config | person | static | dynamic |
|---|---|---|---|
| v3 mixed | s2503795 | 99.8 | 93.5 |
| v3 mixed | s2952925 | 99.5 | 97.5 |
| v3 jan | s2503795 | 91.6 | 95.2 |
| v3 jan | s2952925 | 75.6 | 98.7 |
| v2 design mixed | s2503795 | 99.3 | 100.0 |
| v2 design mixed | s2952925 | 99.8 | 97.6 |

Speed – app replay (2.56 s packets, a result every 0.5 s), median seconds from a change until the new class is shown and kept 4 s (seed 0):

| config | Jan2026 raw | Jan2026 smoothed | ours raw | ours smoothed |
|---|---|---|---|---|
| v3 mixed | 3.0 | 3.0 | 3.0 | 3.0 |
| v3 jan | 3.0 | 3.0 | 3.0 | 3.0 |
| v2 design mixed | 5.0 | 5.6 | 5.0 | 5.6 |

## Social signals (RNN, 8 s, recency)

| config | seeds | folds | breathingNormally | coughing | hyperventilation | other | mean | pass (each ≥75) |
|---|---|---|---|---|---|---|---|---|
| v3 mixed | 2 | 10 | 88.5 ± 0.3 | 83.1 ± 0.3 | 80.3 ± 0.2 | 77.6 ± 0.5 | 82.4 | ✅ |
| v3 jan | 2 | 2 | 86.8 ± 2.0 | 85.0 ± 1.7 | 78.3 ± 3.7 | 77.4 ± 0.4 | 81.9 | ✅ |
| v2 design mixed | 1 | 10 | 86.0 | 82.1 | 79.6 | 78.7 | 81.6 | ✅ |

Each member as an unseen person (their own fold; mean over seeds):

| config | person | breathingNormally | coughing | hyperventilation | other |
|---|---|---|---|---|---|
| v3 mixed | s2503795 | 93.1 | 95.1 | 99.3 | 86.5 |
| v3 mixed | s2952925 | 100.0 | 99.3 | 97.8 | 82.3 |
| v3 jan | s2503795 | 95.8 | 95.8 | 100.0 | 86.2 |
| v3 jan | s2952925 | 100.0 | 98.5 | 100.0 | 82.3 |
| v2 design mixed | s2503795 | 91.0 | 93.6 | 97.4 | 88.1 |
| v2 design mixed | s2952925 | 98.6 | 100.0 | 100.0 | 80.9 |

Speed – app replay (2.56 s packets, a result every 0.5 s), median seconds from a change until the new class is shown and kept 4 s (seed 0):

| config | Jan2026 | ours |
|---|---|---|
| v3 mixed | 3.7 | 5.6 |
| v3 jan | 4.3 | 5.6 |
| v2 design mixed | 5.0 | 4.9 |

*ours* = the group members' recordings; for *mixed* only the fold model that never saw that person is used, for *jan* all fold models (they never see the members). Streams join two recordings with an instant cut, so the times compare models; a real person changes more gradually.

## Exported models (models/)

- final har: ok, 2.6 min, 0.027 ms per window (this Mac), fit on our recordings 99.8 %
- final social: ok, 11.9 min, 0.101 ms per window (this Mac), fit on our recordings 92.3 %

