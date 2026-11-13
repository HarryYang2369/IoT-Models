# Pipeline v4: social signals, CNN window and recency study

The question: for social signals, how do Harry's CNN with a 4 s, 6 s or 8 s window, each with and without training
to favour the most recent seconds, score on the official 10-fold test (the teacher's test), and how fast do they
respond in the app? The deployed design (RNN, 6 s, no recency) runs alongside as the reference.

| Candidate | Model | Window | Favours recent seconds |
|---|---|---|---|
| cnn_w50, cnn_w50rec | CNN | 4 s (50 samples) | no / yes |
| cnn_w75, cnn_w75rec | CNN | 6 s (75) | no / yes |
| cnn_w100, cnn_w100rec | CNN | 8 s (100) | no / yes |
| rnn_w75 | RNN (deployed design) | 6 s | no |

- **Accuracy:** train on 9 official folds of datasetJan2026, test on the 10th, per-class recall averaged over the
  10 folds (recipe `jan`, the default). `--recipe mixed` also adds the group members' recordings to training.
- **Speed:** app replay (2.56 s packets, a result every 0.5 s), on Jan2026 pairs and on both members' same-posture
  recordings (never trained on with `jan`). Reports the median, mean and p90 time to show a change, the time to the
  first result after becoming still, and, per class, how often a change is never shown (for example hyperventilation).
- **Recency splice:** half of the training windows get their first 25–62 % replaced by another, gravity-aligned
  window, and the label is the class of the last 38–75 %. At 8 s this is identical to v3 and the window study.

Run: `./run.sh` (seed 0, about 4.5 h on the Mac alone; resumable). Options: `--folds 5,6,7,8,9` (only these folds), `--cands cnn_w75,rnn_w75` (only these candidates), `--gpu` (NVIDIA GPU on Linux/WSL2).

**Two computers:** the Mac runs `--folds 0,1,2,3,4` and the Windows laptop (RTX 4060, WSL2) runs `--folds 5,6,7,8,9 --gpu`; then the laptop's `results/jobs/*` are copied into the Mac's `results/jobs/` and `--summary-only` rebuilds the report. Step-by-step: `WINDOWS_GPU_GUIDE.md`. The report lists which device ran how many jobs.
`./run.sh --seeds 1,2` adds seeds later. `./run.sh --summary-only` rebuilds the report from the finished jobs.

Output: `results/report.md` (start here), `results/summary.csv`, `results/recall_per_class.csv`,
`results/streams.csv`, `results/jobs/`.

Code: `fullrun4/` is v3's `fullrun3/` (data cache, folds, recipe, replay), with the recency splice made
proportional to the window. `run_v4.py` holds the plan and the summary.
