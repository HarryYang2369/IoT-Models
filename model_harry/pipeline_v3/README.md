# Pipeline v3 – faster models

Two models, both trained to favour the most recent seconds (recency splices), everything else as in v2:

| Task | Model | Window | Recency splice | Chosen from |
|---|---|---|---|---|
| Activity | harry_cnn | 4 s (50) | first 12–25 samples replaced, no gravity alignment, label = last 2–3 s | `../results/har_window_study` (w50rec: dynamic 89.0 %, 3.0 s vs 5.6 s) |
| Social | harry_rnn, weights balanced^0.6 | 8 s (100) | first 25–62 samples replaced, gravity-aligned, label = last 3–6 s | `../results/social_window_study` (w100rec: weakest class 77.3 %) |

- `./run.sh`: full run (resumable). The final models are trained first, then the CV. About 3.5 h on the M1 Pro.
- `./run.sh --smoke`: 1-epoch end-to-end test (writes `results_smoke/`, `models_smoke/`).
- `./run.sh --summary-only`: rebuilds `results/report.md` from the finished jobs.

Outputs:

- `models/`: app files (`harry_har_cnn.*`, `harry_social_rnn.*`) and `INSTALL.md` (how to copy; the social model needs
  `socialWindowSamples = 100` in the app).
- `results/report.md`: per-class recall as marked (datasetJan2026 test windows, 10 folds × 3 seeds) for two training
  recipes (*mixed* = like the final model, *jan* = datasetJan2026 only), each member as an unseen person, and the
  app latency replay against the v2 design.
- `results/recall_per_class.csv`, `results/streams.csv`, `results/jobs/` (every job's settings, log, predictions).

Code: `fullrun3/` = v2's `fullrun2/{config,data,jobs}.py` with per-job window / recency / replay, plus `replay.py`
(ported from the window studies); `run_v3.py` = plan + summary.
