"""Pipeline v4: which social-signal model is most accurate on the official 10 folds, and how fast does it respond?

Candidates (social signals, class weights balanced^0.6, the v2 recipe otherwise):
  cnn_w50 / cnn_w75 / cnn_w100           Harry's multi-branch CNN, 4 / 6 / 8 s window
  cnn_w50rec / cnn_w75rec / cnn_w100rec  the same, trained to favour the most recent seconds (recency splices: half
                                         of the training windows get their first 25-62 % replaced by another,
                                         gravity-aligned window; the label is that of the last 38-75 %)
  rnn_w75                                the deployed design (Conv-GRU, 6 s, no recency) = reference

Accuracy  official 10-fold CV, as the teacher computes it: per-class recall on fold k's datasetJan2026 test windows
          (step 25), averaged over the 10 folds. Default recipe 'jan' = trained on datasetJan2026 folds != k only.
Speed     app replay (fullrun4/replay.py): RESpeck packets of 32 samples (2.56 s), a result every 0.5 s, no smoothing;
          seconds from a social-signal change until the new class is shown and kept 4 s. Jan2026 pairs of fold k
          (unseen subjects) and same-posture pairs of both members' recordings (never trained on with 'jan').

Output: results/report.md (start here), results/summary.csv, results/recall_per_class.csv, results/streams.csv,
        results/jobs/*
Run:    ./run.sh [--seeds 0,1,2] [--recipe mixed]   (resumable)   |   ./run.sh --smoke   |   ./run.sh --summary-only
Split over two computers: give each a different --folds (e.g. Mac --folds 0,1,2,3,4, laptop --folds 5,6,7,8,9), copy
the laptop's results/jobs/* into this results/jobs/ and run --summary-only (WINDOWS_GPU_GUIDE.md). --gpu = use an
NVIDIA GPU (Linux / WSL2).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_DECISIONS = HERE.parent / "pipeline_v2" / "results" / "decisions.json"
CANDS = {   # name: (architecture, window, recency)
    "cnn_w50": ("harry_cnn", 50, False), "cnn_w50rec": ("harry_cnn", 50, True),
    "cnn_w75": ("harry_cnn", 75, False), "cnn_w75rec": ("harry_cnn", 75, True),
    "cnn_w100": ("harry_cnn", 100, False), "cnn_w100rec": ("harry_cnn", 100, True),
    "rnn_w75": ("harry_rnn", 75, False),
}
POWER = 0.6
SOCIAL = ["breathingNormally", "coughing", "hyperventilation", "other"]


def plan(C, smoke, seeds, recipe, folds=None, cands=None):
    folds = [C.PERSON_FOLD["s2503795"]] if smoke else (folds or C.FOLDS)
    cands = cands or list(CANDS)
    jobs = []
    for seed in ([0] if smoke else seeds):
        for k in folds:                                  # fold-major: a usable comparison exists early
            for name in cands:
                arch, W, rec = CANDS[name]
                train = [("jan_social", {"fold_ne": k}, None)]
                if recipe == "mixed":
                    train.append(("real", {"fold_ne": k}, C.REAL_TRAIN_STEP))
                jobs.append(dict(
                    id=f"{name}_{recipe}_s{seed}_f{k}", group="cv", task="social", model=arch, window=W, recency=rec,
                    cand=name, recipe=recipe, seed=seed, fold=k, power=POWER, train=train,
                    tests={"cv": {"task": "social", "window": W, "parts": [("jan_social", {"fold_eq": k}, C.TEST_STEP),
                                                                          ("real", {"fold_eq": k}, C.TEST_STEP)]}},
                    replay=("all" if recipe == "jan" else "own") if seed == 0 else None,
                    finetune=None, init_from=None, monitor=None, save_model=False, depends=[], person=None,
                    export=False, variant="all", prio=seed))
    return sorted(jobs, key=lambda j: j["prio"])


def summarise(C, log):
    import numpy as np
    import pandas as pd
    from collections import Counter
    rec_rows, st_rows, devices = [], [], Counter()
    for f in sorted((C.OUT / "jobs").glob("*.json")):
        r = json.loads(f.read_text())
        j = r["job"]
        if r.get("status") != "ok" or j.get("cand") not in CANDS:
            continue
        devices[r.get("device", "CPU on Darwin (Mac)")] += 1
        z = np.load(C.OUT / "jobs" / f"{j['id']}.npz")
        y, p, person = z["cv__y"], z["cv__P"].argmax(1), z["cv__person"]
        for scope in ["jan", *C.PEOPLE]:
            m = person == scope
            for c, name in enumerate(SOCIAL):
                mc = m & (y == c)
                if mc.any():
                    rec_rows.append({"cand": j["cand"], "recipe": j["recipe"], "seed": j["seed"], "fold": j["fold"],
                                     "scope": scope, "cls": name, "recall": 100 * float((p[mc] == c).mean())})
        st_rows += [{"cand": j["cand"], "recipe": j["recipe"], "fold": j["fold"], **s}
                    for s in r["info"].get("streams", [])]
    if not rec_rows:
        log("summary: no finished jobs yet")
        return
    R, S = pd.DataFrame(rec_rows), pd.DataFrame(st_rows)
    R.to_csv(C.OUT / "recall_per_class.csv", index=False)
    S.to_csv(C.OUT / "streams.csv", index=False)
    lines = ["# Pipeline v4 – social signals: CNN 4 / 6 / 8 s, with and without favouring the most recent seconds", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. **Accuracy** = official 10-fold CV as the teacher computes it: "
             "per-class recall on each fold's datasetJan2026 test windows, averaged over the folds (mean over seeds if "
             "several; ± = seed sd). **Speed** = app replay (2.56 s RESpeck packets, a result every 0.5 s, no smoothing): "
             "seconds from a social-signal change until the new class is shown and kept 4 s; *ours* = both group "
             "members' recordings (same-posture changes). Target: every class ≥ 75 %.", ""]
    rows = []
    for recipe in sorted(R.recipe.unique()):
        Rj = R[(R.recipe == recipe) & (R.scope == "jan")]
        per_seed_fold = Rj.groupby(["cand", "seed", "fold", "cls"]).recall.mean()
        per_seed = per_seed_fold.groupby(["cand", "seed", "cls"]).mean()
        acc = per_seed.groupby(["cand", "cls"]).agg(["mean", "std"]).unstack("cls")
        nseed = per_seed.groupby("cand").apply(lambda s: s.index.get_level_values("seed").nunique())
        nfold = Rj.groupby(["cand", "seed"]).fold.nunique().groupby("cand").min()
        worst = per_seed_fold.groupby(["cand", "seed", "fold"]).min().groupby("cand").min()
        for cand in [c for c in CANDS if c in acc.index]:
            arch, W, rec = CANDS[cand]
            v = acc.loc[cand, "mean"].reindex(SOCIAL)
            sd = acc.loc[cand, "std"].reindex(SOCIAL).fillna(0)
            row = {"recipe": recipe, "cand": cand, "model": "CNN" if arch == "harry_cnn" else "RNN",
                   "window_s": W / C.FS, "recency": rec, "seeds": int(nseed[cand]), "folds": int(nfold[cand]),
                   **{c: v[c] for c in SOCIAL}, **{f"{c}_sd": sd[c] for c in SOCIAL},
                   "mean": v.mean(), "min_class": v.min(), "worst_fold_min": worst[cand], "pass": bool(v.min() >= 75)}
            for person in C.PEOPLE:
                pr = R[(R.recipe == recipe) & (R.cand == cand) & (R.scope == person)].groupby("cls").recall.mean()
                for c in SOCIAL:
                    row[f"{person}_{c}"] = pr.get(c, np.nan)
            for stream in ("jan", "real"):
                d = S[(S.recipe == recipe) & (S.cand == cand) & (S.stream == stream)] if len(S) else S
                if len(d):
                    row[f"lat_med_{stream}"] = d.lat_raw.median(); row[f"lat_mean_{stream}"] = d.lat_raw.mean()
                    row[f"lat_p90_{stream}"] = d.lat_raw.quantile(.9); row[f"miss_{stream}"] = 100 * d.miss_raw.mean()
                    for c in SOCIAL:
                        dc = d[d.target == c]
                        row[f"lat_med_{stream}_{c}"] = dc.lat_raw.median() if len(dc) else np.nan
                        row[f"miss_{stream}_{c}"] = 100 * dc.miss_raw.mean() if len(dc) else np.nan
            rows.append(row)
    T = pd.DataFrame(rows)
    T.round(3).to_csv(C.OUT / "summary.csv", index=False)

    def f1(x, nd=1):
        return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

    for recipe in T.recipe.unique():
        t = T[T.recipe == recipe]
        lines += [f"## Recipe '{recipe}' ({'datasetJan2026 only – the teacher’s setting' if recipe == 'jan' else 'datasetJan2026 + group members'})", "",
                  "### Accuracy (official 10 folds, datasetJan2026 test windows)", "",
                  "| candidate | model | window | favours recent | seeds | folds | breathing | coughing | hyperventilation | "
                  "other | mean | weakest | worst fold (weakest class) | pass |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            cells = [f1(r[c]) + (f" ± {r[c + '_sd']:.1f}" if r.seeds > 1 else "") for c in SOCIAL]
            lines.append(f"| {r.cand} | {r.model} | {r.window_s:g} s | {'yes' if r.recency else 'no'} | {r.seeds} | "
                         f"{r.folds} | " + " | ".join(cells) + f" | {f1(r['mean'])} | {f1(r.min_class)} | "
                         f"{f1(r.worst_fold_min)} | {'✅' if r['pass'] else '❌'} |")
        lines += ["", "### Response time (seed 0, seconds)", "",
                  "| candidate | first result after becoming still | median Jan2026 / ours | mean Jan2026 / ours | "
                  "p90 Jan2026 / ours | never shown % (Jan2026 / ours) | ours, median to breathing / coughing / hyperv. / other | "
                  "ours, never shown % per class (same order) |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            g = lambda k: r.get(k, np.nan)
            lines.append(f"| {r.cand} | {r.window_s:g}–{r.window_s + 2.56:.1f} | {f1(g('lat_med_jan'))} / {f1(g('lat_med_real'))} | "
                         f"{f1(g('lat_mean_jan'))} / {f1(g('lat_mean_real'))} | {f1(g('lat_p90_jan'))} / {f1(g('lat_p90_real'))} | "
                         f"{f1(g('miss_jan'), 0)} / {f1(g('miss_real'), 0)} | "
                         + " / ".join(f1(g(f'lat_med_real_{c}')) for c in SOCIAL) + " | "
                         + " / ".join(f1(g(f'miss_real_{c}'), 0) for c in SOCIAL) + " |")
        lines += ["", "### Each group member as an unseen person (their own fold)", "",
                  "| candidate | person | breathing | coughing | hyperventilation | other |", "|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            for person in C.PEOPLE:
                lines.append(f"| {r.cand} | {person} | " + " | ".join(f1(r.get(f'{person}_{c}', np.nan)) for c in SOCIAL) + " |")
        lines.append("")
    lines += ["Notes: *first result after becoming still* = the window must fill with still data (the app pauses social "
              "results while moving) plus up to one 2.56 s packet. Replay streams join two recordings with an instant "
              "cut, so the times compare candidates; a real person changes more gradually. *Never shown* = the new class "
              "was not shown and held for 4 s within 12 s of the change. Worst fold = the lowest "
              "single-class recall in any one fold.", "",
              "Jobs run on: " + ", ".join(f"{d} ({n})" for d, n in sorted(devices.items())) + ". GPU and CPU results "
              "differ only by floating-point detail, far below the seed-to-seed spread."]
    (C.OUT / "report.md").write_text("\n".join(lines) + "\n")
    log(f"summary -> {C.OUT / 'report.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, fold 8 -> results_smoke/")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--recipe", default="jan", choices=["jan", "mixed"])
    ap.add_argument("--folds", default="", help="only these folds, e.g. 5,6,7,8,9 (default all 10)")
    ap.add_argument("--cands", default="", help=f"only these candidates (default all): {','.join(CANDS)}")
    ap.add_argument("--gpu", action="store_true", help="use the NVIDIA GPU (Linux / WSL2 with tensorflow[and-cuda])")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.gpu:
        os.environ["PDIOT_GPU"] = "1"                # read by fullrun4.jobs.init_worker in every worker
    folds = [int(k) for k in a.folds.split(",") if k.strip()] or None
    cands = [c.strip() for c in a.cands.split(",") if c.strip()] or None
    if cands and set(cands) - set(CANDS):
        raise SystemExit(f"unknown candidate(s) {set(cands) - set(CANDS)}; choose from {list(CANDS)}")
    if a.smoke:
        os.environ["FULLRUN_OUT"] = str(HERE / "results_smoke")
        os.environ["FULLRUN_MODELS"] = str(HERE / "models_smoke")
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(HERE.parent))
    from fullrun4 import config as C
    C.OUT.mkdir(parents=True, exist_ok=True)
    logf = open(C.OUT / "run.log", "a", buffering=1)

    def log(m):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}"
        print(line, flush=True); logf.write(line + "\n")

    if not C.DECISIONS.exists():
        pf = json.loads(V2_DECISIONS.read_text())["person_fold"] if V2_DECISIONS.exists() else {"s2503795": 8, "s2952925": 9}
        C.DECISIONS.write_text(json.dumps({"person_fold": pf, "source": str(V2_DECISIONS)}, indent=1))
    C.apply_decisions()
    log(f"===== pipeline v4 {'(SMOKE TEST) ' if a.smoke else ''}-> {C.OUT}; recipe {a.recipe}; seeds {a.seeds}; "
        f"folds {folds or 'all'}; candidates {cands or 'all'}; {'GPU' if a.gpu else 'CPU'}")
    from fullrun4 import data, jobs
    data.build_cache(log)
    if not a.summary_only:
        p = plan(C, a.smoke, [int(s) for s in a.seeds.split(",")], a.recipe, folds, cands)
        threads = a.threads or max(1, (os.cpu_count() or 8) // a.workers)
        jobs.run_all(p, a.workers, threads, a.smoke, log)
    jobs.init_worker(2)
    summarise(C, log)


if __name__ == "__main__":
    main()
