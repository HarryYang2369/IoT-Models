"""Pipeline v3: the faster models (trained to favour the most recent seconds), trained, cross-validated and exported.

  activity  harry_cnn, 4 s window, recency-trained          -> models/harry_har_cnn.*
  social    harry_rnn, 8 s window, recency-trained, w^0.6   -> models/harry_social_rnn.*

Jobs (resumable: finished jobs are kept, re-running continues):
  final   one model per task trained on ALL data (all datasetJan2026 incl. unlisted recordings + both members),
          exported in the app contract -> models/ (run first, so the files are ready early)
  cv      official 10-fold CV x 3 seeds, two training recipes, always scored on datasetJan2026 test windows
          (= the CW3 marking) and on each member's own fold (unseen person):
            mixed  folds != k of datasetJan2026 + the members whose fold is not k (= how the final model is trained)
            jan    folds != k of datasetJan2026 only (strict: the members are never seen)
          seed 0 also replays the app on change streams (latency, see fullrun3/replay.py)
  ref     the v2 (deployed) designs with the mixed recipe, seed 0, same replay: activity 4 s and social 6 s without
          recency training -> the fair "is it faster?" reference

Output: results/report.md (start here), results/recall_per_class.csv, results/streams.csv, results/jobs/*;
        models/ + models/INSTALL.md
Run:    ./run.sh   (or ./run.sh --smoke for a 1-epoch end-to-end test in results_smoke/, models_smoke/)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_DECISIONS = HERE.parent / "pipeline_v2" / "results" / "decisions.json"
ALL_DATA = "datasetJan2026 (all, incl. recordings not in the fold CSVs) + both group members' recordings, mixed"
TASKS = {"har": dict(model="harry_cnn", window=50, power=None, ref_window=50),
         "social": dict(model="harry_rnn", window=100, power=0.6, ref_window=75)}


def plan(C, smoke):
    folds = [C.PERSON_FOLD["s2503795"]] if smoke else C.FOLDS
    seeds = [0] if smoke else C.SEEDS
    jobs = []

    def J(**kw):
        for k, v in dict(seed=0, fold=None, power=None, finetune=None, init_from=None, monitor=None, save_model=False,
                         depends=[], person=None, export=False, variant="all", tests={}, replay=None,
                         recency=True, recipe="mixed").items():
            kw.setdefault(k, v)
        jobs.append(kw)

    def train(task, recipe, k):
        if k is None:
            return [(f"jan_{task}", {}, None), (f"janx_{task}", {}, None), ("real", {}, C.REAL_TRAIN_STEP)]
        parts = [(f"jan_{task}", {"fold_ne": k}, None)]
        if recipe == "mixed":
            parts.append(("real", {"fold_ne": k}, C.REAL_TRAIN_STEP))
        return parts

    def test(task, W, k):
        return {"cv": {"task": task, "window": W, "parts": [(f"jan_{task}", {"fold_eq": k}, C.TEST_STEP),
                                                           ("real", {"fold_eq": k}, C.TEST_STEP)]}}

    for task, t in TASKS.items():
        J(id=f"final_{task}", group="final", task=task, model=t["model"], window=t["window"], power=t["power"],
          train=train(task, "mixed", None), save_model=True, export=True, prio=0,
          trained_on=ALL_DATA + "; trained to favour the most recent seconds (recency splices)")
    for seed in seeds:
        for recipe in ("mixed", "jan"):
            for k in folds:
                for task, t in TASKS.items():
                    J(id=f"cv_{task}_{recipe}_s{seed}_f{k}", group="cv", task=task, model=t["model"], window=t["window"],
                      power=t["power"], seed=seed, fold=k, recipe=recipe, train=train(task, recipe, k),
                      tests=test(task, t["window"], k), replay=("own" if recipe == "mixed" else "all") if seed == 0 else None,
                      prio=(1 if recipe == "mixed" else 2) if seed == 0 else 5)
    for k in folds:
        for task, t in TASKS.items():
            J(id=f"ref_{task}_mixed_s0_f{k}", group="ref", task=task, model=t["model"], window=t["ref_window"],
              power=t["power"], fold=k, recipe="mixed", recency=False, train=train(task, "mixed", k),
              tests=test(task, t["ref_window"], k), replay="own", prio=3)
    return sorted(jobs, key=lambda j: j["prio"])


# ───────────────────────── summary
def summarise(C, log):
    import numpy as np
    import pandas as pd
    from harry.data import HAR_CLASSES, SOCIAL_CLASSES
    classes = {"har": HAR_CLASSES, "social": SOCIAL_CLASSES}
    rec_rows, st_rows = [], []
    for f in sorted((C.OUT / "jobs").glob("*.json")):
        r = json.loads(f.read_text())
        j = r["job"]
        if r.get("status") != "ok" or j["group"] not in ("cv", "ref"):
            continue
        cfg = f"{'v3' if j['group'] == 'cv' else 'v2 design'} {j['recipe']}"
        z = np.load(C.OUT / "jobs" / f"{j['id']}.npz")
        y, p, person = z["cv__y"], z["cv__P"].argmax(1), z["cv__person"]
        for scope in ["jan", *C.PEOPLE]:
            m = person == scope
            for c, name in enumerate(classes[j["task"]]):
                mc = m & (y == c)
                if mc.any():
                    rec_rows.append({"task": j["task"], "cfg": cfg, "seed": j["seed"], "fold": j["fold"],
                                     "scope": scope, "cls": name, "recall": 100 * float((p[mc] == c).mean())})
        st_rows += [{"task": j["task"], "cfg": cfg, "fold": j["fold"], **s} for s in r["info"].get("streams", [])]
    if not rec_rows:
        log("summary: no finished CV jobs yet")
        return
    R, S = pd.DataFrame(rec_rows), pd.DataFrame(st_rows)
    R.to_csv(C.OUT / "recall_per_class.csv", index=False)
    S.to_csv(C.OUT / "streams.csv", index=False)
    fold_n = R[R.scope == "jan"].groupby(["task", "cfg", "seed"]).fold.nunique()
    # CW3: per-class recall averaged over the folds (per seed), then the mean over seeds
    per_seed = R[R.scope == "jan"].groupby(["task", "cfg", "seed", "cls"]).recall.mean()
    acc = per_seed.groupby(["task", "cfg", "cls"]).agg(["mean", "std", "count"])
    people = R[R.scope != "jan"].groupby(["task", "cfg", "scope", "cls"]).recall.mean()

    def lat(task, cfg, stream, col):
        d = S[(S.task == task) & (S.cfg == cfg) & (S.stream == stream)] if len(S) else S
        return (f"{d[col].median():.1f}" if len(d) and col in d and d[col].notna().any() else "–")

    L = ["# Pipeline v3 – faster models (trained to favour the most recent seconds)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Official 10-fold CV, per-class recall averaged over the folds "
         "(CW3 marking), mean over seeds (± = seed sd). **As marked** = datasetJan2026 test windows only. "
         "*v3* = the new models; *v2 design* = the deployed design (4 s activity / 6 s social, no recency) with the same "
         "code, seed 0 only, as the speed reference. Recipes: *mixed* = members' recordings in the training folds "
         "(how the final model is trained); *jan* = datasetJan2026 only (strict).", ""]
    order = [c for c in ("v3 mixed", "v3 jan", "v2 design mixed") if c in acc.index.get_level_values("cfg")]
    for task in ("har", "social"):
        cl = classes[task]
        if task not in acc.index.get_level_values("task"):
            continue
        L += [f"## {'Activity (CNN, 4 s, recency)' if task == 'har' else 'Social signals (RNN, 8 s, recency)'}", ""]
        if task == "har":
            L += ["| config | seeds | folds | static | dynamic | pass (≥90 / ≥85) | " + " | ".join(cl) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(cl)]
        else:
            L += ["| config | seeds | folds | " + " | ".join(cl) + " | mean | pass (each ≥75) |",
                  "|---|---|---|" + "---|" * len(cl) + "---|---|"]
        for cfg in order:
            if (task, cfg) not in acc.droplevel("cls").index:
                continue
            a = acc.loc[(task, cfg)]
            v = a["mean"].reindex(cl)
            sd = a["std"].reindex(cl).fillna(0)
            ns = int(a["count"].max())
            nf = int(fold_n.loc[(task, cfg)].min())
            cells = [f"{v[c]:.1f}" + (f" ± {sd[c]:.1f}" if ns > 1 else "") for c in cl]
            if task == "har":
                st, dy = v[cl[:5]].mean(), v[cl[5:]].mean()
                ok = "✅" if st >= 90 and dy >= 85 else "❌"
                L.append(f"| {cfg} | {ns} | {nf} | {st:.1f} | {dy:.1f} | {ok} | " + " | ".join(cells) + " |")
            else:
                ok = "✅" if v.min() >= 75 else "❌"
                L.append(f"| {cfg} | {ns} | {nf} | " + " | ".join(cells) + f" | {v.mean():.1f} | {ok} |")
        L += ["", "Each member as an unseen person (their own fold; mean over seeds):", "",
              "| config | person | " + ("static | dynamic" if task == "har" else " | ".join(cl)) + " |",
              "|---|---|" + ("---|---|" if task == "har" else "---|" * len(cl))]
        for cfg in order:
            for person in C.PEOPLE:
                if (task, cfg, person) not in people.droplevel("cls").index:
                    continue
                v = people.loc[(task, cfg, person)].reindex(cl)
                body = (f"{v[cl[:5]].mean():.1f} | {v[cl[5:]].mean():.1f}" if task == "har"
                        else " | ".join(f"{v[c]:.1f}" for c in cl))
                L.append(f"| {cfg} | {person} | {body} |")
        L += ["", "Speed – app replay (2.56 s packets, a result every 0.5 s), median seconds from a change until the new "
              "class is shown and kept 4 s (seed 0):", ""]
        if task == "har":
            L += ["| config | Jan2026 raw | Jan2026 smoothed | ours raw | ours smoothed |", "|---|---|---|---|---|"]
            for cfg in order:
                L.append(f"| {cfg} | {lat(task, cfg, 'jan', 'lat_raw')} | {lat(task, cfg, 'jan', 'lat_sm')} | "
                         f"{lat(task, cfg, 'real', 'lat_raw')} | {lat(task, cfg, 'real', 'lat_sm')} |")
        else:
            L += ["| config | Jan2026 | ours |", "|---|---|---|"]
            for cfg in order:
                L.append(f"| {cfg} | {lat(task, cfg, 'jan', 'lat_raw')} | {lat(task, cfg, 'real', 'lat_raw')} |")
        L.append("")
    L += ["*ours* = the group members' recordings; for *mixed* only the fold model that never saw that person is used, "
          "for *jan* all fold models (they never see the members). Streams join two recordings with an instant cut, "
          "so the times compare models; a real person changes more gradually.", ""]
    ex = []
    for task in ("har", "social"):
        f = C.OUT / "jobs" / f"final_{task}.json"
        if f.exists():
            r = json.loads(f.read_text())
            e = r.get("info", {}).get("export", {})
            ex.append(f"- final {task}: {r.get('status')}, {r.get('seconds', 0) / 60:.1f} min, "
                      f"{e.get('ms_per_window', float('nan')):.3f} ms per window (this Mac), "
                      f"fit on our recordings {100 * e.get('fit_acc_on_real', float('nan')):.1f} %")
    if ex:
        L += ["## Exported models (models/)", "", *ex, ""]
    (C.OUT / "report.md").write_text("\n".join(L) + "\n")
    log(f"summary -> {C.OUT / 'report.md'}")


INSTALL = """# Installing the v3 models in the app (copy by hand)

Made by pipeline_v3 (run_v3.py). Both models are trained on all data (all datasetJan2026 + both group members) and
trained to favour the most recent seconds. Check results/report.md before installing.

Back up the app's current files first:

```bash
mkdir -p ~/IoT/assets/models_v2_backup && cp ~/IoT/assets/models/harry_* ~/IoT/assets/models_v2_backup/
```

## Activity (CNN, 4 s) – no app code change (same 50-sample window)

```bash
cp harry_har_cnn.tflite harry_har_cnn_meta.json harry_har_cnn_example.json ~/IoT/assets/models/
```

## Social (RNN, 8 s) – needs ONE app change first

The app checks the social window length. In `~/IoT/lib/social.dart` change

    const int socialWindowSamples = 75;

to

    const int socialWindowSamples = 100;

(and the comment above it that says 75), then:

```bash
cp harry_social_rnn.tflite harry_social_rnn_meta.json harry_social_rnn_example.json ~/IoT/assets/models/
```

## Check

`flutter run`, then look for the `EXAMPLE app:` line in the debug console: it must match the `_example.json`
file. Run `flutter test` (tests that hard-code the 75-sample social window must be updated with the social change).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, fold 8, seed 0 -> results_smoke/, models_smoke/")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="TF threads per worker (0 = cores / workers)")
    a = ap.parse_args()
    if a.smoke:
        os.environ["FULLRUN_OUT"] = str(HERE / "results_smoke")
        os.environ["FULLRUN_MODELS"] = str(HERE / "models_smoke")
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(HERE.parent))           # model_harry/: the shared model library harry/
    from fullrun3 import config as C
    C.OUT.mkdir(parents=True, exist_ok=True)
    logf = open(C.OUT / "run.log", "a", buffering=1)

    def log(m):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}"
        print(line, flush=True); logf.write(line + "\n")

    if not C.DECISIONS.exists():                   # the members' folds, exactly as in v2
        pf = json.loads(V2_DECISIONS.read_text())["person_fold"] if V2_DECISIONS.exists() else {"s2503795": 8, "s2952925": 9}
        C.DECISIONS.write_text(json.dumps({"person_fold": pf, "source": str(V2_DECISIONS)}, indent=1))
    C.apply_decisions()
    log(f"===== pipeline v3 {'(SMOKE TEST) ' if a.smoke else ''}-> {C.OUT}; members' folds {C.PERSON_FOLD}")
    from fullrun3 import data, jobs
    data.build_cache(log)
    if not a.summary_only:
        p = plan(C, a.smoke)
        (C.OUT / "plan.json").write_text(json.dumps(p, indent=1, default=str))
        threads = a.threads or max(1, (os.cpu_count() or 8) // a.workers)
        jobs.run_all(p, a.workers, threads, a.smoke, log)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / "INSTALL.md").write_text(INSTALL)
    jobs.init_worker(2)
    summarise(C, log)


if __name__ == "__main__":
    main()
