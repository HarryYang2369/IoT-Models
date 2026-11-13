"""Paths and constants of pipeline v3 (2026-11-13): the v2 code (fullrun2), copied, for two models only:

  * activity: harry_cnn, 4 s window (50 samples), trained to favour the most recent seconds
              (recency splices without gravity alignment, last 25-38 samples kept; har_window_study 'w50rec')
  * social:   harry_rnn, 8 s window (100 samples), class weights balanced^0.6, trained to favour the most recent
              seconds (gravity-aligned splices, last 38-75 samples kept; social_window_study 'w100rec')

Everything else is v2: data cache, official folds, group members as extra fold subjects (s2503795 -> fold 8,
s2952925 -> fold 9), recipe (AdamW, 32 epochs, label smoothing 0.1, augmentation), TFLite export in the app contract.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]            # model_harry/pipeline_v3/
LIB = HERE.parent                                      # model_harry/  (shared model library harry/, v1 outputs for history)
ROOT = LIB.parent                                      # IoT/  (data/)
JAN_DIR = ROOT / "data/datasetJan2026"
FOLD_DIR = ROOT / "data/fold_assignments"
PEOPLE = {"s2503795": "Subject A (s2503795)", "s2952925": "Subject B (s2952925)"}
OUT = Path(os.environ.get("FULLRUN_OUT") or HERE / "results")          # results, figures, tables, jobs, knowledge base
MODELS = Path(os.environ.get("FULLRUN_MODELS") or HERE / "models")     # exported app models
CACHE_SRC = HERE / "cache"                             # v3's own signal cache (built from data/ if missing)
STUDY = HERE / "social_window_study"                   # the social window study (accuracy vs latency)
STUDY_OLD = LIB / "results" / "social_window_study"    # where the first run of the study wrote (copied in once)

FS = 12.5
TEST_STEP = 25                 # window step of every test set (2 s), as in Lab 3 B9
REAL_TRAIN_STEP = 6            # group recordings are few -> denser windows when they are used for training
APP_STEPS = [25, 12, 6]
APP_DEFAULT_STEP = 6

ARCHS = {                      # Harry's four architectures; the first one per task is deployed
    "har": ["harry_cnn", "harry_rnn", "harry_mlp", "harry_feat"],
    "social": ["harry_rnn", "harry_cnn", "harry_mlp", "harry_feat"],
}
FINAL = {t: a[0] for t, a in ARCHS.items()}
SOCIAL_POWER = 0.6             # social class weights = balanced ** 0.6
POWER_SWEEP = [0.0, 0.3, 0.8, 1.0]
TARGETS = {"static": 0.90, "dynamic": 0.85, "social": 0.75}
MIN_CLASS_RULE = 0.77          # social window choice: every class >= 77 % (2 points above the mark)
SEEDS = [0, 1, 2]
FOLDS = list(range(10))

# Scoring scopes of every CV job (stored in the 'variant' column of the tables):
SCOPES = ["all", "jan"]        # + "group" and one per person
HEADLINE = "all"
SCOPE_LABEL = {"all": "all test windows (Jan2026 + group)", "jan": "Jan2026 test windows (as marked)",
               "group": "group members' test windows", **{p: f"{lab} test windows" for p, lab in PEOPLE.items()}}

MODEL_NAMES = {
    "harry_cnn": "Multi-branch CNN", "harry_rnn": "Conv-GRU", "harry_mlp": "Gravity-split MLP",
    "harry_feat": "Spectral-feature net",
}
APP_FILES = {"har": "harry_har_cnn", "social": "harry_social_rnn"}   # names the app loads

# ───────────────────────── frozen decisions (written by run_pipeline_v2.py)
DECISIONS = OUT / "decisions.json"
W = {"har": 50, "social": 100}          # v3: 4 s activity, 8 s social
SOCIAL_RECENCY = False
SOCIAL_CANDIDATE = "w100"
HAR_RECIPE = "mixed"
PERSON_FOLD = {}


def apply_decisions():
    global SOCIAL_RECENCY, SOCIAL_CANDIDATE, HAR_RECIPE, PERSON_FOLD
    if not DECISIONS.exists():
        return {}
    d = json.loads(DECISIONS.read_text())
    if "social" in d:
        W["social"] = int(d["social"]["window"])
        SOCIAL_RECENCY = bool(d["social"]["recency"])
        SOCIAL_CANDIDATE = d["social"]["candidate"]
    HAR_RECIPE = d.get("har_recipe", HAR_RECIPE)
    PERSON_FOLD = d.get("person_fold", PERSON_FOLD)
    return d


def recipe(task):
    """'mixed' = group members' training-fold recordings added to datasetJan2026; 'jan' = datasetJan2026 only."""
    return HAR_RECIPE if task == "har" else "mixed"


apply_decisions()
