"""App latency replay for a trained fold model (ported unchanged from har_window_study.py / social_window_study.py,
so the numbers are comparable with those studies).

A stream = the last 12 s of recording A followed by the first 12 s of recording B (a different class). The app is
replayed on it: RESpeck packets of 32 samples (2.56 s; the display can only change at a packet end), a new model
result every STEP = 6 samples (0.5 s). Latency = seconds from the change until the display shows B's class and keeps
it for 4 s; the mean over 4 packet phases is stored per stream.

  jan streams    pairs of datasetJan2026 test recordings of fold k (unseen subjects); social pairs are
                 gravity-aligned at the seam (different subjects), activity pairs are not (a posture change *is* a
                 gravity change)
  real streams   pairs of the group members' recordings. mode "own": only the member(s) whose test fold is k
                 (mixed recipe: the only fold models that never saw that person); mode "all": both members (the
                 Jan2026-only recipe never sees them)

Activity: 'raw' = smoothing off, 'sm' = the app default (majority of the last 3 results).
Social: no smoothing (as in the app); results only from same-posture pairs, so no activity change is involved.
"""
import itertools
from collections import Counter, deque

import numpy as np

from . import config as C
from .data import load_set

PKT, FS, NA, NB, STEP = 32, 12.5, 150, 150, 6
HOLD = int(4 * FS)


def _pairs_jan(set_name, k, rng, n=80):
    sig, meta = load_set(set_name)
    idx = [i for i in np.flatnonzero(meta.fold.to_numpy() == k) if meta.n[i] >= max(NA, NB)]
    y = meta.y.to_numpy()
    out = []
    while len(out) < n:
        a, b = rng.choice(idx, 2, replace=False)
        if y[a] != y[b]:
            out.append((sig[a], sig[b], int(y[a]), int(y[b]), "jan"))
    return out


def _people(fold, mode):
    return [p for p in C.PEOPLE if mode == "all" or C.PERSON_FOLD.get(p) == fold]


def _pairs_real_har(fold, mode, rng, per_person=50):
    sig, meta = load_set("real")
    out = []
    for person in _people(fold, mode):
        stationary = meta.stationary.astype(str).str.lower() == "true"
        m = meta[(meta.person == person) & (~stationary | (meta["sub"] == "normal"))]
        cand = [(a, b) for a, b in itertools.permutations(m.index, 2) if m.har_y[a] != m.har_y[b]]
        for j in rng.choice(len(cand), min(per_person, len(cand)), replace=False):
            a, b = cand[j]
            out.append((sig[a], sig[b], int(meta.har_y[a]), int(meta.har_y[b]), person))
    return out


def _pairs_real_social(fold, mode, rng, per_person=40):
    sig, meta = load_set("real")
    out = []
    for person in _people(fold, mode):
        st = meta[(meta.person == person) & (meta.social_y >= 0)]
        cand = [(a, b) for a, b in itertools.permutations(st.index, 2)
                if st.posture[a] == st.posture[b] and st.social_y[a] != st.social_y[b]]
        for j in rng.choice(len(cand), min(per_person, len(cand)), replace=False):
            a, b = cand[j]
            out.append((sig[a], sig[b], int(meta.social_y[a]), int(meta.social_y[b]), person))
    return out


def _display(P, smooth, phase):
    n = len(P)
    q = deque(maxlen=3)
    since, shown, cur = 0, -1, -1
    disp = np.full(n, -1)
    for t in range(n):
        since += 1
        if P[t] >= 0 and since >= STEP:
            since = 0
            if smooth:
                q.append(P[t]); c = Counter(q); top = max(c.values())
                shown = next(lab for lab in reversed(q) if c[lab] == top)
            else:
                shown = P[t]
        if (t - phase) % PKT == PKT - 1:
            cur = shown
        disp[t] = cur
    return disp


def _latency(disp, target):
    n = len(disp)
    for t in range(NA, n):
        if disp[t] == target and np.all(disp[t:min(n, t + HOLD)] == target):
            return (t - NA + 1) / FS
    return np.nan


def _rows(model, W, streams, align, tag, smooth_modes, classes):
    if not streams:
        return []
    first = NA - 4 * PKT                       # earlier windows cannot affect the display after the change
    Xs, spans = [], []
    for s in streams:
        A, B = s[0][-NA:], s[1][:NB]
        if align:
            B = B - B[:25].mean(0) + A[-25:].mean(0)
        x = np.concatenate([A, B]).astype(np.float32)
        ends = np.arange(max(W - 1, first), len(x))
        Xs.append(np.stack([x[t - W + 1:t + 1] for t in ends])); spans.append(ends)
    Pall = model.predict(np.concatenate(Xs), batch_size=4096, verbose=0).argmax(1)
    rows, o = [], 0
    for s, ends in zip(streams, spans):
        P = np.full(NA + NB, -1); P[ends] = Pall[o:o + len(ends)]; o += len(ends)
        target = s[3]
        row = {"stream": tag, "person": s[4], "from": classes[s[2]], "target": classes[target],
               "steady_acc": float(np.mean(P[-100:] == target))}
        for smooth in smooth_modes:
            lat = [_latency(_display(P, smooth, ph), target) for ph in range(0, PKT, 8)]
            key = "sm" if smooth else "raw"
            row[f"lat_{key}"] = float(np.nanmean(lat)) if not np.all(np.isnan(lat)) else np.nan
            row[f"miss_{key}"] = float(np.mean(np.isnan(lat)))
        rows.append(row)
    return rows


def replay(model, task, W, fold, mode):
    from harry.data import HAR_CLASSES, SOCIAL_CLASSES
    if task == "har":
        rows = _rows(model, W, _pairs_jan("jan_har", fold, np.random.default_rng(100 + fold)), False, "jan",
                     (False, True), HAR_CLASSES)
        rows += _rows(model, W, _pairs_real_har(fold, mode, np.random.default_rng(7)), False, "real",
                      (False, True), HAR_CLASSES)
    else:
        rows = _rows(model, W, _pairs_jan("jan_social", fold, np.random.default_rng(100 + fold)), True, "jan",
                     (False,), SOCIAL_CLASSES)
        rows += _rows(model, W, _pairs_real_social(fold, mode, np.random.default_rng(7)), False, "real",
                      (False,), SOCIAL_CLASSES)
    return rows
