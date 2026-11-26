"""Confirmatory step B: MEBeauty cv_fold 0, ONE run (registered in RESEARCH 1e 'Registration of B', FREEZE.md v1.1/v1.2).
Modes:
  smoke : the same code with fold 4 posing as the test fold (fold 0 is NOT read; trained on SCUT development + MEBeauty folds 1-3; labels of folds 1-4 are readable under data/external/mebeauty/ALLOW_LABELS.md). Bug check only.
  run   : the real run. Refuses unless EXPECTATIONS_WRITTEN is True and data/mebeauty_B_DONE does not exist; creates that file at the start (one run).
Usage: python mebeauty_B.py smoke | run [--rerun]
v1.4 (2026-11-11): in 'run' the numbers go only to results/mebeauty_B_run.txt.part (console: stage names); inputs and library versions are verified against scripts/input_hashes.json BEFORE the marker; the marker is JSON; '--rerun' is the ONE rerun after a crash (FREEZE v1.4);
  the words 'transfers', 'useful', 'transfers (pooled)' and the downgrade of claim 71 are read from A_boosting (SCUT-only model on fold 0), the transfer column; B is reported as pooled-training performance; 'pooling helps' stays B minus A.
  A primary group whose interval includes 0 is 'not decidable' if the upper bound is at or above 0.30 and 'failed' otherwise; the downgrade needs a 'failed' group or a mean of four below 0.25. R2 per group and the per-face predictions (data/mebeauty_B_predictions.csv) are saved.

Design (written 2026-11-09 before fold 0 labels are read; [AI] design; the author's 'давай все доделаем' 2026-11-09 [mine] covers writing and freezing it):
 Training set B = SCUT development faces (3737; tier 2 z inside set, target = judges' z inside set) + MEBeauty folds 1-4 (1970 faces; tier 2 winsorised 1/99 % and z inside gender x ethnicity with folds 1-4 statistics, target z inside group). No sample weights, no group or source indicator.
 Models: boosting (primary) and ridge, settings of scut_cm_harmony.hgb() / ridge(), fitted once.
 Test: the 492 faces of cv_fold 0 (features z with the folds 1-4 statistics of their group, as in mebeauty_transfer.mb_features; label z inside group within fold 0 - Pearson does not depend on it).
 Metric: Pearson and Spearman inside each of caucasian_female, caucasian_male, asian_female, asian_male (primary), the other four groups reported separately; face bootstrap 1000 (95 %).
 Words: 'transfers' = lower bound above 0 in all four primary groups; 'useful' = mean of the four primary Pearson >= 0.30 (both as registered 2026-11-07, applied to boosting; ridge next to it). Power note (label-free, written before the run): fold 0 has only 29 asian_male and 38 asian_female faces (caucasian_female 108, caucasian_male 86); an r of 0.35 on 29 faces has an interval of about [0.00, 0.63], so 'transfers' in all four groups is likely to fail on sample size alone and would then be reported as 'not decidable in asian_male', not as a failure of the model. Companion word [AI]: 'transfers (pooled)' = lower bound above 0 for r over the 261 primary faces with predictions centred inside group. New word [AI]: 'pooling helps' = mean of the four primary Pearson of B (boosting) minus the same for the SCUT-only model (A on fold 0) is >= 0.03 with the paired bootstrap lower bound above 0.
 Secondary (all reported, none chosen after seeing): A on fold 0 (SCUT-only boosting and ridge, refitted here); B without the faces flagged by the frozen quality rules (mebeauty_flags.csv flag_any); B partial correlation given perceived age (genderage.onnx, as review_mebeauty_capture_age.py), mean of four groups.
 Expectation (AI, before): B boosting mean of four primary Pearson 0.38-0.52 (A on folds 1-4 was 0.331; B has seen the photo domain and raters of MEBeauty); A on fold 0 about 0.28-0.38; pooling helps by +0.03 to +0.12; ridge below boosting by 0.00-0.05;
   flagged faces fit worse than the rest; partial given age 0.08-0.15 below the plain value. 'transfers' met; 'useful' met; 'pooling helps' met.
 A smoke run on fold 4 (exploratory, labels allowed) was done after these expectations were written and is not used to change them.
 Downgrade rule (FREEZE v1.1 point 8): if 'transfers' fails in any primary group or the mean of four is below 0.25, the exploratory claim 71 (transfer 0.33) is downgraded to 'not confirmed'.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import glob
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mebeauty_transfer as T
import overnight_common as O
import scut_cm_harmony as S
import run_guard as RG

R, MB = O.R, T.MB
EXPECTATIONS_WRITTEN = True
DONE = R / "data/mebeauty_B_DONE"
OUT = []
RESULTS = R / "results" / "mebeauty_B_run.txt"
PREDS = R / "data" / "mebeauty_B_predictions.csv"
OUTO = RG.Out(None)


def say(s=""):
    OUT.append(s)
    OUTO.say(s)


def pr2(p, y):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def all_labels(smoke):
    parts = [pq.read_table(f, columns=["image_id", "beauty_score", "cv_fold"]).to_pandas() for f in sorted(glob.glob(str(MB / "fbp_extended/*.parquet")))]
    d = pd.concat(parts, ignore_index=True)
    if smoke:
        d = d[d.cv_fold != 0]  # fold 0 labels dropped at once in the smoke run
    return d[["image_id", "beauty_score"]]


def zlab(d):
    d["y"] = d.groupby("group").beauty_score.transform(lambda v: (v - v.mean()) / v.std())
    return d


def corr_table(d, col, rng, n=1000):
    res = {}
    for g in T.PRIMARY + sorted(set(d.group) - set(T.PRIMARY)):
        s = d[d.group == g]
        if len(s) < (3 if g in T.PRIMARY else 20):  # a primary group is never dropped (v1.4)
            continue
        x, y = s[col].values, s.y.values
        bs = [pearsonr(x[i], y[i])[0] for i in (rng.integers(0, len(s), len(s)) for _ in range(n))]
        sp = spearmanr(x, y)[0]
        res[g] = (pearsonr(x, y)[0], np.percentile(bs, 2.5), np.percentile(bs, 97.5), sp, len(s), pr2(x, y))
    return res


def report(title, d, cols, rng):
    say(f"\n{title} (n = {len(d)})")
    out = {}
    for col in cols:
        res = corr_table(d, col, rng)
        say(f"  {col}")
        for g, (r, lo, hi, sp, n, q) in res.items():
            st = "" if g not in T.PRIMARY else (" transfers" if lo > 0 else " not decidable" if hi >= 0.30 else " failed")
            say(f"    {g:18s} n {n:4d} {'primary' if g in T.PRIMARY else 'other  '} Pearson {r:+.3f} [{lo:+.3f}, {hi:+.3f}] Spearman {sp:+.3f} R2 {q:+.3f}" + (st if col == "A_boosting" else ""))
        m4 = np.mean([res[g][0] for g in T.PRIMARY])
        if col == "A_boosting":
            failed = [g for g in T.PRIMARY if res[g][1] <= 0 and res[g][2] < 0.30]
            say(f"    mean of four primary {m4:.3f}; transfers {all(res[g][1] > 0 for g in T.PRIMARY)}; useful {m4 >= 0.30}; claim 71 downgraded: {bool(failed) or m4 < 0.25} (failed groups {failed})")
        else:
            say(f"    mean of four primary {m4:.3f} (the registered words are read from A_boosting only)")
        pp = d[d.group.isin(T.PRIMARY)].copy()
        pp["pc"] = pp[col] - pp.groupby("group")[col].transform("mean")
        pb = [pearsonr(pp.pc.values[i], pp.y.values[i])[0] for i in (rng.integers(0, len(pp), len(pp)) for _ in range(1000))]
        say(f"    {'companion ' if col == 'A_boosting' else 'pooled r, no registered word: '}'transfers (pooled)': r over the {len(pp)} primary faces with prediction centred inside group {pearsonr(pp.pc, pp.y)[0]:+.3f} [{np.percentile(pb, 2.5):+.3f}, {np.percentile(pb, 97.5):+.3f}]; lower bound above 0: {np.percentile(pb, 2.5) > 0}")
        out[col] = m4
    return out


def OUTO_set(o):
    global OUTO
    OUTO = o


def run(smoke=False, rerun=False):
    if not smoke:
        if not EXPECTATIONS_WRITTEN:
            sys.exit("REFUSED: expectations not written.")
        OUTO_set(RG.Out(RG.start(DONE, RESULTS, Path(__file__), rerun, "B")))
    test_fold = 4 if smoke else 0
    rng = np.random.default_rng(20261109)
    df, props, rest, tier2 = O.load()
    mb = T.mb_features(tier2)
    lab = all_labels(smoke)
    mb = zlab(mb.merge(lab, on="image_id", how="left" if not smoke else "inner").assign(group=lambda d: d.ethnicity + "_" + d.gender))
    tr = mb[(mb.cv_fold != 0) & (mb.cv_fold != test_fold)].copy()
    te = mb[mb.cv_fold == test_fold].copy()
    tr["y"] = tr.groupby("group").beauty_score.transform(lambda v: (v - v.mean()) / v.std())
    say(f"{'SMOKE (fold 4 as test)' if smoke else 'B'}: train SCUT {len(df)} + MEBeauty {len(tr)}; test fold {test_fold}: {len(te)} faces")
    Xs, ys = df[tier2].values.astype(float), df.y.values
    Xm, ym = tr[tier2].values.astype(float), tr.y.values
    Xt = te[tier2].values.astype(float)
    for name, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
        a = fac()
        a.fit(Xs, ys)
        te["A_" + name] = a.predict(Xt)
        b = fac()
        b.fit(np.vstack([Xs, Xm]), np.concatenate([ys, ym]))
        te["B_" + name] = b.predict(Xt)
    cols = ["A_boosting", "A_ridge", "B_boosting", "B_ridge"]
    OUTO.stage("fitted; correlations")
    m4 = report("B and A on the test fold", te, cols, rng)
    # pooling helps: paired bootstrap of the difference of the mean of four
    gs = [te[te.group == g] for g in T.PRIMARY]
    ds = []
    for _ in range(1000):
        v = []
        for s in gs:
            i = rng.integers(0, len(s), len(s))
            v.append(pearsonr(s.B_boosting.values[i], s.y.values[i])[0] - pearsonr(s.A_boosting.values[i], s.y.values[i])[0])
        ds.append(np.mean(v))
    say(f"\npooling helps: B - A mean of four (boosting) {m4['B_boosting'] - m4['A_boosting']:+.3f} [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]; word met: {m4['B_boosting'] - m4['A_boosting'] >= 0.03 and np.percentile(ds, 2.5) > 0}")
    OUTO.stage("secondary")
    report("B without flagged faces", te[~te.flag.fillna(False).astype(bool)], ["B_boosting", "B_ridge"], rng)
    # partial given perceived age
    try:
        import onnxruntime as ort
        import age_model as AM
        sess = ort.InferenceSession(str(AM.MODEL), providers=["CPUExecutionProvider"])
        rows = S.load_done(MB / "mebeauty_landmarks_web.jsonl")
        age = AM.estimate(sess, {f: rows[f] for f in te.file.values if f in rows}, MB / "images_aligned")
        t2 = te.merge(age[["file", "age_mesh"]], on="file", how="left").dropna(subset=["age_mesh"]).reset_index(drop=True)
        for name in ("B_boosting", "B_ridge"):
            t2["rp"], t2["ry"] = np.nan, np.nan
            for g in t2.group.unique():
                m = (t2.group == g).values
                A_ = np.column_stack([np.ones(m.sum()), t2.loc[m, "age_mesh"].values])
                for src, dst in ((name, "rp"), ("y", "ry")):
                    v = t2.loc[m, src].values
                    t2.loc[m, dst] = v - A_ @ np.linalg.lstsq(A_, v, rcond=None)[0]
            part = [pearsonr(t2[t2.group == g].rp, t2[t2.group == g].ry)[0] for g in T.PRIMARY]
            say(f"\npartial given perceived age, {name}: mean of four {np.mean(part):.3f} (per group " + ", ".join(f"{g} {v:+.3f}" for g, v in zip(T.PRIMARY, part)) + ")")
    except Exception as e:  # image-side only; reported, not fatal
        say(f"\nAGE PARTIAL NOT COMPUTED (a registered secondary result is missing): {type(e).__name__}: {e}")
        print("[warning] age partial not computed; see the .part file", flush=True)
    if not smoke:
        te[["image_id", "file", "group", "flag", "y"] + cols].to_csv(PREDS, index=False)
        say("\ndone (B is finished; it is not repeated)")
        OUTO.finish(RESULTS)
        print("[done] results/mebeauty_B_run.txt written; predictions in data/mebeauty_B_predictions.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"smoke": lambda: run(True), "run": lambda: run(False, "--rerun" in sys.argv)}[sys.argv[1]]()
