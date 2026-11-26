"""POST HOC held-out ladder (FREEZE v1.5, written before the run; the lockbox labels are already open, so this is post hoc by definition and labelled so everywhere).
Question: does the 'subtract what is not the face' ladder of the page (development: 0.830 -> 0.694 -> 0.616, results/2026-11-11_page_remove_ladder.txt) appear on the held-out SCUT faces?
Modes: smoke (development faces posing as lockbox, as in scut_lockbox_S.py smoke; nothing written) | run (PRIMARY lockbox faces, no twins; one run; refuses if data/posthoc_ladder_DONE exists).
Definition (fixed here, before any held-out ladder number exists):
 - Models are the development-trained models of scut_lockbox_S.py, nothing is refitted on lockbox faces: P = tier 2 boosting on the judges' z inside set (S model), Praw = tier 2 boosting plus six set indicators on the raw 1-5 mean (one fit, development faces only, same settings S.hgb).
 - Features z with development statistics of the set (S pipeline). Capture = scut_lockbox_S/variance_partition QUALITY (six statistics) and age_mesh; for the partials they are winsorised 1/99 %, missing filled and z-scored with development statistics of the set.
 - Rungs as on the page: levers 'source', 'age', 'capture' removed or not. Source kept = pooled Pearson r over all PRIMARY faces of the raw rating with Praw (partial correlation given age and/or the six capture statistics, linear, pooled). Source removed = Pearson r inside each set of y (z) with P, partial given age and/or capture inside the set; the rung value is the mean of the six per-set r.
 - Intervals: face bootstrap 1000 (pooled for source kept; per set, then mean of six, for source removed). All eight rungs are printed, none is chosen.
 - Expectation (written before the run, from the development ladder and the S result): source kept (none) 0.80-0.85; source removed 0.678 (the S mean, as printed); source+age+capture 0.58-0.66; the three drops (source, age, capture) keep the order of the development ladder (source largest).
 - No verdict words; status POST HOC; the development ladder stays development-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_lockbox_S as SL
import variance_partition as VP

R = SL.R
DONE = R / "data/posthoc_ladder_DONE"
OUT = R / "results/posthoc_ladder_lockbox.txt"
SETS = SL.SETS


def resid(v, C):
    C1 = np.column_stack([np.ones(len(v)), C])
    return v - C1 @ np.linalg.lstsq(C1, v, rcond=None)[0]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main(smoke):
    if not smoke and DONE.exists():
        sys.exit("REFUSED: the post hoc ladder was run once.")
    rng = np.random.default_rng(20261111)
    allf, props, tier2, excluded = SL.build()
    if smoke:
        allf = allf[~allf.lock].reset_index(drop=True)
        allf["lock"] = np.random.default_rng(1).random(len(allf)) < 0.25
    dev, lk = allf[~allf.lock].copy(), allf[allf.lock].copy()
    Xd, Xl = np.zeros((len(dev), len(tier2))), np.zeros((len(lk), len(tier2)))
    yd, yl = np.zeros(len(dev)), np.zeros(len(lk))
    capcols = VP.QUALITY + ["age_mesh"]
    Cl = np.zeros((len(lk), len(capcols)))
    for s in SETS:
        md, ml = (dev.setname == s).values, (lk.setname == s).values
        st = SL.zfit(dev[md], tier2)
        Xd[md], Xl[ml] = SL.zapply(st, dev[md], tier2), SL.zapply(st, lk[ml], tier2)
        mu, sd = dev.loc[md, "rating_mean"].mean(), dev.loc[md, "rating_mean"].std()
        yd[md], yl[ml] = (dev.loc[md, "rating_mean"] - mu) / sd, (lk.loc[ml, "rating_mean"] - mu) / sd
        for j, c in enumerate(capcols):
            lo, hi = dev.loc[md, c].quantile([0.01, 0.99])
            dv = dev.loc[md, c].clip(lo, hi)
            Cl[ml, j] = ((lk.loc[ml, c].clip(lo, hi)).fillna(dv.median()) - dv.mean()) / dv.std()
    nn = SL.twins(lk.file.values, dev.file.values)
    keep = ~(nn > 0.6)  # PRIMARY
    P = SL.models(Xd, yd, list(range(len(tier2))))["boosting"].predict(Xl)
    oh_d = pd.get_dummies(dev.setname).reindex(columns=SETS, fill_value=0).values.astype(float)
    oh_l = pd.get_dummies(lk.setname).reindex(columns=SETS, fill_value=0).values.astype(float)
    mraw = SL.S.hgb()
    mraw.fit(np.column_stack([Xd, oh_d]), dev.rating_mean.values)
    Praw = mraw.predict(np.column_stack([Xl, oh_l]))
    raw = lk.rating_mean.values
    sidx = {s: np.where(keep & (lk.setname == s).values)[0] for s in SETS}
    allidx = np.where(keep)[0]
    ci = lambda v: f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
    lines = [f"POST HOC held-out ladder ({'SMOKE: development faces posing as lockbox' if smoke else 'SCUT lockbox PRIMARY'}), n = {len(allidx)} faces; definition in the header of scripts/posthoc_ladder_lockbox.py (FREEZE v1.5)."]
    cidx = {"age": [capcols.index("age_mesh")], "capture": [capcols.index(c) for c in VP.QUALITY]}
    for key in ["", "age", "capture", "age,capture"]:
        cols = sum((cidx[k] for k in key.split(",") if k), [])
        f = (lambda v, i: resid(v[i], Cl[i][:, cols])) if cols else (lambda v, i: v[i])
        name = key or "none"
        r_all = corr(f(raw, allidx), f(Praw, allidx))
        b = [corr(f(raw, allidx[j]), f(Praw, allidx[j])) for j in (rng.integers(0, len(allidx), len(allidx)) for _ in range(1000))]
        lines.append(f"{name:18s} source kept   r {r_all:.3f} {ci(b)}")
        per, bs = {}, np.zeros((1000, len(SETS)))
        for k, s in enumerate(SETS):
            ii = sidx[s]
            per[s] = corr(f(yl, ii), f(P, ii))
            bs[:, k] = [corr(f(yl, ii[j]), f(P, ii[j])) for j in (rng.integers(0, len(ii), len(ii)) for _ in range(1000))]
        lines.append(f"{('source,' + key).rstrip(','):18s} source removed r {np.mean(list(per.values())):.3f} {ci(bs.mean(axis=1))}  per set: " + ", ".join(f"{s} {v:.3f}" for s, v in per.items()))
    lines.append("POST HOC (FREEZE v1.5). Expectation written before the run: none 0.80-0.85; source 0.678; source+age+capture 0.58-0.66; order of drops as in development.")
    text = "\n".join(lines)
    print(text)
    if not smoke:
        DONE.write_text("run once\n", encoding="utf-8")
        OUT.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] == "smoke")
