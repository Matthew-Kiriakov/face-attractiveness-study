"""POST HOC P2 (FREEZE v1.6, written before the run): paired stratified face bootstrap for the drops of the held-out ladder.
Modes: smoke (development faces posing as lockbox; nothing written) | run (PRIMARY lockbox faces; one run; refuses if data/posthoc_ladder_ci_DONE exists).
Models, features and rungs are exactly those of posthoc_ladder_lockbox.py (copied logic); nothing is refitted on lockbox faces."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_lockbox_S as SL
import variance_partition as VP

R = SL.R
DONE = R / "data/posthoc_ladder_ci_DONE"
OUT = R / "results/posthoc_ladder_ci.txt"
SETS = SL.SETS
NB = 1000


def resid(v, C):
    C1 = np.column_stack([np.ones(len(v)), C])
    return v - C1 @ np.linalg.lstsq(C1, v, rcond=None)[0]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main(smoke):
    if not smoke and DONE.exists():
        sys.exit("REFUSED: P2 was run once.")
    rng = np.random.default_rng(20261112)
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
    keep = ~(nn > 0.6)
    P = SL.models(Xd, yd, list(range(len(tier2))))["boosting"].predict(Xl)
    oh_d = pd.get_dummies(dev.setname).reindex(columns=SETS, fill_value=0).values.astype(float)
    oh_l = pd.get_dummies(lk.setname).reindex(columns=SETS, fill_value=0).values.astype(float)
    mraw = SL.S.hgb()
    mraw.fit(np.column_stack([Xd, oh_d]), dev.rating_mean.values)
    Praw = mraw.predict(np.column_stack([Xl, oh_l]))
    raw = lk.rating_mean.values
    sidx = [np.where(keep & (lk.setname == s).values)[0] for s in SETS]
    cidx = {"age": [capcols.index("age_mesh")], "capture": [capcols.index(c) for c in VP.QUALITY]}
    keys = ["", "age", "capture", "age,capture"]

    def rungs(idx_sets):
        allidx = np.concatenate(idx_sets)
        out = {}
        for key in keys:
            cols = sum((cidx[k] for k in key.split(",") if k), [])
            f = (lambda v, i: resid(v[i], Cl[i][:, cols])) if cols else (lambda v, i: v[i])
            out["kept:" + (key or "none")] = corr(f(raw, allidx), f(Praw, allidx))
            out["removed:" + (key or "none")] = float(np.mean([corr(f(yl, i), f(P, i)) for i in idx_sets]))
        return out

    base = rungs(sidx)
    draws = [rungs([i[rng.integers(0, len(i), len(i))] for i in sidx]) for _ in range(NB)]
    ci = lambda v: f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
    lines = [f"POST HOC P2 paired stratified face bootstrap ({'SMOKE' if smoke else 'SCUT lockbox PRIMARY'}), {NB} draws, FREEZE v1.6; faces resampled within sets; model, judges, panel fixed"]
    for k, v in base.items():
        lines.append(f"rung {k:22s} r {v:.3f} {ci([d[k] for d in draws])}")
    pairs = [("step source (kept none -> removed none)", "kept:none", "removed:none"),
             ("kept: none -> age", "kept:none", "kept:age"), ("kept: none -> capture", "kept:none", "kept:capture"),
             ("kept: none -> age+capture", "kept:none", "kept:age,capture"),
             ("removed: none -> age", "removed:none", "removed:age"), ("removed: none -> capture", "removed:none", "removed:capture"),
             ("removed: none -> age+capture", "removed:none", "removed:age,capture")]
    for name, a, b in pairs:
        dd = [d[a] - d[b] for d in draws]
        lines.append(f"drop {name:42s} {base[a] - base[b]:+.3f} {ci(dd)}")
    lines.append("POST HOC (FREEZE v1.6). Drops are not additive shares; step 'source' changes target and model. Expectation: source ~0.145 far above 0; capture ~0.057 above 0; age ~0.022 may include 0.")
    text = "\n".join(lines)
    print(text)
    if not smoke:
        DONE.write_text("run once\n", encoding="utf-8")
        OUT.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] == "smoke")
