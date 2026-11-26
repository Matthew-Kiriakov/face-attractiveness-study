"""Two cheap checks on DEVELOPMENT faces only (lockbox faces are never read; ratings come from rater_models.long_ratings(), which keeps only lock == False). EXPLORATORY, 2026-11-12.
Expectation written before the run: (1) split-half reliability of the 60-judge mean (30 vs 30, Spearman-Brown) 0.90-0.97 per set, so a ceiling for r with that mean of about 0.95-0.98, far above 0.58-0.81;
(2) several jaw measures (jaw-w, jaw_turn, adip_pa, zyg_jaw) correlate |rho| >= 0.5 within sets; M_core minus its jaw part keeps the sign of C11/C12 on development faces.
Run: python scripts/cheap_checks_dev.py   -> results/2026-11-12_cheap_checks_dev.txt"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_lockbox_S as SL
import rater_models as RM

R = SL.R
SETS = SL.SETS
OUT = []


def say(t=""):
    print(t)
    OUT.append(t)


def splithalf():
    a = RM.long_ratings()
    say("1. Split-half reliability of the judges' mean (development faces, 200 random splits of the 60 judges into 30 + 30)")
    say("   set | n faces | r between half-means | Spearman-Brown for 60 judges | sqrt = ceiling for r with the 60-judge mean | mean single-judge r with mean of other 59")
    rng = np.random.default_rng(20261112)
    for s in SETS:
        w = a[a.set == s].groupby(["Filename", "Rater"]).Rating.mean().unstack("Rater")
        w = w.dropna(axis=0, how="any")
        X = w.values
        rs = []
        for _ in range(200):
            p = rng.permutation(X.shape[1])
            rs.append(np.corrcoef(X[:, p[:30]].mean(1), X[:, p[30:]].mean(1))[0, 1])
        r = float(np.mean(rs))
        sb = 2 * r / (1 + r)
        tot = X.sum(1)
        single = np.mean([np.corrcoef(X[:, j], (tot - X[:, j]) / (X.shape[1] - 1))[0, 1] for j in range(X.shape[1])])
        say(f"   {s:12s} n {len(w):4d} | {r:.3f} | {sb:.3f} | {np.sqrt(sb):.3f} | {single:.3f}")
    say("   Reading: the ceiling is for r with THIS panel's mean. It is not the same as agreement of a model with a new panel, and it says nothing about taste shared by all 60 judges.")


def measures():
    allf, props, tier2, excl = SL.build()
    dev = allf[~allf.lock].reset_index(drop=True)
    parts = []
    for s in SETS:
        d = dev[dev.setname == s]
        st = SL.zfit(d, SL.L_COLS)
        m = SL.look_measures(d, st)
        zc = pd.DataFrame(SL.zapply(st, d, SL.L_COLS), columns=SL.L_COLS, index=d.index)
        comp = {k: np.mean([sg * zc[c] for c, sg in SL.COMPONENTS[k]], axis=0) for k in SL.CORE}
        m["M_nojaw"] = np.mean([comp[k] for k in SL.CORE if k != "jaw"], axis=0)
        m["jaw_comp"] = comp["jaw"]
        m["y"] = (d.rating_mean - d.rating_mean.mean()) / d.rating_mean.std()
        m["set"] = s
        m["sex"] = SL.SEX[s[:2]]
        parts.append(m)
    return pd.concat(parts)


def interm():
    M = measures()
    keys = ["chin-philtrum", "zyg_jaw", "canthal", "eye-gap", "sclera_show_low", "face-wh", "thirds_dev", "brow_gap", "brow_td", "M_core", "jaw_turn", "adip_pa", "jaw-w"]
    say("\n2. Rank correlations between the claim measures on development faces (ranks inside each set, pooled; sexes pooled)")
    Rk = M.groupby("set")[keys].rank(pct=True)
    C = Rk.corr()
    say("   pairs with |rho| >= 0.40:")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if abs(C.loc[a, b]) >= 0.40:
                say(f"   {a:16s} {b:16s} {C.loc[a, b]:+.2f}")
    say("   M_core against its own jaw component and against the claim measures of the jaw group: " + ", ".join(f"{k} {C.loc['M_core', k]:+.2f}" for k in ["jaw-w", "jaw_turn", "adip_pa", "zyg_jaw"]))
    say("   M_core without its jaw part, same-sex Spearman with judges' z (mean over sets of the sex), against M_core with the jaw part:")
    for sex, lab, sg in (("M", "C11 men (+M_core)", +1), ("F", "C12 women (-M_core)", -1)):
        for key in ("M_core", "M_nojaw"):
            vals = []
            for s in [x for x in SETS if SL.SEX[x[:2]] == sex]:
                d = M[M.set == s]
                vals.append(sg * spearmanr(d[key], d.y, nan_policy="omit")[0])
            say(f"   {lab:22s} {key:8s} mean sign*rho {np.mean(vals):+.3f}  per set " + ", ".join(f"{v:+.2f}" for v in vals))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    splithalf()
    interm()
    (R / "results/2026-11-12_cheap_checks_dev.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
