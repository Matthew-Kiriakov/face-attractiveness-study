"""C3 and C4 of the external review, from the 60 individual judges (All_Ratings.xlsx). PRE-REGISTERED 2026-10-31 before the numbers. EXPLORATORY, non-lockbox faces only
(lockbox faces are dropped before any rating is used). Usage: python rater_models.py audit | taste. Sets: CM 10k_US, AM DataTang, AM internet, CF 10k_US, AF internet, AF DataTang.

audit (C3, 'explain every single rating'): out-of-fold tier-2 predictions (boosting, mean of 10 CV seeds, z target) converted to points (z x SD of the set's face means + the set mean).
 Per face the standard error of the judges' mean is SD(60 ratings) / sqrt(60). Reported per set: share of faces whose prediction lies within 1 and 2 SE of the face mean (and the same for the
 naive prediction 'set mean'), residual SD in points, mean SE. The 50 largest absolute residuals per set go to results/2026-10-31_audit_top_residuals.csv with the three largest ridge contributions
 (coefficient x z value, ridge fitted on the set) per face. Expectation (Opus, before): 15-25 % of faces within 2 SE, residual SD 0.25-0.35 points against an SE of about 0.08;
 the goal 'at least 80 % of faces within 2 SE' is the milestone definition.
taste (C4, shared versus private taste): per judge j and face f, e_jf = rating - mean of the OTHER 59 judges for that face - judge j's mean offset. Per set and judge a ridge (alpha 10, standardised)
 of e on tier 2 with 5-fold CV over faces gives a held-out R2; null = the same with the faces of e shuffled (20 shuffles per set pooled over judges). Then the PCA of the judge-by-face residual
 matrix per set with parallel analysis (columns shuffled independently, 50 reps, 95th percentile) for the number of taste dimensions, and the correlation of PC1 face scores with the features.
 Variance split of one rating (per set): shared (ICC), shared part predicted by tier 2 (R2 of the face mean x ICC), predictable private part (median judge R2 x (1 - ICC)).
 Expectations (Opus, before): median per-judge held-out R2 of e 0.005-0.02; 1-3 taste PCs above parallel analysis; PC1 explains 3-6 % of the residual variance and correlates |rho| >= 0.2 with smile,
 sex-typicality or jaw/adiposity features. Claim rule: 'private taste is partly structured' needs >= 20 of 60 judges with a held-out R2 interval above 0. Kill: median <= 0.002 and no PC above the null.
 The 7 judges with 'original Rating' repeats are not used as a retest (probably the inconsistent raters)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_common as O

SETS = ["CM 10k_US", "AM DataTang", "AM internet", "CF 10k_US", "AF internet", "AF DataTang"]
RATINGS = O.R / "data/scut-fbp5500/SCUT-FBP5500_v2/All_Ratings.xlsx"


def long_ratings() -> pd.DataFrame:
    lk = pd.read_csv(O.R / "data/lockbox.csv")[["file", "group", "src", "lock"]]
    keep = lk[~lk.lock.astype(bool)]
    a = pd.read_excel(RATINGS, sheet_name="ALL").merge(keep, left_on="Filename", right_on="file")
    a["set"] = a.group + " " + a.src
    return a[a.set.isin(SETS)]


def audit() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df, props, rest, tier2 = O.load()
    y = df.y.values
    a = long_ratings()
    se = a.groupby("Filename").Rating.agg(["mean", "std", "count"])
    se["se"] = se["std"] / np.sqrt(se["count"])
    p = O.oof_seeds(df, tier2, y, "boosting", "audit_t2").mean(axis=0)
    rows, top = [], []
    print("per set: n | SD of face means | mean SE of the judges' mean | residual SD (points) | share within 1 SE / 2 SE | naive (set mean) within 1 SE / 2 SE")
    for s in SETS:
        m = (df.setname == s).values
        f = df.file[m].values
        obs = se.loc[f, "mean"].values
        sd_set, mu = obs.std(), obs.mean()
        pred = p[m] * sd_set + mu
        sef = se.loc[f, "se"].values
        res = obs - pred
        w1, w2 = np.mean(np.abs(res) <= sef), np.mean(np.abs(res) <= 2 * sef)
        n1, n2 = np.mean(np.abs(obs - mu) <= sef), np.mean(np.abs(obs - mu) <= 2 * sef)
        print(f"  {s:12s} n={m.sum():4d} | {sd_set:.2f} | {sef.mean():.3f} | {res.std():.3f} | {w1:.2f} / {w2:.2f} | {n1:.2f} / {n2:.2f}")
        X = df.loc[m, tier2].fillna(0).values
        sc = StandardScaler().fit(X)
        rg = Ridge(alpha=10.0).fit(sc.transform(X), (obs - mu))
        contrib = sc.transform(X) * rg.coef_
        for i in np.argsort(-np.abs(res))[:50]:
            order = np.argsort(-np.abs(contrib[i]))[:3]
            top.append({"set": s, "file": f[i], "observed": obs[i], "predicted": pred[i], "residual": res[i], "se": sef[i], "top_contributions": "; ".join(f"{tier2[k]} {contrib[i, k]:+.2f}" for k in order)})
    pd.DataFrame(top).to_csv(O.R / "results/2026-10-31_audit_top_residuals.csv", index=False)
    pooled = []
    for s in SETS:
        m = (df.setname == s).values
        obs = se.loc[df.file[m].values, "mean"].values
        pooled.append(np.abs(obs - (p[m] * obs.std() + obs.mean())) <= 2 * se.loc[df.file[m].values, "se"].values)
    print(f"\npooled share within 2 SE: {np.concatenate(pooled).mean():.3f} (milestone goal 0.80)")
    print("done", flush=True)


def parallel_threshold(E: np.ndarray, rng, reps=50) -> np.ndarray:
    ev = []
    for _ in range(reps):
        P = np.column_stack([rng.permutation(E[:, j]) for j in range(E.shape[1])])
        ev.append(np.linalg.svd(P - P.mean(axis=0), compute_uv=False) ** 2)
    return np.percentile(np.array(ev), 95, axis=0)


def taste() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df, props, rest, tier2 = O.load()
    a = long_ratings()
    rng = np.random.default_rng(20261031)
    print("per set: ICC | R2 of the face mean by tier 2 (ridge OOF) | median judge held-out R2 of e [null 95th pct] | judges with R2 > null 95th pct | PCs above parallel analysis | PC1 share of residual variance")
    for s in SETS:
        d = a[a.set == s]
        W = d.pivot_table(index="Filename", columns="Rater", values="Rating")
        m = df.setname == s
        files = [f for f in W.index if f in set(df.file[m])]
        W = W.loc[files]
        X = df.set_index("file").loc[files, tier2].fillna(0).values.astype(float)
        n, k = W.shape
        Wv = W.values
        fm = Wv.mean(axis=1)
        res_face = fm.var(ddof=1) - ((Wv - fm[:, None] - Wv.mean(axis=0)[None, :] + Wv.mean()) ** 2).sum() / ((n - 1) * (k - 1)) / k
        res_var = ((Wv - fm[:, None] - Wv.mean(axis=0)[None, :] + Wv.mean()) ** 2).sum() / ((n - 1) * (k - 1))
        icc = res_face / (res_face + res_var)
        E = np.empty_like(Wv)
        for j in range(k):
            others = np.delete(Wv, j, axis=1).mean(axis=1)
            e = Wv[:, j] - others
            E[:, j] = e - e.mean()
        kf = KFold(5, shuffle=True, random_state=0)

        def cv_r2(target):
            pr = np.zeros(n)
            for tr, te in kf.split(X):
                sc = StandardScaler().fit(X[tr])
                pr[te] = Ridge(alpha=10.0).fit(sc.transform(X[tr]), target[tr]).predict(sc.transform(X[te]))
            return 1 - np.sum((target - pr) ** 2) / np.sum((target - target.mean()) ** 2)

        r2_face = cv_r2(fm)
        r2j = np.array([cv_r2(E[:, j]) for j in range(k)])
        null = np.array([cv_r2(rng.permutation(E[:, j])) for j in range(k) for _ in range(1)] + [cv_r2(rng.permutation(E[:, j])) for j in range(k)] + [cv_r2(rng.permutation(E[:, j])) for j in range(k)])
        thr = np.percentile(null, 95)
        ev = np.linalg.svd(E - E.mean(axis=0), compute_uv=False) ** 2
        th = parallel_threshold(E, rng)
        npc = int((ev > th).sum())
        U, S_, Vt = np.linalg.svd(E - E.mean(axis=0), full_matrices=False)
        pc1 = U[:, 0] * S_[0]
        from scipy.stats import spearmanr
        cors = sorted(((c, spearmanr(pc1, df.set_index("file").loc[files, c].fillna(0))[0]) for c in tier2 + ["smile", "mouth_open"] if c in df.columns), key=lambda t: -abs(t[1]))[:4]
        print(f"  {s:12s} ICC {icc:.2f} | face-mean R2 {r2_face:.3f} | median judge R2 {np.median(r2j):+.4f} [null95 {thr:+.4f}] | {int((r2j > thr).sum())}/{k} | PCs {npc} | PC1 {ev[0] / ev.sum():.3f} | PC1 top: " + ", ".join(f"{c} {r:+.2f}" for c, r in cors))
        print(f"      one-rating split: shared {icc:.2f}; shared predicted by tier 2 {r2_face * icc:.3f}; predictable private {max(np.median(r2j), 0) * (1 - icc):.4f}; rest (private unpredictable + noise) {1 - icc - max(np.median(r2j), 0) * (1 - icc):.2f}")
    print("done", flush=True)


if __name__ == "__main__":
    {"audit": audit, "taste": taste}[sys.argv[1]]()
