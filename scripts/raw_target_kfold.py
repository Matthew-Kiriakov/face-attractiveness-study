"""Paper-style setting (author question 2026-10-30): all non-lockbox faces in one K-fold, target = the RAW mean rating (1-5), not z-scored inside each set.
EXPLORATORY. Same sample, folds and models as zgeo_control.py (geometry z-scored inside set, 28 non-geometry features, all 112).
Why: Liang et al. 2018 report r 0.89 on all 5500 faces mixed with raw scores; set means differ (Asian vs Caucasian, women vs men, Internet vs 10k US), so part of
their r can come from telling the populations apart. Here: (a) target z inside set (our usual), (b) raw target, features without any set information,
(c) raw target + one-hot of the set (the model may know the population, as a CNN looking at pixels does). Also the share of raw-rating variance that lies between sets.
Expectation (AI guess, before the run): (b) pooled r about the same or slightly higher than (a) 0.70 because geometry carries some population information;
(c) 0.74-0.80 for the 28+geometry model; the between-set share of variance 10-25 %.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import all_clusters_kfold as A, group_ablation as G, scut_cm_harmony as S, exploration_v1 as E


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df, props, rest, tag = A.load_pooled()
    sets, _ = E.load()
    raw = pd.concat([sets[g][["file", "rating_mean"]] for g in sets]).drop_duplicates("file").set_index("file").rating_mean
    df["raw"] = raw.loc[df.file].values
    for s in df.setname.unique():
        m = df.setname == s
        for c in props:
            lo, hi = df.loc[m, c].quantile([0.01, 0.99])
            v = df.loc[m, c].clip(lo, hi)
            df.loc[m, c] = (v.fillna(v.median()) - v.mean()) / v.std()
    names = list(df.setname.unique())
    for s in names:
        df["set_" + s] = (df.setname == s).astype(float)
    dummies = ["set_" + s for s in names]
    yz, yr = df.y.values, df.raw.values
    tot = yr.var()
    between = sum(((df.raw[df.setname == s].mean() - yr.mean()) ** 2) * (df.setname == s).sum() for s in names) / len(df)
    print(f"n={len(df)}; raw rating mean {yr.mean():.2f} sd {yr.std():.2f}; between-set share of raw variance {between / tot:.3f}")
    print("set means: " + ", ".join(f"{s} {df.raw[df.setname == s].mean():.2f} (n={int((df.setname == s).sum())})" for s in names))
    allg = [c for v in G.GROUPS.values() for c in v]
    variants = {"геометрия z (42)": props, "геометрия z + 28": props + allg, "все 112": props + rest}
    for name, cols in variants.items():
        for fam, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
            for label, y, extra in (("(a) z-цель, без набора", yz, []), ("(b) сырая цель, без набора", yr, []), ("(c) сырая цель + набор", yr, dummies)):
                p = S.oof_predict(fac, df[cols + extra].values.astype(float), y, 2, S.SEED)
                inner = " ".join(f"{s.split()[0]}{s.split()[1][:2]} {np.corrcoef(p[(df.setname == s).values], y[(df.setname == s).values])[0, 1]:.2f}" for s in names)
                print(f"{name:20s} {fam:8s} {label:26s} pooled r {np.corrcoef(p, y)[0, 1]:.3f} R2 {G.r2(y, p):.3f} | inside sets: {inner}", flush=True)


main()
