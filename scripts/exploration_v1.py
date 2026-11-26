"""Exploration on the non-lockbox 75% (protocol v1, RESEARCH.md 1d). EXPLORATORY.

Usage: python exploration_v1.py curve|clusters|vp   (each part checkpoints to data/cache_expl/<part>.json and resumes)
Rules fixed before the run:
 curve:    per group x source with >=200 non-lock faces, ridge and boosting (fixed settings), 20 random splits (25% test),
           train on the first k of the rest, k in 50,100,200,400,800,1600 (<= available). Report mean Pearson r on test.
 clusters: CM 10k US and AM DataTang, 30 subsamples (80% without replacement): adjusted Rand of the re-clustered 42 features vs
           the full-sample clusters, and the share of subsamples in which each full-sample cluster is in the top 3 by grouped
           permutation importance (boosting, 5 folds x1, 4 shuffles).
 vp:       inside one source, ridge, G/Q/A shares (as variance_partition.py) on 30 subsamples (80% without replacement); mean and
           2.5-97.5 percentile of each share. A stability range, not a confidence interval (subsamples overlap).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_clusters as FC
import scut_cm_harmony as S
import variance_partition as VP

R = S.ROOT
CACHE = R / "data/cache_expl"
CACHE.mkdir(exist_ok=True)
SEED = S.SEED


def load():
    cols = list(pd.read_csv(S.CSV_PATH, nrows=1).columns)
    props = cols[cols.index("face-wh"): cols.index("ala-jaw") + 1]
    lock = pd.read_csv(R / "data/lockbox.csv")
    age = pd.read_csv(R / "data/scut_age_mesh_all.csv")
    sets = {"CM": pd.read_csv(S.CSV_PATH).merge(pd.read_csv(R / "data/scut_cm_quality.csv"), on="file").merge(pd.read_csv(R / "data/scut_cm_source.csv"), on="file")}
    for g in ("AM", "CF", "AF"):
        sets[g] = pd.read_csv(R / f"data/scut_{g.lower()}_harmony.csv")
    for g in sets:
        d = sets[g].merge(age[["file", "age_mesh"]], on="file").merge(lock[["file", "lock"]], on="file")
        d = d[~d.lock]
        z = ((d[props] - d[props].median()) / (d[props].quantile(.75) - d[props].quantile(.25))).abs().max(axis=1)
        if (z > 50).any():
            print("excluded as degenerate landmarks (feature > 50 IQR from median):", g, d.loc[z > 50, "file"].tolist(), flush=True)
        sets[g] = d[z <= 50].reset_index(drop=True)
    return sets, props


def plan(sets, minimum=200):
    return [(g, s) for g in ("CM", "AM", "CF", "AF") for s in sets[g].src.value_counts().index if (sets[g].src == s).sum() >= minimum]


def ck(name):
    p = CACHE / f"{name}.json"
    return p, (json.loads(p.read_text()) if p.exists() else {})


def curve(sets, props):
    p, done = ck("curve")
    sizes = [50, 100, 200, 400, 800, 1600]
    for g, s in plan(sets):
        key = f"{g}|{s}"
        if key in done:
            continue
        d = sets[g][sets[g].src == s].reset_index(drop=True)
        X, y = d[props].values.astype(float), d.rating_mean.values
        res = {}
        for name, fac in (("ridge", S.ridge), ("boosting", S.hgb)):
            res[name] = {}
            for k in sizes:
                if k > int(len(d) * 0.75):
                    break
                rs = []
                for rep in range(20):
                    rng = np.random.default_rng(SEED + rep)
                    perm = rng.permutation(len(d))
                    te, tr = perm[: len(d) // 4], perm[len(d) // 4:][:k]
                    m = fac().fit(X[tr], y[tr])
                    rs.append(np.corrcoef(m.predict(X[te]), y[te])[0, 1])
                res[name][k] = [float(np.mean(rs)), float(np.std(rs))]
        done[key] = {"n": len(d), **res}
        p.write_text(json.dumps(done))
        print(key, len(d), {n: {k: round(v[0], 3) for k, v in r.items()} for n, r in res.items()}, flush=True)


def clusters(sets, props):
    p, done = ck("clusters")
    for g, s in (("CM", "10k_US"), ("AM", "DataTang")):
        key = f"{g}|{s}"
        if key in done:
            continue
        d = sets[g][sets[g].src == s].reset_index(drop=True)
        lab0, _ = FC.clusters(d, props, FC.CUT)
        members = {c: [i for i in range(len(props)) if lab0[i] == c] for c in sorted(set(lab0))}
        X, y = d[props].values.astype(float), d.rating_mean.values
        ari, top = [], {c: 0 for c in members}
        for b in range(30):
            rng = np.random.default_rng(SEED + b)
            idx = rng.choice(len(d), int(len(d) * 0.8), replace=False)
            lab, _ = FC.clusters(d.iloc[idx], props, FC.CUT)
            ari.append(adjusted_rand_score(lab0, lab))
            _, dg = FC.grouped_importance(X[idx], y[idx], {c: m for c, m in members.items()}, SEED + b, 4)
            for c in sorted(dg, key=lambda k: -dg[k])[:3]:
                top[c] += 1
        done[key] = {"n": len(d), "ari_mean": float(np.mean(ari)), "ari_min": float(np.min(ari)),
                     "clusters": {str(c): {"features": [props[i] for i in m], "top3_share": top[c] / 30} for c, m in members.items()}}
        p.write_text(json.dumps(done))
        print(key, "ARI", round(np.mean(ari), 2), {c: round(top[c] / 30, 2) for c in top if top[c]}, flush=True)


def vp(sets, props):
    p, done = ck("vp")
    G = {"G": props, "Q": VP.QUALITY, "A": ["age_mesh"]}
    for g, s in plan(sets):
        key = f"{g}|{s}"
        if key in done:
            continue
        d = sets[g][sets[g].src == s].reset_index(drop=True)
        rows = []
        for b in range(30):
            rng = np.random.default_rng(SEED + b)
            sub = d.iloc[rng.choice(len(d), int(len(d) * 0.8), replace=False)].reset_index(drop=True)
            res = VP.partition(sub, G, S.ridge, repeats=1)
            rows.append({"+".join(T): v for T, v in res["comm"].items()} | {"all": res["R2_full"]})
        df = pd.DataFrame(rows)
        done[key] = {"n": len(d), "stats": {c: [float(df[c].mean()), float(df[c].quantile(.025)), float(df[c].quantile(.975)), float(df[c].median())] for c in df}}
        p.write_text(json.dumps(done))
        print(key, len(d), {c: (round(v[0], 3), round(v[3], 3)) for c, v in done[key]["stats"].items()}, flush=True)


if __name__ == "__main__":
    sets, props = load()
    {"curve": curve, "clusters": clusters, "vp": vp}[sys.argv[1]](sets, props)
