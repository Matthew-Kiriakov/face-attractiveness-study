"""Priority 3: cluster the 42 proportions, then importance per cluster vs per single feature. EXPLORATORY.

Rules fixed before the run: average linkage on distance 1 - |Spearman rho| computed on CM (all 750);
cut at distance 0.5 (clusters have mean |rho| >= 0.5 inside). Importance = grouped permutation importance
of the boosting model on held-out folds (5-fold x 2 repeats, seed from S.SEED): drop in held-out R2 when the
columns of a cluster are shuffled together (10 shuffles). Computed within one source at a time, so the source
level gap cannot drive it: CM 10k US (n=598) and AM DataTang (n=1342). Clusters are those of CM, also reported for AM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_cm_harmony as S

R = S.ROOT
CUT = 0.5


def clusters(df: pd.DataFrame, props: list[str], cut: float):
    rho = df[props].rank().corr().abs().fillna(0).values
    dist = 1 - rho
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(dist, checks=False), "average")
    lab = fcluster(z, cut, "distance")
    return lab, rho


def grouped_importance(X: np.ndarray, y: np.ndarray, groups: dict[str, list[int]], seed: int, shuffles: int = 10):
    rng = np.random.default_rng(seed)
    drops = {k: [] for k in groups}
    base = []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed).split(X):
        m = S.hgb().fit(X[tr], y[tr])
        b = r2_score(y[te], m.predict(X[te]))
        base.append(b)
        for k, idx in groups.items():
            d = []
            for _ in range(shuffles):
                Xp = X[te].copy()
                perm = rng.permutation(len(te))
                Xp[:, idx] = Xp[perm][:, idx]
                d.append(b - r2_score(y[te], m.predict(Xp)))
            drops[k].append(np.mean(d))
    return float(np.mean(base)), {k: float(np.mean(v)) for k, v in drops.items()}


def main() -> None:
    cm = pd.read_csv(S.CSV_PATH).merge(pd.read_csv(R / "data/scut_cm_source.csv"), on="file")
    cols = list(cm.columns)
    props = cols[cols.index("face-wh"): cols.index("ala-jaw") + 1]
    am = pd.read_csv(R / "data/scut_am_harmony.csv")

    lab, rho = clusters(cm, props, CUT)
    n = lab.max()
    print(f"42 features -> {n} clusters at distance {CUT} (mean |rho| >= {1 - CUT} inside); CM-based")
    members = {f"C{c}": [i for i in range(len(props)) if lab[i] == c] for c in range(1, n + 1)}
    for c, idx in members.items():
        inner = rho[np.ix_(idx, idx)]
        mean_in = (inner.sum() - len(idx)) / max(1, len(idx) * (len(idx) - 1))
        print(f"  {c} (n={len(idx)}, mean |rho| inside {mean_in:.2f}): {[props[i] for i in idx]}")
    # stability: same cut on AM DataTang
    lab_a, _ = clusters(am[am.src == "DataTang"], props, CUT)
    from sklearn.metrics import adjusted_rand_score
    print(f"  agreement of clusters CM vs AM DataTang (adjusted Rand): {adjusted_rand_score(lab, lab_a):.2f}; AM clusters count {lab_a.max()}")

    for name, d in (("CM 10k US", cm[cm.src == "10k_US"]), ("AM DataTang", am[am.src == "DataTang"])):
        X, y = d[props].values.astype(float), d.rating_mean.values
        single = {props[i]: [i] for i in range(len(props))}
        b, di = grouped_importance(X, y, single, S.SEED, 6)
        _, dg = grouped_importance(X, y, members, S.SEED, 10)
        print(f"\n[{name}] n={len(d)}, held-out R2 of boosting {b:.3f}")
        top_s = sorted(di.items(), key=lambda kv: -kv[1])[:8]
        print("  single features, top by R2 drop:", [(k, round(v, 3)) for k, v in top_s])
        print("  sum of single-feature drops:", round(sum(max(v, 0) for v in di.values()), 3))
        top_g = sorted(dg.items(), key=lambda kv: -kv[1])
        print("  clusters by R2 drop:", [(k, round(v, 3), [props[i] for i in members[k]][:4]) for k, v in top_g if v > 0.003])
        print("  sum of cluster drops:", round(sum(max(v, 0) for v in dg.values()), 3))


if __name__ == "__main__":
    main()
