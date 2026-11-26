"""All feature clusters in ONE pooled sample, K-fold on all non-lockbox faces at once, with pictures. EXPLORATORY (non-lockbox only).

Requested by the author (2026-10-30): take all the features that we have, not only the block names, cluster them (geometry and the rest separately, because the
non-geometry blocks are small), run K-fold on every non-lockbox person in a single sample, and show the top clusters with diagrams.

Rules fixed before the run:
 - Sample: the six within-source sets of exploration_v1.plan (CM 10k US, AM DataTang, AM Internet, CF 10k US, AF Internet, AF DataTang), non-lockbox, degenerate
   landmark faces dropped, pooled into one table. The target is the judges' mean z-scored INSIDE each set, so the source level gap cannot help the model.
 - Features: geometry = the 42 proportions as they are; the rest = capture (6), age (1), colour/contrast 2a (15), block 2b (11), block 2c (23), symmetry (10),
   dimorphism new features (4), each winsorised at 1/99% and z-scored inside its set, missing values -> set median. The 'rest' is clustered as one table, regardless of its block.
 - Clustering: average linkage on 1 - |Spearman| of the pooled table, cut at 0.5 (as in feature_clusters.py), separately for geometry and for the rest.
 - Model: HistGradientBoosting with the frozen settings, 5-fold x 2 repeats, ONE set of folds for everything; ridge (frozen) next to it. Grouped permutation importance
   per cluster = drop of held-out R2 when the columns of the cluster are shuffled together within the test fold (8 shuffles); also for the two big groups (all geometry, all the rest).
 - Ladder: G; G+Q+A; G+rest; rest only; pooled r and r inside each set.
Expectations (AI guesses): pooled held-out R2 of the full boosting model 0.30-0.45 (r 0.55-0.67); top clusters from geometry: eye shape + brow gap, chin-philtrum, nose-lip;
 from the rest: the contrast cluster (brow, eye, mouth against skin) and a texture or capture cluster; the rest as a group adds <= 0.03 R2 over geometry+capture+age.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block2b_analysis as B2
import block2c_analysis as B3c
import block3_symmetry as B3
import block5_dimorphism as B5
import exploration_v1 as E
import scut_cm_harmony as S
import variance_partition as VP

R = S.ROOT
OUT = R / "results"
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9894", "#0b0b0b", "#52514e"
GEOM_NAME = "геометрия"
CUT = 0.5


def load_pooled():
    sets, props = E.load()
    col = pd.read_csv(R / "data/scut_colour_features.csv")
    f2b = pd.read_csv(R / "data/scut_block2b_features.csv")
    f2c = pd.read_csv(R / "data/scut_block2c_features.csv")
    f3 = pd.read_csv(R / "data/scut_block3_symmetry.csv")
    f5 = pd.read_csv(R / "data/scut_block5_dimorphism.csv")
    blocks = {"капча": VP.QUALITY, "возраст": ["age_mesh"], "цвет/контраст": [c for c in col.columns if c != "file"],
              "2b": [c for c in f2b.columns if c != "file"], "2c": [c for c in f2c.columns if c != "file"],
              "симметрия": [c for c in f3.columns if c != "file"], "димо-новое": B5.NEW}
    rest = [c for v in blocks.values() for c in v]
    tag = {c: k for k, v in blocks.items() for c in v}
    parts = []
    for g, s in E.plan(sets):
        d = sets[g][sets[g].src == s].copy()
        d = d.drop(columns=[c for c in d.columns if c in set(rest) - set(VP.QUALITY) - {"age_mesh"}], errors="ignore")
        for f in (col, f2b, f2c, f3, f5):
            d = d.merge(f, on="file", how="left")
        d = d.reset_index(drop=True)
        for c in rest:
            lo, hi = d[c].quantile([0.01, 0.99])
            v = d[c].clip(lo, hi)
            d[c] = (v.fillna(v.median()) - v.mean()) / v.std()
        d["y"] = (d.rating_mean - d.rating_mean.mean()) / d.rating_mean.std()
        d["setname"] = f"{g} {s}"
        parts.append(d[["file", "setname", "y"] + props + rest])
    return pd.concat(parts, ignore_index=True), props, rest, tag


def cluster(df, cols):
    rho = df[cols].rank().corr().abs().fillna(0).values
    dist = 1 - rho
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(dist, checks=False), "average")
    lab = fcluster(z, CUT, "distance")
    return lab, rho, leaves_list(z)


def name_cluster(cols, idx, rho, tag):
    sub = rho[np.ix_(idx, idx)]
    order = np.argsort(-(sub.sum(axis=1) - 1)) if len(idx) > 1 else [0]
    top = [cols[idx[i]] for i in order[:3]]
    return " + ".join(top) + (f" (+{len(idx) - 3})" if len(idx) > 3 else "")


def grouped_importance(X, y, groups, folds, shuffles=8, seed=S.SEED):
    rng = np.random.default_rng(seed)
    drops = {k: [] for k in groups}
    base = []
    for tr, te in folds:
        m = S.hgb().fit(X[tr], y[tr])
        b = r2_score(y[te], m.predict(X[te]))
        base.append(b)
        for k, idx in groups.items():
            dd = []
            for _ in range(shuffles):
                Xp = X[te].copy()
                Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]
                dd.append(b - r2_score(y[te], m.predict(Xp)))
            drops[k].append(np.mean(dd))
    return float(np.mean(base)), {k: float(np.mean(v)) for k, v in drops.items()}, {k: float(np.std(v)) for k, v in drops.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df, props, rest, tag = load_pooled()
    y = df.y.values
    print(f"pooled sample: n={len(df)}; sets {df.setname.value_counts().to_dict()}; geometry {len(props)}, rest {len(rest)} features", flush=True)
    lab_g, rho_g, ord_g = cluster(df, props)
    lab_r, rho_r, ord_r = cluster(df, rest)
    clusters = {}
    for kind, cols, lab, rho in (("G", props, lab_g, rho_g), ("R", rest, lab_r, rho_r)):
        for c in sorted(set(lab)):
            idx = [i for i in range(len(cols)) if lab[i] == c]
            clusters[f"{kind}{c}"] = {"kind": kind, "cols": [cols[i] for i in idx], "name": name_cluster(cols, idx, rho, tag),
                                      "blocks": sorted({tag.get(cols[i], GEOM_NAME) for i in idx})}
    print(f"clusters: geometry {len(set(lab_g))}, rest {len(set(lab_r))}", flush=True)
    allcols = props + rest
    X = df[allcols].values.astype(float)
    pos = {c: i for i, c in enumerate(allcols)}
    groups = {k: [pos[c] for c in v["cols"]] for k, v in clusters.items()}
    folds = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=S.SEED).split(X))

    # ladder
    sets_names = list(df.setname.unique())
    qa = VP.QUALITY + ["age_mesh"]
    models = {"геометрия": props, "геометрия + капча + возраст": props + qa, "всё": allcols, "без геометрии": rest}
    ladder, preds = {}, {}
    for name, cols in models.items():
        for fam, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
            p = S.oof_predict(fac, df[cols].values.astype(float), y, 2, S.SEED)
            preds[(name, fam)] = p
            row = {"pooled": (float(np.corrcoef(p, y)[0, 1]), r2_score(y, p))}
            for s in sets_names:
                m = (df.setname == s).values
                row[s] = (float(np.corrcoef(p[m], y[m])[0, 1]), r2_score(y[m], p[m]))
            ladder[(name, fam)] = row
            print(f"ladder {name:28s} {fam:8s} pooled r {row['pooled'][0]:.3f} R2 {row['pooled'][1]:.3f} | " + " ".join(f"{s.split()[0]}{s.split()[1][:2]} {row[s][0]:.2f}" for s in sets_names), flush=True)

    # importance
    big = {"вся геометрия": [pos[c] for c in props], "всё остальное": [pos[c] for c in rest]}
    base, drops, sds = grouped_importance(X, y, {**groups, **big}, folds)
    print(f"\nHGB held-out R2 on the pooled sample {base:.3f}; group drops: " + ", ".join(f"{k} {drops[k]:.3f}" for k in big), flush=True)
    imp = sorted(((k, drops[k], sds[k]) for k in groups), key=lambda t: -t[1])
    print("top clusters by grouped permutation importance (R2 drop):")
    for k, v, sd in imp[:25]:
        print(f"  {k:4s} {v:+.4f} (SD over folds {sd:.4f}) n={len(clusters[k]['cols']):2d} blocks {clusters[k]['blocks']}: {clusters[k]['name']}")
    print("sum of cluster drops:", round(sum(max(v, 0) for _, v, _ in imp), 3), flush=True)

    # cluster x set correlations
    rep = {}
    for k, v in clusters.items():
        cols = v["cols"]
        sub = df[cols].values.astype(float)
        sgn = np.sign([spearmanr(sub[:, 0], sub[:, j])[0] or 1 for j in range(sub.shape[1])])
        z = (sub - sub.mean(0)) / sub.std(0)
        rep[k] = (z * sgn).mean(axis=1)
    cs = pd.DataFrame({k: [spearmanr(rep[k][(df.setname == s).values], y[(df.setname == s).values])[0] for s in sets_names] for k in clusters}, index=sets_names).T
    pd.DataFrame({k: {"kind": v["kind"], "n": len(v["cols"]), "blocks": "+".join(v["blocks"]), "name": v["name"], "members": ", ".join(v["cols"]),
                      "drop": drops[k], **{s: cs.loc[k, s] for s in sets_names}} for k, v in clusters.items()}).T.to_csv(OUT / "2026-10-30_all_clusters.csv")

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    top = imp[:20]
    fig, ax = plt.subplots(figsize=(11, 8))
    ypos = np.arange(len(top))[::-1]
    for yp, (k, v, sd) in zip(ypos, top):
        ax.barh(yp, v, color=BLUE if clusters[k]["kind"] == "G" else ORANGE, height=0.55)
        ax.errorbar(v, yp, xerr=sd, color=INK, lw=0.8, capsize=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{clusters[k]['name'][:58]}" for k, _, _ in top], fontsize=8.5, color=INK)
    ax.set_xlabel("на сколько падает R² на незнакомых лицах, если перемешать признаки кластера (чем больше, тем важнее)")
    ax.set_title(f"Топ-20 кластеров, один общий K-fold на {len(df)} лицах вне сейфа (синий — геометрия, оранжевый — остальное)", loc="left", color=INK, fontsize=11)
    ax.text(0.99, 0.04, f"R² всей модели {base:.2f}\nвся геометрия −{drops['вся геометрия']:.2f}\nвсё остальное −{drops['всё остальное']:.2f}", transform=ax.transAxes, ha="right", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-30_fig1_cluster_importance.png", dpi=150)
    plt.close(fig)

    # fig 2 correlation heatmap ordered by cluster
    order_cols = [props[i] for i in ord_g] + [rest[i] for i in ord_r]
    M = df[order_cols].rank().corr().abs().values
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f7fb", BLUE, "#0b2a52"])
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ng = len(props)
    ax.axhline(ng - 0.5, color=ORANGE, lw=1.5)
    ax.axvline(ng - 0.5, color=ORANGE, lw=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Корреляция признаков (по модулю), порядок по кластерам: геометрия — левый верхний квадрат, остальное — правый нижний", loc="left", fontsize=10.5, color=INK)
    ax.text(ng / 2, -2, "ГЕОМЕТРИЯ (42)", ha="center", color=BLUE, fontsize=10)
    ax.text(ng + (len(rest)) / 2, -2, f"ОСТАЛЬНОЕ ({len(rest)})", ha="center", color=ORANGE, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("|Spearman|")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-30_fig2_feature_correlation.png", dpi=150)
    plt.close(fig)

    # fig 3 ladder, dots per set
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = ["pooled"] + sets_names
    cols_m = {"геометрия": BLUE, "геометрия + капча + возраст": GREY, "всё": ORANGE}
    for j, lab in enumerate(labels):
        for i, (name, c) in enumerate(cols_m.items()):
            v = ladder[(name, "boosting")][lab][0]
            ax.scatter(j + (i - 1) * 0.18, v, s=60, color=c, zorder=3, label=name if j == 0 else None)
            ax.text(j + (i - 1) * 0.18, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["все вместе"] + sets_names, fontsize=9)
    ax.set_ylabel("r с оценкой судей (внутри набора)")
    ax.set_ylim(0.3, 0.85)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    ax.set_title("Что добавляется к геометрии (boosting, общие фолды)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-30_fig3_ladder.png", dpi=150)
    plt.close(fig)

    # fig 4 cluster x set heatmap
    topk = [k for k, _, _ in imp[:25]]
    Z = cs.loc[topk, sets_names].values
    div = LinearSegmentedColormap.from_list("div", [BLUE, "#f2f1ee", ORANGE])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(Z, cmap=div, vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(sets_names)))
    ax.set_xticklabels(sets_names, rotation=30, ha="right")
    ax.set_yticks(range(len(topk)))
    ax.set_yticklabels([f"{k}: {clusters[k]['name'][:48]}" for k in topk], fontsize=8, color=INK)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z[i, j]:+.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_title("Связь кластера с оценкой судей внутри каждого набора (Spearman; оранжевый — выше оценка, синий — ниже)", loc="left", fontsize=10.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-30_fig4_cluster_by_set.png", dpi=150)
    plt.close(fig)

    # fig 5 predicted vs actual, small multiples
    p = preds[("всё", "boosting")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), sets_names):
        m = (df.setname == s).values
        ax.scatter(p[m], y[m], s=6, color=BLUE, alpha=0.45, linewidths=0)
        ax.plot([-2.5, 2.5], [-2.5, 2.5], color=GREY, lw=0.8)
        ax.set_title(f"{s}: r {np.corrcoef(p[m], y[m])[0, 1]:.2f}, n={int(m.sum())}", fontsize=10, color=INK, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("предсказание модели (честное, лицо не в обучении)")
    for ax in axes[:, 0]:
        ax.set_ylabel("оценка судей (z внутри набора)")
    fig.suptitle("Предсказание против оценки судей, модель на всех признаках", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-30_fig5_pred_vs_actual.png", dpi=150)
    plt.close(fig)
    print("figures written", flush=True)


if __name__ == "__main__":
    main()
