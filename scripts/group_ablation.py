"""Which non-geometry group carries the gain? Leave-one-group-out from tier 2 and add-one-group to geometry, on the pooled non-lockbox K-fold. EXPLORATORY.

Same sample, target and folds as all_clusters_kfold.py / proxy_ladder.py. Tier 2 = geometry (42) + 28 non-geometry face features, in seven groups declared before the run:
  contrast    brow_L, brow_contrast, eye_contrast, mouth_contrast
  outline     adip_pa, adip_lower, avg_dist
  chin_lip    chin_dL, chin_sd, lip_dL
  brows       brow_thick, brow_dens, brow_arch
  eyes_lids   sclera_show_low, sclera_show_up, dark_dL, dark_da, dark_db, bag_range, crease_dist, crease_depth, lid_dL, ridge_dL
  nose_tip    tip_w, tip_h, tip_shade
  jaw_chin    jaw_turn, chin_w
A) drop of held-out R2 when one group is removed from tier 2 (geometry + all 28); B) gain of held-out R2 when one group is added to geometry alone.
Boosting and ridge (frozen settings), 5-fold x 2 with one set of folds; the interval is a paired bootstrap of the faces (300 resamples) on the same out-of-fold predictions.
Expectations (AI guesses): A) largest drops for contrast (0.01-0.04 in r terms, about 0.02-0.05 in R2) and eyes_lids; chin_lip 0.005-0.02; outline small because geometry covers it
(< 0.01 R2); brows, nose_tip, jaw_chin <= 0.01. B) contrast +0.03-0.06 R2, outline and chin_lip +0.01-0.04, the rest <= 0.02.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import all_clusters_kfold as A
import scut_cm_harmony as S

R, OUT = A.R, A.OUT
CACHE = R / "data/cache_expl/group_ablation.npz"
GROUPS = {
    "контраст": ["brow_L", "brow_contrast", "eye_contrast", "mouth_contrast"],
    "контур лица": ["adip_pa", "adip_lower", "avg_dist"],
    "тон подбородка и губы": ["chin_dL", "chin_sd", "lip_dL"],
    "брови": ["brow_thick", "brow_dens", "brow_arch"],
    "глаза и веки": ["sclera_show_low", "sclera_show_up", "dark_dL", "dark_da", "dark_db", "bag_range", "crease_dist", "crease_depth", "lid_dL", "ridge_dL"],
    "кончик носа": ["tip_w", "tip_h", "tip_shade"],
    "челюсть и подбородок": ["jaw_turn", "chin_w"],
}


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df, props, rest, tag = A.load_pooled()
    y = df.y.values
    allg = [c for v in GROUPS.values() for c in v]
    assert len(allg) == 28 and all(c in rest for c in allg), "tier-2 groups do not match the data"
    saved = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
    variants = {"tier2": props + allg, "geometry": props}
    for g, cols in GROUPS.items():
        variants[f"minus {g}"] = props + [c for c in allg if c not in cols]
        variants[f"geometry + {g}"] = props + cols
    for name, cols in variants.items():
        for fam, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
            key = f"{name}|{fam}"
            if key in saved:
                continue
            saved[key] = S.oof_predict(fac, df[cols].values.astype(float), y, 2, S.SEED)
            np.savez(CACHE, **saved)
            print(f"{key:55s} R2 {r2(y, saved[key]):.4f} r {np.corrcoef(saved[key], y)[0, 1]:.3f}", flush=True)

    rng = np.random.default_rng(20261030)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(300)]

    def delta(a, b, fam):
        pa, pb = saved[f"{a}|{fam}"], saved[f"{b}|{fam}"]
        d0 = r2(y, pa) - r2(y, pb)
        bs = [r2(y[i], pa[i]) - r2(y[i], pb[i]) for i in idx]
        return d0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    rows = {}
    print("\nA) R2 lost when the group is removed from tier 2 (boosting / ridge), 95% paired bootstrap")
    for g in GROUPS:
        rows[("minus", g)] = {fam: delta("tier2", f"minus {g}", fam) for fam in ("boosting", "ridge")}
        b, r = rows[("minus", g)]["boosting"], rows[("minus", g)]["ridge"]
        print(f"  {g:24s} boosting {b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}] | ridge {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]")
    print("\nB) R2 gained when the group is added to geometry alone")
    for g in GROUPS:
        rows[("plus", g)] = {fam: delta(f"geometry + {g}", "geometry", fam) for fam in ("boosting", "ridge")}
        b, r = rows[("plus", g)]["boosting"], rows[("plus", g)]["ridge"]
        print(f"  {g:24s} boosting {b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}] | ridge {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]")
    tot_b = delta("tier2", "geometry", "boosting")
    tot_r = delta("tier2", "geometry", "ridge")
    print(f"\nall 28 together over geometry: boosting {tot_b[0]:+.4f} [{tot_b[1]:+.4f}, {tot_b[2]:+.4f}] | ridge {tot_r[0]:+.4f} [{tot_r[1]:+.4f}, {tot_r[2]:+.4f}]")
    print(f"sum of single-group gains (boosting): {sum(rows[('plus', g)]['boosting'][0] for g in GROUPS):+.4f}")

    GREY, BLUE, ORANGE, INK, MUTED = A.GREY, A.BLUE, A.ORANGE, A.INK, A.MUTED
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    names = list(GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, kind, title in ((axes[0], "plus", "добавить группу к одной геометрии: прирост R²"), (axes[1], "minus", "убрать группу из полного набора: потеря R²")):
        order = sorted(names, key=lambda g: rows[("plus", g)]["boosting"][0])
        yp = np.arange(len(order))
        v = np.array([rows[(kind, g)]["boosting"] for g in order])
        ax.barh(yp, v[:, 0], color=ORANGE if kind == "plus" else BLUE, height=0.5)
        ax.errorbar(v[:, 0], yp, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=INK, lw=0.9, capsize=2)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_yticks(yp)
        ax.set_yticklabels([f"{g} ({len(GROUPS[g])})" for g in order], color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("ΔR² на незнакомых лицах (boosting), усы — 95% интервал")
    fig.suptitle("Какая группа негеометрических признаков что несёт (3737 лиц вне сейфа, один K-fold)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-30_fig8_group_ablation.png", dpi=150)
    plt.close(fig)
    print("done", flush=True)


if __name__ == "__main__":
    main()
