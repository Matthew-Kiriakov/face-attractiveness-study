"""Image-side audit of the MEBeauty features against SCUT (NO ratings: neither set's labels are read; written before the run).
1. Coverage: missing values per tier-2 column in MEBeauty (aligned crops), trichion found rate against SCUT.
2. Domain shift: for every tier-2 column the median of MEBeauty (all groups pooled, raw units) minus the median of SCUT non-lockbox faces (pooled, raw units), in SCUT IQR units; the columns with the largest shifts are listed. Raw units are compared (not z inside group), because the question is how different the photos and faces are.
3. Outlier flag, calibrated on SCUT first: inside each set (SCUT) / each group (MEBeauty) z = (x - mean) / SD of the raw column; count the tier-2 columns with |z| > 3 per face. The threshold k (flag if count >= k) is the smallest k for which at most 3 % of SCUT faces are flagged; the same k is then applied to MEBeauty, and the flagged share is reported for MEBeauty overall and per ethnicity x gender group.
Expectation (AI, before): trichion found in 90-97 % of MEBeauty faces (SCUT about 93 %?), missing values below 3 % in every column except the ones needing the area above the head; largest shifts in colour and contrast columns (stock photo lighting), up to 1-2 IQR; k between 3 and 6; MEBeauty flagged share 5-12 %, more in the groups with fewer faces to define a mean.
Usage: python mebeauty_feature_audit.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_common as O
import scut_cm_harmony as S

R = S.ROOT


def scut_raw(tier2):
    sets, props = __import__("exploration_v1").load()
    parts = []
    col = [pd.read_csv(R / f"data/{n}.csv") for n in ("scut_colour_features", "scut_block2b_features", "scut_block2c_features", "scut_block5_dimorphism")]
    for g, d in sets.items():
        d = d.copy()
        d["setname"] = g + " " + d.src
        for f in col:
            d = d.merge(f[["file"] + [c for c in f.columns if c not in d.columns and c != "file"]], on="file", how="left")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df, props, rest, tier2 = O.load()
    sc = scut_raw(tier2)
    mb = pd.read_csv(R / "data/external/mebeauty/mebeauty_features.csv")
    mb["group"] = mb.ethnicity + "_" + mb.gender
    print(f"SCUT non-lockbox {len(sc)} faces, MEBeauty {len(mb)} faces, tier-2 columns {len(tier2)}")
    print(f"\n1. trichion found: SCUT {sc.trichion_found.mean():.3f}, MEBeauty {mb.trichion_found.mean():.3f}")
    miss = mb[tier2].isna().mean().sort_values(ascending=False)
    print("   MEBeauty missing share, top:", {k: round(v, 3) for k, v in miss.head(6).items()}, "| columns with any missing:", int((miss > 0).sum()))
    ms = sc[tier2].isna().mean().sort_values(ascending=False)
    print("   SCUT missing share, top:", {k: round(v, 3) for k, v in ms.head(4).items()})

    print("\n2. median shift MEBeauty - SCUT in SCUT IQR units (raw, pooled)")
    iqr = sc[tier2].quantile(.75) - sc[tier2].quantile(.25)
    sh = ((mb[tier2].median() - sc[tier2].median()) / iqr.replace(0, np.nan)).sort_values(key=abs, ascending=False)
    for k, v in sh.head(12).items():
        print(f"   {v:+.2f}  {k}")
    print(f"   columns with |shift| > 0.5 IQR: {int((sh.abs() > 0.5).sum())} of {len(sh)}; > 1.0: {int((sh.abs() > 1.0).sum())}")

    def counts(d, key):
        out = np.zeros(len(d))
        for s in d[key].unique():
            m = (d[key] == s).values
            x = d.loc[m, tier2]
            z = (x - x.mean()) / x.std().replace(0, np.nan)
            out[m] = (z.abs() > 3).sum(axis=1).values
        return out
    sc["cnt"] = counts(sc, "setname")
    mb["cnt"] = counts(mb, "group")
    k = next(k for k in range(1, 40) if (sc.cnt >= k).mean() <= 0.03)
    print(f"\n3. outlier flag: SCUT share with count >= k: " + ", ".join(f"k={j} {100 * (sc.cnt >= j).mean():.1f} %" for j in range(2, 9)))
    print(f"   chosen k = {k} (smallest with at most 3 % of SCUT flagged)")
    print("   MEBeauty share with count >= k: " + ", ".join(f"k={j} {100 * (mb.cnt >= j).mean():.1f} %" for j in range(2, 9)))
    print(f"   MEBeauty flagged at k = {k}: {(mb.cnt >= k).sum()} faces ({100 * (mb.cnt >= k).mean():.1f} %)")
    print(mb.assign(flag=mb.cnt >= k).groupby("group").flag.agg(["size", "mean"]).round(3).to_string())
    mb[["file", "image_id", "cnt"]].assign(flag=mb.cnt >= k, k=k).to_csv(R / "data/external/mebeauty/mebeauty_flags_sd.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
