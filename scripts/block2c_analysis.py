"""Block 2c (the author's list: eye whites, under-eye, lids, nose tip, skin texture, brows): unique gain over the existing blocks.
EXPLORATORY, non-lockbox only. Same setup as block2b_analysis.py (written before any rating was joined to these features).

 - Features: data/scut_block2c_features.csv, winsorised 1/99% and standardised inside group x source; NaN -> median.
 - Base = G + Q + A + K (contrast sub-block of 2a). Ridge, 5-fold x 1, 30 subsamples of 80%; gain = held-out R2 with minus without.
 - Pre-declared sub-blocks as in block2c_features.py. Entry rule: gain >= 0.01 and 2.5th percentile > 0 in at least half of the sets.
Expectations (AI guesses, before the run): sclera features weak, |rho| < 0.10, gain ~0 (landmark noise on the lid is the largest);
 dark circles / under-eye: rho(dark_dL) -0.25..-0.05; lid: crease_depth matters only in the Asian sets (|rho| 0.10-0.20 there, ~0 in European sets);
 nose tip: ~0; texture and spots: rho -0.25..-0.05 but a likely duplicate of sharpness (Q) so gain < 0.01;
 brows: brow_dens and brow_thick positive for men (rho 0.05-0.20), ~0 for women; gain of brows 0.005-0.02.
 Whole block gain over the base 0.005-0.03; entry rule probably met in <= 2 of 6 sets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block2b_analysis as B
import exploration_v1 as E
import scut_cm_harmony as S
import variance_partition as VP

R = S.ROOT
PARTS = {
    "eye_white": ["sclera_show_low", "sclera_show_up", "sclera_L", "sclera_a", "sclera_b"],
    "under_eye": ["dark_dL", "dark_da", "dark_db", "bag_range"],
    "lid": ["crease_dist", "crease_depth", "lid_dL"],
    "nose_tip": ["tip_w", "tip_h", "tip_shade"],
    "texture": ["tex_L", "tex_a", "spot_red", "spot_dark", "fore_tex_L"],
    "brows": ["brow_dens", "brow_thick", "brow_arch"],
}
ALL = [c for v in PARTS.values() for c in v]


def main() -> None:
    sets, props = E.load()
    b2 = pd.read_csv(R / "data/scut_block2c_features.csv")
    col = pd.read_csv(R / "data/scut_colour_features.csv")[["file"] + B.K]
    base = props + VP.QUALITY + ["age_mesh"] + B.K
    names = list(PARTS) + ["whole 2c"]
    table = []
    for g, s in E.plan(sets):
        d = sets[g][sets[g].src == s].merge(b2, on="file", how="left").merge(col, on="file", how="left").reset_index(drop=True)
        for c in ALL + B.K:
            lo, hi = d[c].quantile([0.01, 0.99])
            v = d[c].clip(lo, hi)
            d[c] = (v.fillna(v.median()) - v.mean()) / v.std()
        rho = {c: spearmanr(d[c], d.rating_mean)[0] for c in ALL}
        print(f"{g} {s:8s} n={len(d):4d} rho: " + " ".join(f"{c} {v:+.2f}" for c, v in rho.items()))
        res = {k: [] for k in names}
        r0s = []
        for b in range(30):
            rng = np.random.default_rng(S.SEED + b)
            sub = d.iloc[rng.choice(len(d), int(len(d) * 0.8), replace=False)].reset_index(drop=True)
            y = sub.rating_mean.values
            for k, cols in PARTS.items():
                res[k].append(B.gain(sub, base, cols, y)[0])
            gw, r0 = B.gain(sub, base, ALL, y)
            res["whole 2c"].append(gw)
            r0s.append(r0)
        print(f"      base R2 {np.mean(r0s):.3f}; gain " + "; ".join(f"{k} {np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]" for k, v in res.items()), flush=True)
        table.append({k: (np.mean(v), np.percentile(v, 2.5)) for k, v in res.items()})
    print("\nEntry rule (gain >= 0.01 and 2.5th percentile > 0), sets met out of", len(table))
    for k in names:
        print(f"  {k:12s} {sum(1 for t in table if t[k][0] >= 0.01 and t[k][1] > 0)}")


if __name__ == "__main__":
    main()
