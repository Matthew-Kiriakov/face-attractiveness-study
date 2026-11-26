"""Block 2b (averageness, adiposity, beard, expression, pose): unique gain over the existing blocks. EXPLORATORY, non-lockbox only.

Rules and expectations fixed before the run (written 2026-10-29 before any rating was joined to these features):
 - Features: data/scut_block2b_features.csv (block2b_features.py), winsorised at the 1st/99th percentile and standardised inside each group x source.
 - Sets: group x source with >= 200 non-lockbox faces; degenerate-landmark faces dropped (as exploration_v1.py).
 - Base = G (42 proportions) + Q (capture) + A (age) + K (the contrast sub-block of 2a: brow_L, brow_contrast, eye_contrast, mouth_contrast).
   Also reported over G+Q+A only. Ridge, 5-fold x 1, 30 subsamples (80%). Gain = held-out R2 with the block minus without it.
 - Pre-declared sub-blocks: shape (avg_dist), adiposity (adip_pa, adip_lower), beard (chin_dL, chin_sd, lip_dL), expression (smile, mouth_open, teeth),
   pose (roll_abs, half_dev). Whole block 2b = all of them.
 - Entry rule (RESEARCH.md 1d): gain over the base >= 0.01 and 2.5th percentile > 0 in at least half of the sets.
Expectations (AI guesses): rho(avg_dist) in -0.30..-0.10 (further from the mean = lower rating), gain of shape 0.01-0.03; adiposity |rho| < 0.15, gain < 0.01;
 beard |rho| < 0.15 in men, ~0 in women, gain < 0.01; smile rho +0.10..+0.25 in all groups and the largest gain of the block (>= 0.01 in at least 3 sets);
 mouth_open and teeth weak; pose |rho| < 0.10, gain ~0. Whole block gain 0.01-0.05, entry rule probably met through expression and shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exploration_v1 as E
import scut_cm_harmony as S
import variance_partition as VP

R = S.ROOT
K = ["brow_L", "brow_contrast", "eye_contrast", "mouth_contrast"]
PARTS = {
    "shape": ["avg_dist"],
    "adiposity": ["adip_pa", "adip_lower"],
    "beard": ["chin_dL", "chin_sd", "lip_dL"],
    "expression": ["smile", "mouth_open", "teeth"],
    "pose": ["roll_abs", "half_dev"],
}
ALL = [c for v in PARTS.values() for c in v]


def gain(sub, base, extra, y):
    r0 = VP.r2(y, S.oof_predict(S.ridge, sub[base].values.astype(float), y, 1, S.SEED))
    r1 = VP.r2(y, S.oof_predict(S.ridge, sub[base + extra].values.astype(float), y, 1, S.SEED))
    return r1 - r0, r0


def main() -> None:
    sets, props = E.load()
    b2 = pd.read_csv(R / "data/scut_block2b_features.csv")
    col = pd.read_csv(R / "data/scut_colour_features.csv")[["file"] + K]
    base_gqa = props + VP.QUALITY + ["age_mesh"]
    base = base_gqa + K
    names = list(PARTS) + ["whole 2b"]
    table = []
    for g, s in E.plan(sets):
        d = sets[g][sets[g].src == s].merge(b2, on="file", how="left").merge(col, on="file", how="left").reset_index(drop=True)
        for c in ALL + K:
            lo, hi = d[c].quantile([0.01, 0.99])
            v = d[c].clip(lo, hi)
            d[c] = (v.fillna(v.median()) - v.mean()) / v.std()
        rho = {c: spearmanr(d[c], d.rating_mean)[0] for c in ALL}
        print(f"{g} {s:8s} n={len(d):4d} rho: " + " ".join(f"{c} {v:+.2f}" for c, v in rho.items()))
        res = {k: [] for k in names + ["whole 2b over GQA"]}
        r0s = []
        for b in range(30):
            rng = np.random.default_rng(S.SEED + b)
            sub = d.iloc[rng.choice(len(d), int(len(d) * 0.8), replace=False)].reset_index(drop=True)
            y = sub.rating_mean.values
            for k, cols in PARTS.items():
                res[k].append(gain(sub, base, cols, y)[0])
            gw, r0 = gain(sub, base, ALL, y)
            res["whole 2b"].append(gw)
            r0s.append(r0)
            res["whole 2b over GQA"].append(gain(sub, base_gqa, ALL, y)[0])
        print(f"      base R2 {np.mean(r0s):.3f}; gain " + "; ".join(f"{k} {np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]" for k, v in res.items()), flush=True)
        table.append({k: (np.mean(v), np.percentile(v, 2.5)) for k, v in res.items()})
    print("\nEntry rule (gain >= 0.01 and 2.5th percentile > 0), sets met out of", len(table))
    for k in names:
        print(f"  {k:12s} {sum(1 for t in table if t[k][0] >= 0.01 and t[k][1] > 0)}")


if __name__ == "__main__":
    main()
