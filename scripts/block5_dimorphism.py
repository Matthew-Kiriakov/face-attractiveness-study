"""Block 5: sexual dimorphism, from the author's list (screenshot, 2026-10-30): jaw, brow ridge, brow thickness, eyes, face shape, facial hair, hairline, lips,
nose in frontal view (profile is out of scope). EXPLORATORY, non-lockbox only. Usage: python block5_dimorphism.py features | analysis

Directions are the author's "masculine" directions, written before the run:
  jaw       wider jaw to face width (jaw-w up), angular jaw (jaw_turn up), wider chin (chin_w up)
  brow      low brows / small brow-lid gap (brow-gap-l, brow-gap-r down), thick and dense brows (brow_thick, brow_dens up), deep brow ridge shade (ridge_dL up)
  eyes      smaller eyes to face width (eye-face down)
  nose      wider nose to face width in frontal view (nose-face up)
  lips      thinner lips (lip_thick down; lip_thick = (upper lip height + lower lip height) / mouth width)
  shape     wider face (face-wh up)
  facehair  stubble/beard proxies from block 2b (chin_dL, lip_dL, chin_sd up)
  hairline  higher upper third (third-upper up; the author notes that this bot rewards a receded hairline as dimorphic and penalises it as a feature, so both
            readings are kept and only the sign of the observed relation is reported; hairline marker is noisy)
Femininity is the opposite direction. The same index is computed for every group; the dimorphism hypothesis predicts a positive relation with ratings in men
and a negative one in women.
New features measured here: jaw_turn (mean turning angle in degrees of the jaw line at the gonion, both sides), chin_w (chin width / jaw width), lip_thick,
ridge_dL (forehead L minus L of the patch between brow peak and upper lid, in L units).

Index: each component is z-scored inside group x source (non-lockbox statistics), multiplied by +1 / -1 for its direction, the components of a sub-score are
averaged, then M_core = mean of the six sub-scores jaw, brow, eyes, nose, lips, shape; M_full = mean of those six plus facehair and hairline. No fitted weights.

Rules and expectations (before the run):
 - Within every non-lockbox set (>= 200 faces): Spearman of M_core, M_full and each sub-score with the judges' mean; Pearson r of M_core alone against the rating.
 - Gain of the new features (jaw_turn, chin_w, lip_thick, ridge_dL) over G+Q+A+K (ridge, 30 subsamples); entry rule gain >= 0.01 and 2.5th percentile > 0 in at least half of the sets.
 - Expectations (AI guesses): men rho(M_core) +0.05..+0.30, women -0.30..-0.05; brow, jaw and nose sub-scores carry the sign, lips and hairline do not;
   gain of the new features < 0.01 (jaw, nose, eyes, shape, brow gap are already among the 42 proportions); M_core alone r < 0.25 in absolute value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block2a_colour as C
import block2b_analysis as B2
import exploration_v1 as E
import noise_attenuation as N
import scut_cm_harmony as S
import variance_partition as VP

R = S.ROOT
NEW = ["jaw_turn", "chin_w", "lip_thick", "ridge_dL"]


def turn(a, b, c):
    v1, v2 = b - a, c - b
    ang = np.degrees(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2))
    return abs(float(ang))


def features() -> None:
    rows = []
    for g in ("CM", "AM", "CF", "AF"):
        for name, r in N.cache_for(g).items():
            if r.get("error") or not r.get("lm"):
                continue
            lm = np.array(S.fill_iris([None if p is None else list(p) for p in r["lm"]]), dtype=float)
            P = lm[:, :2] * np.array([r["w"], r["h"]])
            ipd = float(np.hypot(*(P[468] - P[473])))
            f = {"file": name}
            f["jaw_turn"] = (turn(P[58], P[172], P[149]) + turn(P[288], P[397], P[378])) / 2
            f["chin_w"] = float(np.hypot(*(P[148] - P[377])) / np.hypot(*(P[172] - P[397])))
            mouth_w = float(np.hypot(*(P[61] - P[291])))
            f["lip_thick"] = float((abs(P[13][1] - P[0][1]) + abs(P[17][1] - P[14][1])) / mouth_w)
            L = C.lab(cv2.imread(str(S.IMAGES / name)))
            fore = float(np.median(L[C.disc(L.shape, *P[151], 0.14 * ipd)][:, 0]))
            ridge = []
            for brow, lid in ((105, 159), (334, 386)):
                cx, cy = (P[brow] + P[lid]) / 2
                ridge.append(float(np.median(L[C.disc(L.shape, cx, cy, 0.07 * ipd)][:, 0])))
            f["ridge_dL"] = fore - float(np.mean(ridge))
            rows.append(f)
        print(g, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(R / "data/scut_block5_dimorphism.csv", index=False)


COMPONENTS = {
    "jaw": [("jaw-w", 1), ("jaw_turn", 1), ("chin_w", 1)],
    "brow": [("brow-gap-l", -1), ("brow-gap-r", -1), ("brow_thick", 1), ("brow_dens", 1), ("ridge_dL", 1)],
    "eyes": [("eye-face", -1)],
    "nose": [("nose-face", 1)],
    "lips": [("lip_thick", -1)],
    "shape": [("face-wh", 1)],
    "facehair": [("chin_dL", 1), ("lip_dL", 1), ("chin_sd", 1)],
    "hairline": [("third-upper", 1)],
}
CORE = ["jaw", "brow", "eyes", "nose", "lips", "shape"]


def analysis() -> None:
    sets, props = E.load()
    new = pd.read_csv(R / "data/scut_block5_dimorphism.csv")
    f2b = pd.read_csv(R / "data/scut_block2b_features.csv")
    f2c = pd.read_csv(R / "data/scut_block2c_features.csv")[["file", "brow_thick", "brow_dens"]]
    col = pd.read_csv(R / "data/scut_colour_features.csv")[["file"] + B2.K]
    base = props + VP.QUALITY + ["age_mesh"] + B2.K
    table = []
    for g, s in E.plan(sets):
        d = sets[g][sets[g].src == s].merge(new, on="file", how="left").merge(f2b, on="file", how="left").merge(f2c, on="file", how="left").merge(col, on="file", how="left").reset_index(drop=True)
        need = sorted({c for v in COMPONENTS.values() for c, _ in v} | set(B2.K) | set(NEW))
        for c in need:
            lo, hi = d[c].quantile([0.01, 0.99])
            v = d[c].clip(lo, hi)
            d[c] = (v.fillna(v.median()) - v.mean()) / v.std()
        sub = {k: np.mean([sg * d[c] for c, sg in v], axis=0) for k, v in COMPONENTS.items()}
        d["M_core"] = np.mean([sub[k] for k in CORE], axis=0)
        d["M_full"] = np.mean([sub[k] for k in CORE + ["facehair", "hairline"]], axis=0)
        y = d.rating_mean.values
        rho = {k: spearmanr(v, y)[0] for k, v in {**sub, "M_core": d.M_core, "M_full": d.M_full}.items()}
        print(f"{g} {s:8s} n={len(d):4d} rho: " + " ".join(f"{k} {v:+.2f}" for k, v in rho.items()) + f"; r(M_core) {np.corrcoef(d.M_core, y)[0, 1]:+.3f}")
        gains = []
        for b in range(30):
            rng = np.random.default_rng(S.SEED + b)
            sb = d.iloc[rng.choice(len(d), int(len(d) * 0.8), replace=False)].reset_index(drop=True)
            gains.append(B2.gain(sb, base, NEW, sb.rating_mean.values)[0])
        table.append((np.mean(gains), np.percentile(gains, 2.5)))
        print(f"      gain of new features {np.mean(gains):+.3f} [{np.percentile(gains, 2.5):+.3f}, {np.percentile(gains, 97.5):+.3f}]", flush=True)
    print("\nEntry rule met in", sum(1 for m, lo in table if m >= 0.01 and lo > 0), "of", len(table), "sets")


if __name__ == "__main__":
    {"features": features, "analysis": analysis}[sys.argv[1]]()
