"""Block 3 (symmetry) from landmarks: features for all faces, then the analysis on the non-lockbox faces only. EXPLORATORY.
Usage: python block3_symmetry.py features | analysis

Method: the midline is the total-least-squares line through the axis points (10, 168, 6, 1, 2, 0, 17, 152). Each left point is compared with
the mirror image (reflection across the midline) of its right partner; the distance is divided by the inter-pupillary distance (IPD).
Features:
  sym_rms        RMS of the pair distances over all pairs (global fluctuating asymmetry)
  sym_eyes, sym_brows, sym_mouth, sym_nose, sym_jaw    the same per region
  sym_h, sym_v   RMS of the component across the midline (width) and along it (height offset between the sides)
  sym_axis       RMS distance of the axis points from the midline / IPD (nose, chin, lip centre off-axis)
  sym_tilt       SD (degrees) of the angles of the lines through pairs (eyes, pupils, brows, mouth corners, cheek, jaw) vs the midline normal
Pose is a confound (a turned head looks asymmetric): roll_abs and half_dev (from block2b_features.csv) join the base in the analysis.

Rules and expectations fixed before the run:
 - Same machinery as block2b_analysis.py: base = G + Q + A + K (contrast) + pose; ridge, 5-fold x 1, 30 subsamples of 80%; features winsorised 1/99% and
   standardised inside group x source; non-lockbox sets with >= 200 faces; entry rule gain >= 0.01 and 2.5th percentile > 0 in at least half of the sets.
 - Pre-declared sub-blocks: global (sym_rms), regions (sym_eyes, sym_brows, sym_mouth, sym_nose, sym_jaw), components (sym_h, sym_v), axis (sym_axis), tilt (sym_tilt).
Expectations (AI guesses): rho(sym_rms) in -0.20..0 (more asymmetry = lower rating), weak because the detector's own error (~0.02-0.03 IPD) is as large as real
 asymmetry and the face mesh is regularised towards a symmetric template; gain of the whole block < 0.01 in most sets (entry rule met in <= 1 of 6);
 the largest single signal, if any, in sym_axis or sym_mouth.
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
import noise_attenuation as N
import scut_cm_harmony as S
import variance_partition as VP

R = S.ROOT
AXIS = [10, 168, 6, 1, 2, 0, 17, 152]
REGIONS = {
    "eyes": [(33, 263), (133, 362), (159, 386), (145, 374), (468, 473)],
    "brows": [(46, 276), (107, 336), (105, 334), (70, 300), (66, 296)],
    "mouth": [(61, 291), (37, 267), (91, 321), (84, 314)],
    "nose": [(98, 327), (48, 278), (45, 275), (129, 358)],
    "jaw": [(234, 454), (172, 397), (58, 288), (132, 361)],
}
TILT = [(33, 263), (468, 473), (46, 276), (61, 291), (234, 454), (172, 397)]
PARTS = {
    "global": ["sym_rms"],
    "regions": ["sym_eyes", "sym_brows", "sym_mouth", "sym_nose", "sym_jaw"],
    "components": ["sym_h", "sym_v"],
    "axis": ["sym_axis"],
    "tilt": ["sym_tilt"],
}
ALL = [c for v in PARTS.values() for c in v]
POSE = ["roll_abs", "half_dev"]


def feats(P: np.ndarray) -> dict:
    ipd = float(np.hypot(*(P[468] - P[473])))
    A = P[AXIS]
    c = A.mean(axis=0)
    u, s, vt = np.linalg.svd(A - c)
    d = vt[0] / np.linalg.norm(vt[0])  # along the midline
    n = np.array([-d[1], d[0]])  # across
    def comp(p):  # (across, along) coordinates relative to the centroid
        q = p - c
        return np.array([q @ n, q @ d])
    f, all_d, hh, vv = {}, [], [], []
    for reg, pairs in REGIONS.items():
        ds = []
        for a, b in pairs:
            ca, cb = comp(P[a]), comp(P[b])
            diff = np.array([ca[0] + cb[0], ca[1] - cb[1]])  # mirror: across-coordinate flips sign
            ds.append(np.hypot(*diff) / ipd)
            hh.append(diff[0] / ipd)
            vv.append(diff[1] / ipd)
        f[f"sym_{reg}"] = float(np.sqrt(np.mean(np.square(ds))))
        all_d += ds
    f["sym_rms"] = float(np.sqrt(np.mean(np.square(all_d))))
    f["sym_h"], f["sym_v"] = float(np.sqrt(np.mean(np.square(hh)))), float(np.sqrt(np.mean(np.square(vv))))
    f["sym_axis"] = float(np.sqrt(np.mean([(comp(P[i])[0] / ipd) ** 2 for i in AXIS])))
    ang = []
    for a, b in TILT:
        ca, cb = comp(P[a]), comp(P[b])
        v = cb - ca
        ang.append(np.degrees(np.arctan2(v[1], v[0])))
    ang = np.array(ang)
    ang = np.where(np.abs(ang) > 90, ang - 180 * np.sign(ang), ang)
    f["sym_tilt"] = float(np.std(ang))
    return f


def make_features() -> None:
    rows = []
    for g in ("CM", "AM", "CF", "AF"):
        for name, r in N.cache_for(g).items():
            if r.get("error") or not r.get("lm"):
                continue
            lm = np.array(S.fill_iris([None if p is None else list(p) for p in r["lm"]]), dtype=float)
            rows.append({"file": name, **feats(lm[:, :2] * np.array([r["w"], r["h"]]))})
        print(g, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(R / "data/scut_block3_symmetry.csv", index=False)


def analysis() -> None:
    sets, props = E.load()
    sy = pd.read_csv(R / "data/scut_block3_symmetry.csv")
    other = pd.read_csv(R / "data/scut_block2b_features.csv")[["file"] + POSE]
    col = pd.read_csv(R / "data/scut_colour_features.csv")[["file"] + B.K]
    base = props + VP.QUALITY + ["age_mesh"] + B.K + POSE
    names = list(PARTS) + ["whole 3"]
    table = []
    for g, s in E.plan(sets):
        d = sets[g][sets[g].src == s].merge(sy, on="file", how="left").merge(other, on="file", how="left").merge(col, on="file", how="left").reset_index(drop=True)
        for c in ALL + B.K + POSE:
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
            res["whole 3"].append(gw)
            r0s.append(r0)
        print(f"      base R2 {np.mean(r0s):.3f}; gain " + "; ".join(f"{k} {np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]" for k, v in res.items()), flush=True)
        table.append({k: (np.mean(v), np.percentile(v, 2.5)) for k, v in res.items()})
    print("\nEntry rule (gain >= 0.01 and 2.5th percentile > 0), sets met out of", len(table))
    for k in names:
        print(f"  {k:12s} {sum(1 for t in table if t[k][0] >= 0.01 and t[k][1] > 0)}")


if __name__ == "__main__":
    {"features": make_features, "analysis": analysis}[sys.argv[1]]()
