"""Block 6, 'angularity' (the last block of the author's bot): jaw definition, cheek dryness, cheekbone prominence, chin definition, submental zone. PRE-REGISTERED 2026-10-31.
Usage: python angularity.py features | analysis. EXPLORATORY, non-lockbox faces in the analysis. 2D only (no depth in the caches), so 'definition' is the sharpness of the light/shade edge.

Features (MediaPipe landmark indices; ipd = distance between the iris centres; L = Lab lightness of the photo; skin_L = median L of the two cheek discs 50 and 280):
  jaw_edge    along the jaw arcs (left 172,136,150,149 and right 397,365,379,378) the profile of L is sampled along the outward radial direction from -0.25 to +0.25 ipd
              (11 samples); per contour point the steepest step between neighbouring samples after a 3-point smoothing; median over the arc points; divided by skin_L
  jaw_step    mean L of the inner samples (-0.20..-0.08 ipd) minus mean L of the outer samples (+0.08..+0.20 ipd), median over the arc points, divided by skin_L (sign kept: face lighter than neck is positive)
  chin_edge, chin_step   the same two measures on the chin arc 176,148,152,377,400
  cheek_hollow   (L at the cheek apple 50 or 280, radius 0.10 ipd) minus (L at the lower cheek: mean of landmarks 207 and 213, or 427 and 433, radius 0.10 ipd), divided by skin_L, mean of both sides
                 (a hollow cheek is darker under the cheekbone)
  jaw_angle   angle at landmark 172 (or 397) between the directions to 58 (or 288) and to 148 (or 377), degrees, mean of both sides (small = angular jaw)
  chin_angle  angle at landmark 152 between the directions to 176 and 400, degrees (small = pointed chin)
  zyg_jaw     distance 116-345 divided by distance 58-288 (cheekbone width over jaw width at the gonion)
Submental zone: not measurable from the front; adip_lower (existing) is the only proxy.

Expectations (AI guesses, before any look at the numbers):
 1) jaw_edge and chin_edge correlate 0.3-0.7 with each other; their within-set Spearman with the judges' z is below |0.15|; they correlate with existing jaw_turn, chin_w below |0.3|.
 2) chin_step correlates with chin_dL above |0.3| (the author's guess that chin definition is the chin shading).
 3) cheek_hollow correlates with adip_pa at -0.3 or stronger (lean face, darker cheek hollow).
 4) jaw_angle correlates with jaw-w (existing proportion) above |0.3|.
 5) the eight together add less than 0.01 held-out R2 to tier 2 (geometry z + 28), and no single one adds more than 0.005 with the lower end of its interval above zero.
Analysis: mean within-set Spearman with the judges' z and with existing features; boosting and ridge R2 gain over tier 2 for all eight together and one at a time, 300-resample paired bootstrap, same folds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import all_clusters_kfold as A
import block2a_colour as C
import group_ablation as G
import noise_attenuation as N
import scut_cm_harmony as S

R = S.ROOT
CSV = R / "data/scut_block6_angularity.csv"
NEW = ["jaw_edge", "jaw_step", "chin_edge", "chin_step", "cheek_hollow", "jaw_angle", "chin_angle", "zyg_jaw"]
OFFS = np.linspace(-0.25, 0.25, 11)


def ang(a, b, c):
    u, v = b - a, c - a
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))


def arc_feats(Lc, P, ids, centre, ipd, skin):
    prof = []
    for i in ids:
        d = P[i] - centre
        n = d / (np.linalg.norm(d) + 1e-9)
        xs = P[i][0] + n[0] * OFFS * ipd
        ys = P[i][1] + n[1] * OFFS * ipd
        prof.append(map_coordinates(Lc, [ys, xs], order=1, mode="nearest"))
    prof = np.array(prof)
    sm = np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3, mode="same"), 1, prof)[:, 1:-1]
    edge = np.median(np.abs(np.diff(sm, axis=1)).max(axis=1)) / skin
    inner, outer = prof[:, (OFFS >= -0.201) & (OFFS <= -0.079)].mean(axis=1), prof[:, (OFFS >= 0.079) & (OFFS <= 0.201)].mean(axis=1)
    return float(edge), float(np.median(inner - outer) / skin)


def features() -> None:
    rows = []
    for g in ("CM", "AM", "CF", "AF"):
        for name, r in N.cache_for(g).items():
            if r.get("error") or not r.get("lm"):
                continue
            lm = np.array(S.fill_iris([None if p is None else list(p) for p in r["lm"]]), dtype=float)
            w, h = r["w"], r["h"]
            P = lm[:, :2] * np.array([w, h])
            ipd = float(np.hypot(*(P[468] - P[473])))
            Lc = C.lab(cv2.imread(str(S.IMAGES / name)))[:, :, 0].astype(float)
            disc = lambda i, rad=0.10: float(np.median(Lc[C.disc(Lc.shape + (3,), *P[i], rad * ipd)])) if len(Lc[C.disc(Lc.shape + (3,), *P[i], rad * ipd)]) else np.nan
            skin = float(np.median(np.concatenate([Lc[C.disc(Lc.shape + (3,), *P[i], 0.14 * ipd)] for i in (50, 280)])))
            centre = P[1]
            f = {"file": name}
            f["jaw_edge"], f["jaw_step"] = arc_feats(Lc, P, [172, 136, 150, 149, 397, 365, 379, 378], centre, ipd, skin)
            f["chin_edge"], f["chin_step"] = arc_feats(Lc, P, [176, 148, 152, 377, 400], centre, ipd, skin)
            hol = []
            for apple, lo in ((50, (207, 213)), (280, (427, 433))):
                mid = (P[lo[0]] + P[lo[1]]) / 2
                v = Lc[C.disc(Lc.shape + (3,), mid[0], mid[1], 0.10 * ipd)]
                hol.append((disc(apple) - float(np.median(v))) / skin if len(v) else np.nan)
            f["cheek_hollow"] = float(np.nanmean(hol))
            f["jaw_angle"] = (ang(P[172], P[58], P[148]) + ang(P[397], P[288], P[377])) / 2
            f["chin_angle"] = ang(P[152], P[176], P[400])
            f["zyg_jaw"] = float(np.linalg.norm(P[116] - P[345]) / np.linalg.norm(P[58] - P[288]))
            rows.append(f)
        print(g, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)


def analysis() -> None:
    from scipy.stats import spearmanr
    sys.stdout.reconfigure(encoding="utf-8")
    import scut_cm_harmony as S2
    df, props, rest, tag = A.load_pooled()
    df = df.merge(pd.read_csv(CSV), on="file", how="left")
    for s in df.setname.unique():
        m = df.setname == s
        for c in props + NEW:
            lo, hi = df.loc[m, c].quantile([0.01, 0.99])
            v = df.loc[m, c].clip(lo, hi)
            df.loc[m, c] = (v.fillna(v.median()) - v.mean()) / v.std()
    y = df.y.values
    sets_ = df.setname.unique()

    def within(a, b):
        return float(np.nanmean([spearmanr(df.loc[df.setname == s, a], df.loc[df.setname == s, b], nan_policy="omit")[0] for s in sets_]))

    print("mean within-set Spearman with the judges' z")
    for c in NEW:
        print(f"  {c:13s} {within(c, 'y'):+.2f}   per set " + " ".join(f"{s.split()[0]}{s.split()[1][:2]} {spearmanr(df.loc[df.setname == s, c], df.loc[df.setname == s, 'y'])[0]:+.2f}" for s in sets_))
    print("\nwith existing features")
    for a, b in (("jaw_edge", "chin_edge"), ("jaw_edge", "jaw_turn"), ("jaw_edge", "chin_w"), ("chin_edge", "chin_w"), ("chin_step", "chin_dL"), ("chin_edge", "chin_dL"), ("cheek_hollow", "adip_pa"),
                 ("cheek_hollow", "adip_lower"), ("jaw_angle", "jaw-w"), ("jaw_angle", "jaw_turn"), ("zyg_jaw", "jaw-w"), ("chin_angle", "chin_w"), ("jaw_step", "chin_dL")):
        print(f"  {a:13s} vs {b:12s} {within(a, b):+.2f}")
    allg = [c for v in G.GROUPS.values() for c in v]
    base = props + allg
    rng = np.random.default_rng(20261031)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(300)]
    cache = {}

    def run(name, cols):
        for fam, fac in (("boosting", S2.hgb), ("ridge", S2.ridge)):
            cache[f"{name}|{fam}"] = S2.oof_predict(fac, df[cols].values.astype(float), y, 2, S2.SEED)

    run("tier2", base)
    run("tier2 + all 8", base + NEW)
    for c in NEW:
        run(f"tier2 + {c}", base + [c])
    print("\nR2 gain over tier 2 (boosting [95%] | ridge [95%])")
    for name in ["tier2 + all 8"] + [f"tier2 + {c}" for c in NEW]:
        out = []
        for fam in ("boosting", "ridge"):
            pa, pb = cache[f"{name}|{fam}"], cache[f"tier2|{fam}"]
            bs = [G.r2(y[i], pa[i]) - G.r2(y[i], pb[i]) for i in idx]
            out.append(f"{G.r2(y, pa) - G.r2(y, pb):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
        print(f"  {name:22s}", " | ".join(out))
    print("done", flush=True)


if __name__ == "__main__":
    {"features": features, "analysis": analysis}[sys.argv[1]]()
