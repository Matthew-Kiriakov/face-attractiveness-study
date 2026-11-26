"""Geometry (42) and the 28 face features of tier 2 on the MEBeauty aligned 256 px crops (IMAGE SIDE ONLY: no rating, no label is read; written before the run).
Same code as the SCUT pipeline: landmarks from the browser detector (js/detect_batch.js, group MEB -> data/external/mebeauty/mebeauty_landmarks_web.jsonl), geometry from js/metrics.js via node (scripts/measure_batch.mjs), the colour / contrast block (block2a_colour.one), block 2b (avg_dist, adip_pa, adip_lower, chin_dL, chin_sd, lip_dL), block 2c (brows, eyes/lids, nose tip), block 5 (jaw_turn, chin_w, ridge_dL).
avg_dist uses the mean shape of each gender x ethnicity group computed on cv_fold 1-4 faces only (fold 0 stays out of every group statistic).
Output: data/external/mebeauty/mebeauty_features.csv (image id = file stem, gender, ethnicity, cv_fold from manifest.csv, trichion_found, roll_deg, half_ratio, 42 geometry columns, all block features).
Known domain differences, not corrected: 256 px instead of 350 px (patch radii scale with IPD, but blur sigma, kernels and rounding are in pixels), stock photos, aligned crops.
Usage: python mebeauty_features.py"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block2a_colour as C
import block2b_features as B2
import block2c_features as B3
import noise_attenuation as N
import scut_cm_harmony as S

R = S.ROOT
MB = R / "data/external/mebeauty"
IMG = MB / "images_aligned"
LM = MB / "mebeauty_landmarks_web.jsonl"
MEAS = MB / "mebeauty_measured.json"


def turn(a, b, c):
    v1, v2 = b - a, c - b
    return abs(float(np.degrees(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2))))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    subprocess.run([str(S.NODE), str(S.MEASURE_JS), str(LM), str(MEAS)], check=True)
    meas = json.loads(MEAS.read_text(encoding="utf-8"))
    ids = [m["id"] for m in meas["catalog"]]
    geo_cols = list(pd.read_csv(R / "data/scut_af_harmony.csv", nrows=1).columns)
    geo_cols = geo_cols[geo_cols.index("face-wh"): geo_cols.index("ala-jaw") + 1]
    assert all(c in ids for c in geo_cols), "geometry ids do not match"
    man = pd.read_csv(MB / "manifest.csv")
    man["file"] = man.image_id + "." + man.ext
    man["group"] = man.ethnicity + "_" + man.gender
    info = man.set_index("file")
    rows = {}
    for r in meas["rows"]:
        if r.get("error") or not r.get("values"):
            continue
        d = {"file": r["file"], "trichion_found": int(r["trichion_found"]), "roll_deg": r["roll"], "half_ratio": r["half"]}
        d.update({c: r["values"].get(c) for c in geo_cols})
        rows[r["file"]] = d
    print(f"geometry measured for {len(rows)} faces")

    cache = {}
    for line in LM.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error") and r.get("lm"):
                cache[r["file"]] = r
    P_all, shapes = {}, {}
    for name, r in cache.items():
        lm = np.array(S.fill_iris([None if p is None else list(p) for p in r["lm"]]), dtype=float)
        P = lm[:, :2] * np.array([r["w"], r["h"]])
        P_all[name] = (lm, P, r["w"], r["h"])
        shapes[name] = P[N.POINT_IDX]
    ref = {}
    for g, sub in man.groupby("group"):
        use = [shapes[f] for f, fo in zip(sub.file, sub.cv_fold) if fo != 0 and f in shapes]
        ref[g] = B2.mean_shape(use)
        print(f"mean shape {g}: {len(use)} faces (folds 1-4)")
    fails = []
    for name, (lm, P, w, h) in P_all.items():
        if name not in rows:
            continue
        try:
            ipd = float(np.hypot(*(P[468] - P[473])))
            f = {}
            f.update(C.one(IMG / name, lm, w, h))
            f["avg_dist"] = float(100 * np.sqrt(((B2.align(shapes[name], ref[info.loc[name, "group"]]) - ref[info.loc[name, "group"]]) ** 2).sum(axis=1).mean()))
            f["adip_pa"], f["adip_lower"] = B2.pa(P[B2.OVAL]), B2.pa(P[B2.LOWER])
            f["smile"] = float((P[[13, 14], 1].mean() - P[[61, 291], 1].mean()) / ipd)
            f["mouth_open"] = float((P[14, 1] - P[13, 1]) / ipd)
            f.update(B2.image_feats(IMG / name, P, w, h, ipd))
            f.update(B3.one(IMG / name, P, w, h))
            f["jaw_turn"] = (turn(P[58], P[172], P[149]) + turn(P[288], P[397], P[378])) / 2
            f["chin_w"] = float(np.hypot(*(P[148] - P[377])) / np.hypot(*(P[172] - P[397])))
            L = C.lab(cv2.imread(str(IMG / name)))
            fore = float(np.median(L[C.disc(L.shape, *P[151], 0.14 * ipd)][:, 0]))
            ridge = []
            for brow, lid in ((105, 159), (334, 386)):
                cx, cy = (P[brow] + P[lid]) / 2
                ridge.append(float(np.median(L[C.disc(L.shape, cx, cy, 0.07 * ipd)][:, 0])))
            f["ridge_dL"] = fore - float(np.mean(ridge))
            rows[name].update(f)
        except Exception as e:
            fails.append((name, str(e)))
    df = pd.DataFrame(list(rows.values()))
    df = df.merge(man[["file", "image_id", "gender", "ethnicity", "cv_fold"]], on="file", how="left")
    df.to_csv(MB / "mebeauty_features.csv", index=False)
    print(f"written {len(df)} rows, {df.shape[1]} columns; failed {len(fails)} {fails[:3]}")
    print("done")


if __name__ == "__main__":
    main()
