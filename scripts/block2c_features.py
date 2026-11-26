"""Block 2c features (the author's list of eye/skin/brow/nose markers that a frontal 350 px photo can give). All faces; no ratings used.

Sub-blocks (declared before the analysis):
  eye_white  sclera_show_low, sclera_show_up (sclera visible below / above the iris, in iris radii; negative = covered by the lid),
             sclera_L, sclera_a, sclera_b (colour of the white of the eye outside the iris disc)
  under_eye  dark_dL (cheek L - under-eye L), dark_da, dark_db (under-eye minus cheek in a and b), bag_range (range of the row-mean L, 0.08-0.35 IPD below the lower lid)
  lid        crease_dist, crease_depth (darkest row above the upper lid margin: distance in IPD, depth in L), lid_dL (lid skin L - cheek L)
  nose_tip   tip_w (nostril-low distance / alar width), tip_h (subnasale-tip distance / nose length), tip_shade (L at the tip - L at the alae)
  texture    tex_L, tex_a (MAD of the high-passed L and a in cheek patches), spot_red, spot_dark (share of pixels with a high-pass a > 3 / L < -6),
             fore_tex_L (the same MAD in the forehead patch)
  brows      brow_dens (share of dark pixels in the dilated brow polygon), brow_thick (mean dark-pixel run across the brow, IPD), brow_arch (peak height over the head-tail chord)
High-pass = Lab channel minus a Gaussian blur of sigma 0.15 IPD. Dark = L < cheek-skin L - 12. Left/right averaged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block2a_colour as C
import noise_attenuation as N
import scut_cm_harmony as S

R = S.ROOT
EYES = {  # upper lid, lower lid, iris centre, iris ring, polygon, outer corner, inner corner, brow low point
    "L": (159, 145, 468, (469, 470, 471, 472), C.EYE_L, 33, 133, 52),
    "R": (386, 374, 473, (474, 475, 476, 477), C.EYE_R, 263, 362, 282),
}
BROWS = {"L": (107, 105, 46, C.BROW_L), "R": (336, 334, 276, C.BROW_R)}


def mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def nm(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def one(path: Path, P: np.ndarray, w: int, h: int) -> dict:
    img = cv2.imread(str(path))
    L = C.lab(img)
    ipd = float(np.hypot(*(P[468] - P[473])))
    yy, xx = np.mgrid[0:h, 0:w]
    cheeks = [L[C.disc(L.shape, *P[i], 0.14 * ipd)] for i in (50, 280)]
    cheek = np.concatenate(cheeks)
    skin_L, skin_a, skin_b = (float(np.median(cheek[:, k])) for k in range(3))
    f = {}
    acc = {k: [] for k in ["su", "sl", "sL", "sa", "sb", "ddL", "dda", "ddb", "bag", "cd", "cdep", "lid", "arch"]}
    for side, (up, lo, ic, ring, poly_idx, outer, inner, browlow) in EYES.items():
        rad = float(np.mean([np.hypot(*(P[i] - P[ic])) for i in ring]))
        acc["su"].append(((P[ic][1] - rad) - P[up][1]) / rad)
        acc["sl"].append((P[lo][1] - (P[ic][1] + rad)) / rad)
        pm = C.poly(L.shape, P[poly_idx]) & (np.hypot(xx - P[ic][0], yy - P[ic][1]) > 1.1 * rad)
        pm = cv2.erode(pm.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        if pm.sum() >= 6:
            px = L[pm]
            acc["sL"].append(float(np.median(px[:, 0])))
            acc["sa"].append(float(np.median(px[:, 1])))
            acc["sb"].append(float(np.median(px[:, 2])))
        ux, uy = P[lo][0], P[lo][1] + 0.17 * ipd
        under = L[C.disc(L.shape, ux, uy, 0.07 * ipd)]
        if len(under):
            acc["ddL"].append(skin_L - float(np.median(under[:, 0])))
            acc["dda"].append(float(np.median(under[:, 1])) - skin_a)
            acc["ddb"].append(float(np.median(under[:, 2])) - skin_b)
        y0, y1 = int(round(P[lo][1] + 0.08 * ipd)), int(round(P[lo][1] + 0.35 * ipd))
        x0, x1 = int(round(P[lo][0] - 0.07 * ipd)), int(round(P[lo][0] + 0.07 * ipd))
        if 0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w:
            rows = L[y0:y1, x0:x1, 0].mean(axis=1)
            rows = np.convolve(rows, np.ones(3) / 3, mode="valid")
            acc["bag"].append(float(rows.max() - rows.min()) if len(rows) else np.nan)
        # lid crease: darkest row above the upper lid margin
        yl = P[up][1]
        top = max(yl - 0.30 * ipd, P[browlow][1] + 2)
        x0, x1 = int(round(P[up][0] - 0.06 * ipd)), int(round(P[up][0] + 0.06 * ipd))
        ya, yb = int(round(top)), int(round(yl)) - 2
        if yb - ya >= 4 and 0 <= ya and yb <= h and 0 <= x0 < x1 <= w:
            rows = L[ya:yb, x0:x1, 0].mean(axis=1)
            k = int(np.argmin(rows))
            acc["cd"].append((yl - (ya + k)) / ipd)
            acc["cdep"].append(float(np.median(rows) - rows[k]))
            acc["lid"].append(float(np.median(rows)) - skin_L)
        head, peak, tail, _poly = BROWS[side]
        chord = P[tail] - P[head]
        n = np.hypot(*chord)
        acc["arch"].append(abs(chord[0] * (P[peak][1] - P[head][1]) - chord[1] * (P[peak][0] - P[head][0])) / n / n)
    f["sclera_show_up"], f["sclera_show_low"] = nm(acc["su"]), nm(acc["sl"])
    f["sclera_L"], f["sclera_a"], f["sclera_b"] = nm(acc["sL"]), nm(acc["sa"]), nm(acc["sb"])
    f["dark_dL"], f["dark_da"], f["dark_db"], f["bag_range"] = nm(acc["ddL"]), nm(acc["dda"]), nm(acc["ddb"]), nm(acc["bag"])
    f["crease_dist"], f["crease_depth"], f["lid_dL"] = nm(acc["cd"]), nm(acc["cdep"]), nm(acc["lid"])
    f["brow_arch"] = nm(acc["arch"])
    # nose tip
    f["tip_w"] = float(np.hypot(*(P[48] - P[278])) / np.hypot(*(P[98] - P[327])))
    f["tip_h"] = float((P[2][1] - P[1][1]) / np.hypot(*(P[168] - P[2])))
    tip = L[C.disc(L.shape, *P[1], 0.07 * ipd)][:, 0]
    ala = np.concatenate([L[C.disc(L.shape, *P[i], 0.07 * ipd)][:, 0] for i in (98, 327)])
    f["tip_shade"] = float(np.median(tip) - np.median(ala)) if len(tip) and len(ala) else np.nan
    # texture
    sig = max(1.0, 0.15 * ipd)
    hp = L - cv2.GaussianBlur(L.astype(np.float32), (0, 0), sig).astype(float)
    cm = np.zeros(L.shape[:2], bool)
    for i in (50, 280):
        cm |= C.disc(L.shape, *P[i], 0.14 * ipd)
    fm = C.disc(L.shape, *P[151], 0.14 * ipd)
    f["tex_L"], f["tex_a"] = mad(hp[cm][:, 0]), mad(hp[cm][:, 1])
    f["spot_red"], f["spot_dark"] = float(np.mean(hp[cm][:, 1] > 3)), float(np.mean(hp[cm][:, 0] < -6))
    f["fore_tex_L"] = mad(hp[fm][:, 0])
    # brows
    dark = L[..., 0] < skin_L - 12
    pm = np.zeros(L.shape[:2], bool)
    for side in BROWS:
        pm |= C.poly(L.shape, P[BROWS[side][3]])
    pm = cv2.dilate(pm.astype(np.uint8), np.ones((max(3, int(0.06 * ipd)),) * 2, np.uint8)).astype(bool)
    f["brow_dens"] = float(dark[pm].mean()) if pm.sum() else np.nan
    runs = (dark & pm).sum(axis=0)
    runs = runs[runs > 0]
    f["brow_thick"] = float(runs.mean() / ipd) if len(runs) else np.nan
    return f


def main() -> None:
    rows = []
    for g in ("CM", "AM", "CF", "AF"):
        cache = N.cache_for(g)
        for name, r in cache.items():
            if r.get("error") or not r.get("lm"):
                continue
            lm = np.array(S.fill_iris([None if p is None else list(p) for p in r["lm"]]), dtype=float)
            P = lm[:, :2] * np.array([r["w"], r["h"]])
            try:
                rows.append({"file": name, **one(S.IMAGES / name, P, r["w"], r["h"])})
            except Exception as e:
                print("failed", name, e)
        print(g, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(R / "data/scut_block2c_features.csv", index=False)


if __name__ == "__main__":
    main()
