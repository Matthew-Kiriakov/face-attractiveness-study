"""Block 2a features: colours (Lab) and contrast, from the SCUT images and the landmark caches. All faces are measured (a feature
extractor sees no ratings); analysis later uses only non-lockbox faces.

Lab is CIE Lab from OpenCV (L 0-255 scaled back to 0-100; a, b centred at 0). Patches are circles of radius r*IPD around mesh points.
Features per face:
  skin_L, skin_a (redness), skin_b (yellowness): median over both cheek patches (mesh 50 and 280, r 0.14 IPD)
  skin_even: median absolute deviation of L inside the cheek patches (smoothness proxy; also image sharpness)
  lr_diff: |L left cheek - L right cheek| (lighting asymmetry nuisance)
  fore_L: median L of the forehead patch (mesh 151, r 0.14 IPD)
  iris_L, iris_a, iris_b: median Lab of the lower half of the iris ring (0.45-0.85 of the iris radius), both eyes
  hair_L: median L of a patch above the forehead (x of mesh 10, y 0.6 IPD above it, r 0.15 IPD); NaN if outside the image
  hair_skin_dL: hair_L - skin_L
  brow_L: mean L of the darkest 40% pixels inside the brow polygons; brow_contrast: skin_L - brow_L
  eye_contrast: skin_L - mean L inside the eye-opening polygons
  mouth_contrast: mean a inside the outer-lip polygon - skin_a
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_cm_harmony as S

R = S.ROOT
BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
EYE_L = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
EYE_R = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def lab(img_bgr: np.ndarray) -> np.ndarray:
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    x[..., 0] *= 100 / 255
    x[..., 1:] -= 128
    return x


def disc(shape, cx, cy, r):
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 1, -1)
    return m.astype(bool)


def poly(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 1)
    return m.astype(bool)


def one(path: Path, lm: np.ndarray, w: int, h: int) -> dict:
    img = cv2.imread(str(path))
    L = lab(img)
    P = lm[:, :2] * np.array([w, h])
    ipd = float(np.hypot(*(P[468] - P[473])))
    f = {}
    ch = [disc(L.shape, *P[i], 0.14 * ipd) for i in (50, 280)]
    cheeks = [L[m] for m in ch]
    skin = np.concatenate(cheeks)
    f["skin_L"], f["skin_a"], f["skin_b"] = (float(np.median(skin[:, k])) for k in range(3))
    f["skin_even"] = float(np.median(np.abs(skin[:, 0] - np.median(skin[:, 0]))))
    f["lr_diff"] = float(abs(np.median(cheeks[0][:, 0]) - np.median(cheeks[1][:, 0])))
    f["fore_L"] = float(np.median(L[disc(L.shape, *P[151], 0.14 * ipd)][:, 0]))
    ring = []
    for c, ring_idx in ((468, (469, 470, 471, 472)), (473, (474, 475, 476, 477))):
        rad = float(np.mean([np.hypot(*(P[i] - P[c])) for i in ring_idx]))
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.hypot(xx - P[c][0], yy - P[c][1])
        m = (d >= 0.45 * rad) & (d <= 0.85 * rad) & (yy >= P[c][1])
        if m.sum() >= 5:
            ring.append(L[m])
    if ring:
        ir = np.concatenate(ring)
        f["iris_L"], f["iris_a"], f["iris_b"] = (float(np.median(ir[:, k])) for k in range(3))
    else:
        f["iris_L"] = f["iris_a"] = f["iris_b"] = np.nan
    hx, hy = P[10][0], P[10][1] - 0.6 * ipd
    r = 0.15 * ipd
    if hy - r >= 0 and 0 <= hx - r and hx + r < w:
        f["hair_L"] = float(np.median(L[disc(L.shape, hx, hy, r)][:, 0]))
    else:
        f["hair_L"] = np.nan
    f["hair_skin_dL"] = f["hair_L"] - f["skin_L"]
    brow = L[poly(L.shape, P[BROW_L]) | poly(L.shape, P[BROW_R])][:, 0]
    if len(brow) >= 10:
        k = max(1, int(0.4 * len(brow)))
        f["brow_L"] = float(np.sort(brow)[:k].mean())
    else:
        f["brow_L"] = np.nan
    f["brow_contrast"] = f["skin_L"] - f["brow_L"]
    eye = L[poly(L.shape, P[EYE_L]) | poly(L.shape, P[EYE_R])][:, 0]
    f["eye_contrast"] = f["skin_L"] - float(eye.mean()) if len(eye) else np.nan
    lips = L[poly(L.shape, P[LIPS])][:, 1]
    f["mouth_contrast"] = float(lips.mean()) - f["skin_a"] if len(lips) else np.nan
    return f


def main() -> None:
    out = []
    for g in ("CM", "AM", "CF", "AF"):
        cache = S.load_done(S.CACHE) if g == "CM" else S.load_done(R / "data" / f"scut_{g.lower()}_landmarks_web.jsonl")
        for name, r in cache.items():
            if r.get("error") or not r.get("lm"):
                continue
            lm = np.array(S.fill_iris([None if p is None else list(p) for p in r["lm"]]), dtype=float)
            try:
                out.append({"file": name, **one(S.IMAGES / name, lm, r["w"], r["h"])})
            except Exception as e:  # keep going, record the failure
                print("failed", name, e)
        print(g, len(out), flush=True)
    pd.DataFrame(out).to_csv(R / "data/scut_colour_features.csv", index=False)


if __name__ == "__main__":
    main()
