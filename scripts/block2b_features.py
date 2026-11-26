"""Block 2b features: averageness, face adiposity, beard/stubble, expression and pose. Measured on all faces (no ratings are used);
the mean shape per group is computed from NON-lockbox faces only. Analysis: block2b_analysis.py.

Features:
  avg_dist      RMS distance of the 40 feature landmarks to the group's mean shape after Procrustes alignment (centroid size units x100)
  adip_pa       perimeter^2 / area of the face oval (mesh silhouette polygon); lower = rounder, fuller face
  adip_lower    the same for the lower-face polygon (cheek to cheek via the chin)
  chin_dL, lip_dL, chin_sd   cheek-skin L minus L at the chin patch / at the upper-lip patch; SD of L in the chin patch (stubble texture)
  smile         (y of lip centre - mean y of the mouth corners) / IPD; positive = corners up
  mouth_open    inner-lip gap / IPD
  teeth         share of bright, low-chroma pixels inside the inner-mouth polygon
  roll_abs, half_dev   |head roll| in degrees and |half-face ratio - 1| from the harmony tables (pose nuisance)
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
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LOWER = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


def pa(pts: np.ndarray) -> float:
    d = np.roll(pts, -1, axis=0) - pts
    per = np.hypot(d[:, 0], d[:, 1]).sum()
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(per ** 2 / area)


def align(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0)
    X = X / np.sqrt((X ** 2).sum())
    u, _, vt = np.linalg.svd(X.T @ ref)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, d]) @ vt
    return X @ r


def mean_shape(shapes: list[np.ndarray]) -> np.ndarray:
    ref = shapes[0] - shapes[0].mean(axis=0)
    ref /= np.sqrt((ref ** 2).sum())
    for _ in range(4):
        ref = np.mean([align(s, ref) for s in shapes], axis=0)
        ref /= np.sqrt((ref ** 2).sum())
    return ref


def image_feats(path: Path, P: np.ndarray, w: int, h: int, ipd: float) -> dict:
    L = C.lab(cv2.imread(str(path)))
    cheek = np.concatenate([L[C.disc(L.shape, *P[i], 0.14 * ipd)] for i in (50, 280)])
    skin_L = float(np.median(cheek[:, 0]))
    f = {}
    cx, cy = (P[17] + P[152]) / 2
    chin = L[C.disc(L.shape, cx, cy, 0.14 * ipd)]
    f["chin_dL"] = skin_L - float(np.median(chin[:, 0]))
    f["chin_sd"] = float(chin[:, 0].std())
    ux, uy = (P[0] + P[2]) / 2
    f["lip_dL"] = skin_L - float(np.median(L[C.disc(L.shape, ux, uy, 0.08 * ipd)][:, 0]))
    inner = L[C.poly(L.shape, P[INNER])]
    f["teeth"] = float(np.mean((inner[:, 0] > 70) & (np.hypot(inner[:, 1], inner[:, 2]) < 18))) if len(inner) >= 10 else 0.0
    return f


def main() -> None:
    lock = pd.read_csv(R / "data/lockbox.csv").set_index("file").lock
    rows = []
    for g in ("CM", "AM", "CF", "AF"):
        cache = N.cache_for(g)
        names, shapes, P_all = [], [], {}
        for name, r in cache.items():
            if r.get("error") or not r.get("lm"):
                continue
            lm = np.array(S.fill_iris([None if p is None else list(p) for p in r["lm"]]), dtype=float)
            P = lm[:, :2] * np.array([r["w"], r["h"]])
            P_all[name] = (P, r["w"], r["h"])
            names.append(name)
            shapes.append(P[N.POINT_IDX])
        ref = mean_shape([s for n, s in zip(names, shapes) if not lock.get(n, True)])
        harm = pd.read_csv(S.CSV_PATH if g == "CM" else R / "data" / f"scut_{g.lower()}_harmony.csv").set_index("file")
        for name, s in zip(names, shapes):
            P, w, h = P_all[name]
            ipd = float(np.hypot(*(P[468] - P[473])))
            f = {"file": name, "avg_dist": float(100 * np.sqrt(((align(s, ref) - ref) ** 2).sum(axis=1).mean()))}
            f["adip_pa"], f["adip_lower"] = pa(P[OVAL]), pa(P[LOWER])
            f["smile"] = float((P[[13, 14], 1].mean() - P[[61, 291], 1].mean()) / ipd)
            f["mouth_open"] = float((P[14, 1] - P[13, 1]) / ipd)
            f.update(image_feats(S.IMAGES / name, P, w, h, ipd))
            if name in harm.index:
                f["roll_abs"] = abs(float(harm.loc[name, "roll_deg"]))
                f["half_dev"] = abs(float(harm.loc[name, "half_ratio"]) - 1)
            rows.append(f)
        print(g, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(R / "data/scut_block2b_features.csv", index=False)


if __name__ == "__main__":
    main()
