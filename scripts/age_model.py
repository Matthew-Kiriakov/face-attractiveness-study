"""Perceived-age estimate (InsightFace genderage, models/genderage.onnx) with a landmark-box crop, and what it does.

Crop: centre of the 468-point mesh box, scale 96 / (1.5 * max(box w, h)) as in insightface Attribute.get. The earlier
scripts/scut_age.py used the WHOLE SCUT image as the box (350 px crops with the face about 60% of the width), so the face
was smaller in the 96 px canvas than the network expects. This script uses the mesh box everywhere (SCUT and the author's
photos), so all ages here are on the same footing. The number is a network's estimate of age, not a true age.
Run without -I (onnxruntime is in the user site-packages).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_cm_harmony as S

R = S.ROOT
MODEL = R / "models" / "genderage.onnx"


def blob(bgr: np.ndarray, lm: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    x0, y0 = lm.min(0)
    x1, y1 = lm.max(0)
    cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    scale = 96.0 / (max((x1 - x0) * w, (y1 - y0) * h) * 1.5)
    m = np.array([[scale, 0.0, 48.0 - scale * cx], [0.0, scale, 48.0 - scale * cy]], dtype=np.float32)
    canvas = cv2.warpAffine(bgr, m, (96, 96), borderValue=0.0)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None]


def estimate(session, rows: dict[str, dict], folder: Path) -> pd.DataFrame:
    name = session.get_inputs()[0].name
    out = []
    for f, c in rows.items():
        if c.get("error"):
            continue
        bgr = cv2.imread(str(folder / f))
        if bgr is None:
            continue
        pred = session.run(None, {name: blob(bgr, np.array(c["lm"][:468]))})[0][0]
        out.append({"file": f, "age_mesh": float(pred[2] * 100.0), "p_male_raw": float(pred[1])})
    return pd.DataFrame(out)


def main() -> None:
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    sets = {
        "CM": (S.load_done(S.CACHE), S.IMAGES),
        "AM": (S.load_done(R / "data/scut_am_landmarks_web.jsonl"), S.IMAGES),
        "A": (S.load_done(R / "data/author_landmarks_web.jsonl"), R / "data/author_photos"),
    }
    frames = {k: estimate(sess, rows, folder) for k, (rows, folder) in sets.items()}
    for k, df in frames.items():
        print(f"{k}: n={len(df)}, age mean {df.age_mesh.mean():.1f} SD {df.age_mesh.std():.1f} range {df.age_mesh.min():.0f}-{df.age_mesh.max():.0f}")
    pd.concat([frames["CM"], frames["AM"]]).to_csv(R / "data" / "scut_age_mesh.csv", index=False)
    frames["A"].to_csv(R / "data" / "author_age.csv", index=False)  # git-ignored

    # old whole-image estimate vs mesh-box estimate on CM
    old = pd.read_csv(R / "data" / "scut_age.csv").set_index("file")
    cm = frames["CM"].set_index("file")
    j = cm.join(old[["age"]], how="inner")
    print(f"\nold whole-image age vs mesh-box age on CM: r {np.corrcoef(j.age, j.age_mesh)[0, 1]:.3f}, mean old {j.age.mean():.1f}, mean new {j.age_mesh.mean():.1f}")

    # age vs rating in SCUT, within source
    cmh = pd.read_csv(S.CSV_PATH).merge(pd.read_csv(R / "data/scut_cm_source.csv"), on="file").merge(frames["CM"], on="file")
    cols = list(cmh.columns)
    props = cols[cols.index("face-wh"): cols.index("ala-jaw") + 1]
    print("\nage (mesh-box) vs judges' rating, CM:")
    for k, g in [("all", cmh)] + list(cmh.groupby("src")):
        print(f"  {k:9s} n={len(g)} r {np.corrcoef(g.age_mesh, g.rating_mean)[0, 1]:+.3f}, age mean {g.age_mesh.mean():.1f} SD {g.age_mesh.std():.1f}")
    am = pd.read_csv(R / "data/scut_am_harmony.csv").merge(frames["AM"], on="file")
    for k, g in [("AM all", am)] + [(f"AM {s}", g) for s, g in am.groupby("src") if len(g) > 100]:
        print(f"  {k:9s} n={len(g)} r {np.corrcoef(g.age_mesh, g.rating_mean)[0, 1]:+.3f}, age mean {g.age_mesh.mean():.1f} SD {g.age_mesh.std():.1f}")

    # does age add to geometry? held-out, within 10k US
    g = cmh[cmh.src == "10k_US"].reset_index(drop=True)
    y = g.rating_mean.values
    X = g[props].values.astype(float)
    for tag, M in (("geometry", X), ("age only", g[["age_mesh"]].values), ("geometry + age", np.column_stack([X, g.age_mesh.values]))):
        for name, fac in (("ridge", S.ridge), ("boosting", S.hgb)):
            p = S.oof_predict(fac, M, y, 5, S.SEED)
            print(f"  10k US held-out {tag:15s}{name:9s} r {S.pearson(y, p):.3f}")
    print("  age vs geometry model (OOF boosting) correlation, 10k US: %.3f" % np.corrcoef(g.age_mesh, S.oof_predict(S.hgb, X, y, 5, S.SEED))[0, 1])

    # the author's photos: spread of age across photos vs between SCUT faces
    a = frames["A"].set_index("file")
    feats = pd.read_csv(R / "data" / "author_features.csv").set_index("file")
    front = [f for f in feats.index if abs(feats.loc[f, "roll_deg"]) <= 7 and 0.85 <= feats.loc[f, "half_ratio"] <= 1.18]
    mh = S.hgb().fit(cmh[props].values.astype(float), cmh.rating_mean.values)
    pred = pd.Series(mh.predict(feats[props].values.astype(float)), index=feats.index)
    print(f"\nauthor's photos: age (all 20) mean {a.age_mesh.mean():.1f} SD {a.age_mesh.std():.1f} range {a.age_mesh.min():.0f}-{a.age_mesh.max():.0f}; frontal {len(front)}: mean {a.loc[front, 'age_mesh'].mean():.1f} SD {a.loc[front, 'age_mesh'].std():.1f} range {a.loc[front, 'age_mesh'].min():.0f}-{a.loc[front, 'age_mesh'].max():.0f}")
    print(f"  variance of age across his frontal photos / across SCUT CM faces: {a.loc[front, 'age_mesh'].var() / cmh.age_mesh.var():.2f} (geometry model, boosting: 0.17)")
    print(f"  across his frontal photos: age vs geometry-model score r {np.corrcoef(a.loc[front, 'age_mesh'], pred.loc[front])[0, 1]:+.2f} (n={len(front)})")
    print("  per photo (file, age, model score):", [(f, round(a.loc[f, 'age_mesh'], 1), round(pred.loc[f], 2)) for f in sorted(front)])


if __name__ == "__main__":
    main()
