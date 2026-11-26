"""Step 1: how much of the rating signal does frame quality alone carry?

Reads cached landmarks (no re-detection), computes 9 capture features per CM image,
writes data/scut_cm_quality.csv, then fits the same CV protocol as scut_cm_harmony.py.
Feature list and reading rules are fixed in STATE.md before the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_cm_harmony as S

OUT = S.ROOT / "data" / "scut_cm_quality.csv"
QUALITY = ["img_w", "img_h", "face_share", "sharp", "luma", "contrast", "face_px", "jpeg_q", "bytes_pp"]


def laplacian(a: np.ndarray) -> np.ndarray:
    return a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]


def features() -> pd.DataFrame:
    cache = S.load_done(S.CACHE)
    rows = []
    for name in sorted(cache, key=lambda n: int(n[2:-4])):
        c = cache[name]
        if c.get("error"):
            continue
        path = S.IMAGES / name
        with Image.open(path) as im:
            q = np.mean(im.quantization[0]) if hasattr(im, "quantization") else np.nan
            rgb = np.asarray(im.convert("RGB"))
        h, w = rgb.shape[:2]
        lm = np.array(c["lm"][:468])
        x0, y0 = np.clip(lm.min(0), 0, 1)
        x1, y1 = np.clip(lm.max(0), 0, 1)
        X0, X1, Y0, Y1 = int(x0 * w), int(x1 * w), int(y0 * h), int(y1 * h)
        gray = rgb @ np.array([0.299, 0.587, 0.114])
        face = gray[Y0:Y1, X0:X1]
        rows.append(
            dict(
                file=name, img_w=w, img_h=h,
                face_share=(x1 - x0) * (y1 - y0),
                sharp=float(laplacian(face).var()),
                luma=float(face.mean()), contrast=float(face.std()),
                face_px=X1 - X0, jpeg_q=float(q),
                bytes_pp=path.stat().st_size / (w * h),
            )
        )
    return pd.DataFrame(rows)


def rep(tag, X, y, h_only=False):
    out = {}
    for n, f in (("ridge", S.ridge), ("hgb", S.hgb)):
        p = S.oof_predict(f, X, y, 5, S.SEED)
        out[n] = (S.pearson(y, p), S.r2_score(y, p))
        print(f"{tag:16s}{n:6s} r={out[n][0]:.3f} R2={out[n][1]:.3f}")
    return out


def main() -> None:
    q = features()
    q.to_csv(OUT, index=False)
    df = pd.read_csv(S.CSV_PATH).merge(q, on="file")
    ag = pd.read_csv(S.ROOT / "data" / "scut_age.csv").set_index("file")
    df["age"] = ag.loc[df.file, "age"].values
    print("n", len(df))
    y = df.rating_mean.values
    cols = list(pd.read_csv(S.CSV_PATH, nrows=1).columns)
    props = cols[cols.index("face-wh"): cols.index("ala-jaw") + 1]
    print(q[QUALITY].describe().loc[["mean", "std", "min", "max"]].round(2).T.to_string())
    print("\nunivariate r with rating_mean / age")
    for c in QUALITY:
        a, b = S.pearson(df[c], y), S.pearson(df[c], df.age)
        print(f"  {c:10s}", "constant" if a is None else f"{a:+.3f}  {b:+.3f}")
    print()
    Q = df[QUALITY].values.astype(float)
    rep("quality only", Q, y)
    rep("props", df[props].values.astype(float), y)
    rep("props+quality", np.column_stack([df[props].values, Q]).astype(float), y)
    rep("props+qual+age", np.column_stack([df[props].values, Q, df.age.values]).astype(float), y)
    rng = np.random.default_rng(S.SEED)
    nl = [abs(S.pearson(p := rng.permutation(y), S.oof_predict(S.ridge, Q, p, 1, S.SEED + 1 + i))) for i in range(30)]
    print(f"null (shuffled) ridge on quality: mean |r| {np.mean(nl):.3f} max {max(nl):.3f}")
    print("\nEXPLORATORY (not pre-registered): which quality features carry it")
    for tag, cols_ in (("size only", ["face_share", "face_px"]),
                       ("no size", ["sharp", "luma", "contrast", "bytes_pp"]),
                       ("bytes_pp only", ["bytes_pp"])):
        rep(tag, df[cols_].values.astype(float), y)
    rep("props+no size", np.column_stack([df[props].values, df[["sharp", "luma", "contrast", "bytes_pp"]].values]).astype(float), y)
    print("size vs props: ridge R2 of face_px from props:", end=" ")
    pp = S.oof_predict(S.ridge, df[props].values.astype(float), df.face_px.values, 5, S.SEED)
    print(f"{S.r2_score(df.face_px.values, pp):.3f}")
    print("quality -> age (ridge):", end=" ")
    p = S.oof_predict(S.ridge, Q, df.age.values, 5, S.SEED)
    print(f"r={S.pearson(df.age.values, p):.3f}")


if __name__ == "__main__":
    main()
