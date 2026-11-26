"""Variance partition (commonality analysis) of the judges' mean rating: geometry G, capture Q, age A, source S.

Rules in STATE.md (2026-10-28, written before the run). Held-out R2 from 5-fold x 2 repeated CV.
Run without -I (onnxruntime, for the age of the women's faces).

Commonality of a set T of groups, from the held-out R2 of every subset of groups:
  C(T) = (-1)^(|T|-1) * sum over U subset of T of (-1)^|U| * R2(all groups minus U)    (R2 of the empty set = 0)
Unique share of a group = C({group}); shared share of several groups = C(that set). The shares add up to R2(all).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import age_model as AGE
import scut_cm_harmony as S

R = S.ROOT
QUALITY = ["face_share", "sharp", "luma", "contrast", "face_px", "bytes_pp"]  # img_w, img_h, jpeg_q are constants in SCUT


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def load_sets() -> dict[str, pd.DataFrame]:
    cols = list(pd.read_csv(S.CSV_PATH, nrows=1).columns)
    props = cols[cols.index("face-wh"): cols.index("ala-jaw") + 1]
    sess = ort.InferenceSession(str(AGE.MODEL), providers=["CPUExecutionProvider"])
    age = pd.read_csv(R / "data/scut_age_mesh.csv")  # CM and AM
    for g in ("CF", "AF"):
        age = pd.concat([age, AGE.estimate(sess, S.load_done(R / "data" / f"scut_{g.lower()}_landmarks_web.jsonl"), S.IMAGES)])
    age[["file", "age_mesh"]].to_csv(R / "data" / "scut_age_mesh_all.csv", index=False)
    cm = pd.read_csv(S.CSV_PATH).merge(pd.read_csv(R / "data/scut_cm_quality.csv"), on="file").merge(pd.read_csv(R / "data/scut_cm_source.csv"), on="file")
    out = {"CM": cm}
    for g in ("AM", "CF", "AF"):
        out[g] = pd.read_csv(R / "data" / f"scut_{g.lower()}_harmony.csv")
    for g in out:
        out[g] = out[g].merge(age[["file", "age_mesh"]], on="file")
    return out, props


def partition(df: pd.DataFrame, groups: dict[str, list[str]], fac, repeats: int = 2) -> dict:
    y = df.rating_mean.values
    names = list(groups)
    r2s = {frozenset(): 0.0}
    for k in range(1, len(names) + 1):
        for sub in itertools.combinations(names, k):
            X = df[[c for g in sub for c in groups[g]]].values.astype(float)
            r2s[frozenset(sub)] = r2(y, S.oof_predict(fac, X, y, repeats, S.SEED))
    full = frozenset(names)
    comm = {}
    for k in range(1, len(names) + 1):
        for T in itertools.combinations(names, k):
            val = 0.0
            for m in range(0, len(T) + 1):
                for U in itertools.combinations(T, m):
                    val += (-1) ** m * r2s[full - frozenset(U)]
            comm[T] = (-1) ** (len(T) - 1) * val
    assert abs(sum(comm.values()) - r2s[full]) < 1e-9
    return {"R2_full": r2s[full], "r2s": r2s, "comm": comm}


def describe(res: dict) -> str:
    items = sorted(res["comm"].items(), key=lambda kv: (len(kv[0]), kv[0]))
    return "; ".join(f"{'+'.join(T)} {v:+.3f}" for T, v in items) + f" | all together {res['R2_full']:.3f}"


def main() -> None:
    sets, props = load_sets()
    G = {"G": props, "Q": QUALITY, "A": ["age_mesh"]}
    plan = [("CM", "10k_US"), ("CM", "internet"), ("CF", "10k_US"), ("CF", "internet"), ("AM", "DataTang"), ("AM", "internet"), ("AF", "DataTang"), ("AF", "internet")]
    for name, fac in (("ridge", S.ridge), ("boosting", S.hgb)):
        print(f"\n=== inside one source, {name}: held-out R2 shares (letter alone = unique, letters joined by + = shared); G geometry, Q capture, A age")
        for st, src in plan:
            d = sets[st][sets[st].src == src].reset_index(drop=True)
            if len(d) < 200:
                continue
            print(f"  {st} {src:9s} n={len(d):4d}: {describe(partition(d, G, fac))}")
    print("\n=== pooled per set with source dummies S (ridge)")
    for st in ("CM", "CF", "AM", "AF"):
        d = sets[st].copy()
        d = d[d.src.map(d.src.value_counts()) >= 100].reset_index(drop=True)
        dummies = pd.get_dummies(d.src, prefix="src").astype(float)
        d = pd.concat([d, dummies], axis=1)
        res = partition(d, {**G, "S": list(dummies.columns)}, S.ridge)
        r = res["r2s"]
        print(f"  {st} n={len(d):4d} sources {d.src.value_counts().to_dict()}: {describe(res)}")
        print(f"      alone: S {r[frozenset(['S'])]:.3f}, G {r[frozenset(['G'])]:.3f}, Q {r[frozenset(['Q'])]:.3f}, A {r[frozenset(['A'])]:.3f}; G+S {r[frozenset(['G', 'S'])]:.3f}")


if __name__ == "__main__":
    main()
