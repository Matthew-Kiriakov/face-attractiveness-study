"""Source of each CM image from the dataset's own Images_Sources.xlsx (10k US Adults Face DB vs Internet).

Writes data/scut_cm_source.csv and compares with the capture clusters. EXPLORATORY.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_cm_harmony as S

SRC = S.IMAGES.parent / "Images_Sources.xlsx"


def read_sources() -> pd.DataFrame:
    ws = load_workbook(SRC, read_only=True)["Caucasian_facial_images"]
    rows = []
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if not r[0]:
            continue
        name = re.sub(r"[,.]jpg$", ".jpg", str(r[0]).strip())
        if not name.startswith("CM"):
            continue
        rows.append({"file": name, "src": "10k_US" if r[1] else ("internet" if r[2] else "none"), "url": r[2]})
    return pd.DataFrame(rows)


def main() -> None:
    s = read_sources()
    s[["file", "src"]].to_csv(S.ROOT / "data" / "scut_cm_source.csv", index=False)
    d = pd.read_csv(S.CSV_PATH).merge(s, on="file").merge(pd.read_csv(S.ROOT / "data" / "scut_cm_quality_clusters.csv"), on="file")
    print("n", len(d), d.src.value_counts().to_dict())
    print(pd.crosstab(d.cl, d.src).to_string())
    cols = list(pd.read_csv(S.CSV_PATH, nrows=1).columns)
    props = cols[cols.index("face-wh"): cols.index("ala-jaw") + 1]
    y = d.rating_mean.values
    P = d[props].values.astype(float)
    glob = S.oof_predict(S.hgb, P, y, 5, S.SEED)
    for k, g in d.groupby("src"):
        i = g.index.values
        within = S.oof_predict(S.ridge, P[i], y[i], 5, S.SEED)
        tr = (d.src == k).values
        cross = S.ridge().fit(P[tr], y[tr]).predict(P[~tr])
        print(f"{k:9s} n={len(i):3d} rating mean {y[i].mean():.2f} sd {y[i].std():.2f} | global-model r {S.pearson(y[i], glob[i]):.3f} | within-CV ridge r {S.pearson(y[i], within):.3f} | train here -> other r {S.pearson(y[~tr], cross):.3f} | trichion_found {g.trichion_found.mean():.2f} above {g.trichion_above.mean():.2f}")
    d["n"] = d.file.str[2:-4].astype(int)
    print("internet per block of 50:", d.groupby((d.n - 1) // 50).apply(lambda g: int((g.src == "internet").sum())).tolist())
    ws = d.groupby("src").rating_mean.mean()
    print("rating level difference internet - 10k:", round(ws["internet"] - ws["10k_US"], 2))


if __name__ == "__main__":
    main()
