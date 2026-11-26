"""Lockbox split (protocol v1, RESEARCH.md 1d): 25% of every group x source, picked by a fixed hash of the file name.

Run once. Output data/lockbox.csv (file, group, src, lock). Nothing may be tuned, plotted or inspected on lock==True rows
until the single final run. Rank by sha1("20261029:" + file) inside each group x source, lowest 25% are locked.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scut_cm_harmony as S

R = S.ROOT


def main() -> None:
    out = R / "data/lockbox.csv"
    if out.exists():
        sys.exit("lockbox.csv exists: the split is frozen, do not redo it")
    cm = pd.read_csv(S.CSV_PATH).merge(pd.read_csv(R / "data/scut_cm_source.csv"), on="file")[["file", "src"]].assign(group="CM")
    parts = [cm]
    for g in ("AM", "CF", "AF"):
        parts.append(pd.read_csv(R / f"data/scut_{g.lower()}_harmony.csv")[["file", "src"]].assign(group=g))
    d = pd.concat(parts, ignore_index=True)
    d["h"] = d.file.map(lambda f: int(hashlib.sha1(f"20261029:{f}".encode()).hexdigest(), 16))
    d["rank"] = d.groupby(["group", "src"]).h.rank(method="first")
    d["n"] = d.groupby(["group", "src"]).h.transform("size")
    d["lock"] = d["rank"] <= (d["n"] * 0.25).round()
    d[["file", "group", "src", "lock"]].to_csv(out, index=False)
    print(d.groupby(["group", "src"]).lock.agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
