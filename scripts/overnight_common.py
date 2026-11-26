"""Shared helpers for the overnight batch (2026-10-31): pooled non-lockbox sample with geometry z-scored inside each set (the corrected setting), tier-2 feature list,
multi-seed out-of-fold predictions with a disk cache, and summaries of paired differences across seeds. EXPLORATORY; the lockbox is never read (load_pooled drops it)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import all_clusters_kfold as A
import group_ablation as G
import scut_cm_harmony as S

R = S.ROOT
CACHE_DIR = R / "data/cache_expl/overnight"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [20261023 + 17 * k for k in range(10)]
TIER2_GROUPS = G.GROUPS
ALLG = [c for v in G.GROUPS.values() for c in v]


def zset(df: pd.DataFrame, cols: list[str]) -> None:
    """Winsorise 1/99 % and z-score inside each set (in place)."""
    for s in df.setname.unique():
        m = df.setname == s
        for c in cols:
            lo, hi = df.loc[m, c].quantile([0.01, 0.99])
            v = df.loc[m, c].clip(lo, hi)
            df.loc[m, c] = (v.fillna(v.median()) - v.mean()) / v.std()


def load(extra_csvs: dict[str, list[str]] | None = None, z_extra: bool = True):
    """Returns df (pooled, y = judges' z inside set), props (z-scored), rest, tier2 columns. extra_csvs: {csv path relative to ROOT: [columns]} merged on file."""
    df, props, rest, tag = A.load_pooled()
    zset(df, props)
    added = []
    for path, cols in (extra_csvs or {}).items():
        x = pd.read_csv(R / path)[["file"] + cols]
        df = df.merge(x, on="file", how="left")
        added += cols
    if z_extra and added:
        zset(df, added)
    return df, props, rest, props + ALLG


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def oof_seeds(df: pd.DataFrame, cols: list[str], y: np.ndarray, fam: str, tag: str, seeds=SEEDS) -> np.ndarray:
    """(n_seeds, n) out-of-fold predictions, cached on disk by (tag, family, columns)."""
    X = df[cols].values.astype(float)
    h = hashlib.md5((fam + "|".join(cols) + "|" + ",".join(map(str, seeds))).encode())
    h.update(np.ascontiguousarray(np.nan_to_num(X, nan=-9999.0)).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    f = CACHE_DIR / f"{h.hexdigest()[:20]}.npy"  # keyed by the data themselves, so identical fits are shared between jobs (tag is only a label)
    if f.exists():
        return np.load(f)
    fac = S.hgb if fam == "boosting" else S.ridge
    out = np.array([S.oof_predict(fac, X, y, 2, s) for s in seeds])
    np.save(f, out)
    return out


def diff(y, pa: np.ndarray, pb: np.ndarray) -> tuple[float, float, float, float]:
    """Delta R2 (a minus b): mean over seeds, SD over seeds, 95% face-bootstrap interval on the seed-mean predictions (300 resamples)."""
    d = np.array([r2(y, a) - r2(y, b) for a, b in zip(pa, pb)])
    ma, mb = pa.mean(axis=0), pb.mean(axis=0)
    rng = np.random.default_rng(20261031)
    bs = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        bs.append(r2(y[i], ma[i]) - r2(y[i], mb[i]))
    return float(d.mean()), float(d.std(ddof=1)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t) -> str:
    return f"{t[0]:+.4f} (sd over seeds {t[1]:.4f}) [{t[2]:+.4f}, {t[3]:+.4f}]"


def within(df: pd.DataFrame, a: str, b: str) -> float:
    from scipy.stats import spearmanr
    return float(np.nanmean([spearmanr(df.loc[df.setname == s, a], df.loc[df.setname == s, b], nan_policy="omit")[0] for s in df.setname.unique()]))
