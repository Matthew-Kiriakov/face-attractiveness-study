"""Same person in the development sample and the SCUT lockbox? (IMAGE SIDE ONLY: ArcFace embeddings, no rating read; the lockbox labels stay closed; the author said yes 2026-11-09; written before the run.)
Embeddings: data/scut_embedding.npz (buffalo_l, 512-d, L2-normalised). For each lockbox face: cosine to its nearest development face (non-lockbox faces of the pooled sample, as in the main runs). Reported: share of lockbox faces with a development neighbour above 0.5 / 0.6 / 0.7 (0.6 is the threshold of scut_duplicates_arcface.py); the same split by set; pairs inside the lockbox; for comparison the share of development faces with another development neighbour above 0.6 (2.5 % in scut_duplicates_arcface.py).
Reviewer's rule (2026-11-09): more than about 2 % of lockbox faces with a development twin above 0.6 means the dedup rule must be fixed before the lockbox run S.
Expectation (AI, before): 1.5-4 % of the lockbox faces with a development twin above 0.6 (as dev-dev 2.5 %), mostly in the DataTang sets.
Usage: python scut_dup_lockbox.py"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_common as O

R = O.R


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df, *_ = O.load()
    lock = pd.read_csv(R / "data/lockbox.csv")
    lk = lock[lock.lock].file.values
    z = np.load(R / "data/scut_embedding.npz", allow_pickle=True)
    idx = {f: i for i, f in enumerate(z["files"])}
    lk = np.array([f for f in lk if f in idx])
    dev = df.file.values
    E = z["emb"].astype(float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    Xd = E[[idx[f] for f in dev]]
    Xl = E[[idx[f] for f in lk]]
    S = Xl @ Xd.T
    nn = S.max(axis=1)
    lines = [f"lockbox faces with embeddings {len(lk)}, development faces {len(dev)}"]
    lines.append(f"nearest development neighbour of a lockbox face: median {np.median(nn):.3f}, 90 % {np.percentile(nn, 90):.3f}, 99 % {np.percentile(nn, 99):.3f}, max {nn.max():.3f}")
    for t in (0.5, 0.6, 0.7):
        lines.append(f"  lockbox faces with a development neighbour above {t}: {int((nn > t).sum())} ({100 * (nn > t).mean():.1f} %)")
    sets = pd.Series([f[:2] + ("-int" if lock.set_index('file').loc[f, 'src'] == 'internet' else "-dt") for f in lk])
    lines.append("by group x source at 0.6: " + ", ".join(f"{g} {int(((nn > 0.6) & (sets.values == g)).sum())}/{int((sets.values == g).sum())}" for g in sorted(sets.unique())))
    Sd = Xd @ Xd.T
    np.fill_diagonal(Sd, -1)
    nd = Sd.max(axis=1)
    lines.append(f"reference: development faces with another development neighbour above 0.6: {100 * (nd > 0.6).mean():.1f} %")
    Sl = Xl @ Xl.T
    np.fill_diagonal(Sl, -1)
    lines.append(f"inside the lockbox: faces with another lockbox neighbour above 0.6: {int((Sl.max(axis=1) > 0.6).sum())} ({100 * (Sl.max(axis=1) > 0.6).mean():.1f} %)")
    lines.append(f"rule of the reviewer (more than about 2 % -> fix the dedup rule): {'TRIGGERED' if (nn > 0.6).mean() > 0.02 else 'not triggered'}")
    text = "\n".join(lines)
    print(text)
    (R / "results" / f"{date.today().isoformat()}_scut_dup_lockbox.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
