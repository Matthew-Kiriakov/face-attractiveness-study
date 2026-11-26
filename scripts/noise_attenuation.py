"""Landmark noise attenuation by simulation extrapolation (SIMEX). EXPLORATORY, non-lockbox faces only.

Rules fixed before the run:
 - Every landmark that the 42 features use (POINTS in js/metrics.js, 42 of them) gets independent Gaussian noise of SD sigma*IPD per axis
   (IPD = distance between the two iris centres, per face); hairline and glabella points are left alone.
 - sigma_extra in 0, .01, .02, .03, .045 per axis, 3 noise draws each, features recomputed by the page's own formulas (measure_batch.mjs),
   then held-out r of ridge and boosting (5-fold x 2 repeats, fixed settings) against the judges' mean.
 - Present noise of the detector: the hand check gave median radial error 0.030 IPD for detector vs hand and 0.016 for hand vs hand.
   A 2D Gaussian has median radial error 1.177*sigma, so total 0.0255 and hand 0.0136 per axis; detector-own sigma0 = sqrt(0.0255^2-0.0136^2) = 0.0216.
 - Extrapolation: r as a quadratic function of lambda = sigma_extra^2, evaluated at lambda = -sigma0^2 (the noise-free landmarks).
 Sets: CM 10k US, AM DataTang, AF DataTang, CF 10k US (the four largest-n within-source sets). Checkpointed in data/cache_expl/noise.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exploration_v1 as E
import scut_cm_harmony as S

R = S.ROOT
TMP = R / "data/cache_expl"
LEVELS = [0.0, 0.01, 0.02, 0.03, 0.045]
DRAWS = 3
SIGMA0 = float(np.sqrt(0.0255 ** 2 - 0.0136 ** 2))
POINT_IDX = [10, 152, 168, 1, 2, 6, 197, 195, 98, 327, 45, 48, 275, 278, 33, 133, 159, 145, 468, 263, 362, 386, 374, 473, 46, 107, 276, 336, 61, 291, 0, 37, 267, 13, 14, 17, 234, 454, 172, 397]
SETS = [("CM", "10k_US"), ("AM", "DataTang"), ("AF", "DataTang"), ("CF", "10k_US")]


def cache_for(g: str) -> dict:
    return S.load_done(S.CACHE) if g == "CM" else S.load_done(R / "data" / f"scut_{g.lower()}_landmarks_web.jsonl")


def features(rows: list[dict], props: list[str]) -> pd.DataFrame:
    inp, out = TMP / "noise_in.jsonl", TMP / "noise_out.json"
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    subprocess.run([str(S.NODE), str(S.MEASURE_JS), str(inp), str(out)], check=True)
    res = json.loads(out.read_text(encoding="utf-8"))["rows"]
    return pd.DataFrame([{"file": r["file"], **{p: r["values"].get(p) for p in props}} for r in res if not r.get("error") and r.get("values")]).set_index("file")


def main() -> None:
    sets, props = E.load()
    p = TMP / "noise.json"
    done = json.loads(p.read_text()) if p.exists() else {}
    for g, s in SETS:
        d = sets[g][sets[g].src == s].set_index("file")
        cache = cache_for(g)
        base = [cache[f] for f in d.index if f in cache]
        files = [r["file"] for r in base]
        y = d.loc[files, "rating_mean"].values
        for sig in LEVELS:
            for k in range(DRAWS if sig > 0 else 1):
                key = f"{g}|{s}|{sig}|{k}"
                if key in done:
                    continue
                rng = np.random.default_rng(S.SEED + k)
                rows = []
                for r in base:
                    lm = np.array(r["lm"], dtype=float)
                    ipd = float(np.hypot(*((lm[468] - lm[473]) * np.array([r["w"], r["h"]]))))
                    new = lm.copy()
                    for i in POINT_IDX:
                        new[i] += rng.normal(0, sig * ipd, 2) / np.array([r["w"], r["h"]])
                    rows.append({**r, "lm": new.tolist()})
                X = features(rows, props).loc[files]
                res = {}
                for name, fac in (("ridge", S.ridge), ("boosting", S.hgb)):
                    pr = S.oof_predict(fac, X.values.astype(float), y, 2, S.SEED)
                    res[name] = float(np.corrcoef(pr, y)[0, 1])
                done[key] = res
                p.write_text(json.dumps(done))
                print(key, {n: round(v, 3) for n, v in res.items()}, flush=True)
    print("\nSIMEX extrapolation (quadratic in lambda = sigma_extra^2 to lambda = -sigma0^2), sigma0 = %.4f per axis" % SIGMA0)
    for g, s in SETS:
        for name in ("ridge", "boosting"):
            lam = np.array(LEVELS) ** 2
            r = np.array([np.mean([done[f"{g}|{s}|{sig}|{k}"][name] for k in range(DRAWS if sig > 0 else 1)]) for sig in LEVELS])
            c = np.polyfit(lam, r, 2)
            r0 = float(np.polyval(c, -SIGMA0 ** 2))
            print(f"  {g} {s:9s} {name:8s} r by sigma_extra {np.round(r, 3).tolist()} -> noise-free {r0:.3f} (gain {r0 - r[0]:+.3f}); relative {r[0] / r0:.3f}")


if __name__ == "__main__":
    main()
