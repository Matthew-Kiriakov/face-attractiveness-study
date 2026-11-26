"""MEBeauty transfer (registered in RESEARCH.md section 1e). Two stages.
 predict : LABEL-FREE. Fits the frozen SCUT tier-2 models (HistGradientBoosting and Ridge, settings of scut_cm_harmony: hgb(), ridge()) on the 3737 non-lockbox SCUT faces (features z inside set, target y = judges' z inside set) and predicts the MEBeauty faces (features winsorised 1/99 % and z-scored inside gender x ethnicity on cv_fold 1-4 faces only; fold 0 is predicted with the same group statistics but is never scored). Writes data/external/mebeauty/transfer_predictions.csv (no ratings of any kind).
 score   : reads the MEBeauty beauty_score. REFUSES to run unless FREEZE.md exists in the project root and contains the line 'STATUS: FROZEN'. Variants A, A2, C, D as registered (B is a later registration). Writes results/<date>_mebeauty_transfer.txt.
Metrics: Spearman and Pearson inside groups with face-bootstrap intervals (500); 'transfers' = lower bound of the interval above 0 in all four primary groups (caucasian_female, caucasian_male, asian_female, asian_male); 'useful' = mean of the four primary Pearson correlations >= 0.30 (both for the boosting model; ridge reported next to it). Expectation (AI, before; RESEARCH 1e): primary correlations 0.30-0.50.
Usage: python mebeauty_transfer.py predict | score"""
import glob
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_common as O
import scut_cm_harmony as S

R = O.R
MB = R / "data/external/mebeauty"
PRIMARY = ["caucasian_female", "caucasian_male", "asian_female", "asian_male"]


def zgroup(d, cols, key, ref_mask):
    """Winsorise 1/99 % and z-score inside each group with statistics from the reference faces only."""
    out = d.copy()
    for g in d[key].unique():
        m = (d[key] == g).values
        mr = m & ref_mask
        for c in cols:
            lo, hi = d.loc[mr, c].quantile([0.01, 0.99])
            v = d.loc[m, c].clip(lo, hi)
            vr = d.loc[mr, c].clip(lo, hi)
            med = vr.median()
            vr = vr.fillna(med)
            out.loc[m, c] = (v.fillna(med) - vr.mean()) / vr.std()
    return out


def mb_features(tier2):
    mb = pd.read_csv(MB / "mebeauty_features.csv")
    mb["group"] = mb.ethnicity + "_" + mb.gender
    z = zgroup(mb, tier2, "group", (mb.cv_fold != 0).values)
    flags = pd.read_csv(MB / "mebeauty_flags.csv")[["file", "flag_any"]].rename(columns={"flag_any": "flag"})  # sd, roll, yaw, mouth, hair rules of mebeauty_flags.py
    return z.merge(flags, on="file", how="left")


def predict():
    sys.stdout.reconfigure(encoding="utf-8")
    df, props, rest, tier2 = O.load()
    mb = mb_features(tier2)
    Xs, ys = df[tier2].values.astype(float), df.y.values
    Xm = mb[tier2].values.astype(float)
    out = mb[["image_id", "file", "group", "cv_fold", "flag"]].copy()
    for name, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
        m = fac()
        m.fit(Xs, ys)
        out["pred_" + name] = m.predict(Xm)
    out.to_csv(MB / "transfer_predictions.csv", index=False)
    print(f"predictions for {len(out)} MEBeauty faces written (no ratings read); columns {list(out.columns)}")
    print("prediction SD per model:", {c: round(float(out[c].std()), 3) for c in out.columns if c.startswith("pred_")})
    print("done")


def boot(x, y, fn, rng, n=500):
    r = fn(x, y)[0]
    bs = [fn(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return r, *np.nanpercentile(bs, [2.5, 97.5])


def labels():
    parts = [pq.read_table(f, columns=["image_id", "beauty_score", "cv_fold"]).to_pandas() for f in sorted(glob.glob(str(MB / "fbp_extended/*.parquet")))]
    d = pd.concat(parts, ignore_index=True)
    return d[d.cv_fold != 0][["image_id", "beauty_score"]]  # fold 0 labels are dropped at once and never used


def score():
    fr = R / "FREEZE.md"
    allow = MB / "ALLOW_LABELS.md"
    if not (fr.exists() and "STATUS: FROZEN" in fr.read_text(encoding="utf-8")) and not allow.exists():
        sys.exit("REFUSED: neither FREEZE.md with 'STATUS: FROZEN' nor data/external/mebeauty/ALLOW_LABELS.md (the author's explicit permission) exists; labels are not read (RESEARCH 1e).")
    print("label gate:", "FREEZE.md" if fr.exists() and "STATUS: FROZEN" in fr.read_text(encoding="utf-8") else "ALLOW_LABELS.md (exploratory permission)")
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(20261108)
    pr = pd.read_csv(MB / "transfer_predictions.csv").merge(labels(), on="image_id")
    pr = pr[pr.cv_fold != 0].copy()
    pr["y"] = pr.groupby("group").beauty_score.transform(lambda v: (v - v.mean()) / v.std())
    lines = []

    def block(title, d):
        lines.append(f"\n{title}  (n = {len(d)})")
        res = {}
        for model in ("boosting", "ridge"):
            lines.append(f"  model {model}")
            for g in PRIMARY + sorted(set(d.group) - set(PRIMARY)):
                s = d[d.group == g]
                if len(s) < 30:
                    continue
                p = boot(s["pred_" + model].values, s.y.values, pearsonr, rng)
                sp = boot(s["pred_" + model].values, s.y.values, spearmanr, rng)
                tag = "primary" if g in PRIMARY else "other"
                lines.append(f"    {g:18s} n {len(s):4d} {tag:8s} Pearson {p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}]  Spearman {sp[0]:+.3f} [{sp[1]:+.3f}, {sp[2]:+.3f}]")
                res[(model, g)] = p
            lo = [res[(model, g)][1] for g in PRIMARY if (model, g) in res]
            mean4 = np.mean([res[(model, g)][0] for g in PRIMARY if (model, g) in res])
            lines.append(f"    mean of the four primary Pearson {mean4:.3f}; lower bounds above 0 in all four: {all(v > 0 for v in lo) and len(lo) == 4}; useful (>= 0.30): {mean4 >= 0.30}")

    block("A  frozen SCUT models -> MEBeauty folds 1-4", pr)
    block("A2 the same without flagged faces", pr[~pr.flag.fillna(False).astype(bool)])
    df, props, rest, tier2 = O.load()
    mb = mb_features(tier2).merge(labels(), on="image_id")
    tr = mb[mb.cv_fold != 0].copy()
    tr["y"] = tr.groupby("group").beauty_score.transform(lambda v: (v - v.mean()) / v.std())
    lines.append("\nC  trained on MEBeauty folds 1-4 -> SCUT non-lockbox, inside each SCUT set")
    for name, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
        m = fac()
        m.fit(tr[tier2].values.astype(float), tr.y.values)
        p = m.predict(df[tier2].values.astype(float))
        lines.append(f"  model {name}: " + ", ".join(f"{s} r {np.corrcoef(p[(df.setname == s).values], df.y.values[(df.setname == s).values])[0, 1]:+.3f}" for s in df.setname.unique()))
    lines.append("\nD  trained on the Caucasian groups of folds 1-4 -> each other group (descriptive)")
    base = tr[tr.ethnicity == "caucasian"]
    for name, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
        m = fac()
        m.fit(base[tier2].values.astype(float), base.y.values)
        row = []
        for g in sorted(tr.group.unique()):
            s = tr[tr.group == g]
            p = m.predict(s[tier2].values.astype(float))
            row.append(f"{g} r {np.corrcoef(p, s.y.values)[0, 1]:+.3f}" + (" (training)" if g.startswith("caucasian") else ""))
        lines.append(f"  model {name}: " + ", ".join(row))
    text = "\n".join(lines)
    print(text)
    (R / "results" / f"{date.today().isoformat()}_mebeauty_transfer.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    {"predict": predict, "score": score}[sys.argv[1]]()
