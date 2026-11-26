"""Confirmatory step S: the SCUT lockbox, ONE run (FREEZE.md v1 + amendment v1.1; protocol RESEARCH 1d).
Modes:
  dry : features only. Checks coverage of the lockbox faces (missing columns, exclusions, duplicates, set sizes). Rating columns are dropped at once and never printed.
  dev : DEVELOPMENT faces only (the lockbox is not touched). Prints the reference numbers used for the expectations below (random 5-fold, 3 seeds, boosting and ridge).
  run : the lockbox run. Refuses unless EXPECTATIONS_WRITTEN is True and data/lockbox_S_DONE does not exist; creates that file at the start (one run only, never repeated after seeing a result).
  smoke : the same code as run on a random 25 % of the DEVELOPMENT faces posing as lockbox (no lockbox face is read, no DONE file); only to catch bugs.
  ldev : (v1.3) the L block (lookmaxxing claims) on ALL development faces, measures only; prints the reference for the L expectations below; no lockbox face is read.
Usage: python scut_lockbox_S.py dry | dev | ldev | smoke | run [--rerun]
v1.4 (2026-11-11): in 'run' the numbers go only to results/S_lockbox_run.txt.part (flushed per line; the console shows stage names); inputs and library versions are verified against scripts/input_hashes.json BEFORE the marker is written;
  the marker is JSON (time, script sha256, versions); '--rerun' is the ONE rerun after a crash allowed by FREEZE v1.4; predictive R2 is printed next to r; per-face predictions are saved to data/lockbox_S_predictions.csv; per-set counts for H1, H2, H3 are printed (no overall word for H1).

Definitions (written 2026-11-09 before any lockbox rating is read; [AI] design, frozen by the author's yes to FREEZE v1.1):
 Sets: the six within-source sets of the development analysis: CM 10k_US, AM DataTang, AM internet, CF 10k_US, AF internet, AF DataTang (lockbox faces of lab / GuangZhou sources are not in any set and are only counted).
 Pipeline: features = tier 2 (42 proportions + 28 face features; overnight_common.load list), uncorrected; winsorise 1/99 % and z-score with the DEVELOPMENT statistics of the same set (no lockbox statistic);
   missing -> development median of the set. Exclusion: any of the 42 proportions beyond 50 IQR of the development median of its group (the rule excluded one development photograph). Target = judges' mean, z with development statistics (Pearson does not depend on it).
 Models: fitted ONCE on the 3737 development faces pooled (as in the main runs): boosting (primary) and ridge, tier 2; and the same on the 42 proportions alone (geometry).
 Duplicates: lockbox faces with a development neighbour above 0.6 (ArcFace cosine, data/scut_embedding.npz) are 'twins'. PRIMARY = without twins; SECONDARY = all lockbox faces. Both are printed everywhere.
 H1: Pearson r inside each set, face bootstrap 1000, 95 % interval; band 0.40-0.70 applies to GEOMETRY (boosting and ridge); tier 2 reported next to it. Pooled r printed for display only.
 H2: inside each set, for each judge j: r(model, mean of the other 59 judges) minus r(judge j, the same mean); mean over the 60 judges; paired face bootstrap 1000. Supported only if the interval is above 0. (Model = boosting tier 2.)
 H3: for each test set T: model trained on (a) the development faces of T's own set, (b) the development faces of the OTHER population (ethnicity), (c) the development faces of the SAME population, OTHER sex; each pool subsampled to the size of the smallest
   of the three (20 random subsamples, predictions averaged), boosting tier 2. Population gap = r(a) - r(b); sex gap = r(a) - r(c); paired face bootstrap 1000. Supported (population) if the gap >= 0.10 with the interval above 0.
 H5: inside each lockbox set (CV inside the lockbox, not a test of the frozen model), ridge, commonality analysis of geometry G (42), capture Q (6), age A (1): unique shares from variance_partition.partition on 30 random 80 % subsamples; mean and 2.5-97.5 percentiles (a stability range, not a confidence interval).
 H4 stays exploratory (not in this script). Sets with fewer than 100 lockbox faces after exclusions carry no verdict.

EXPECTATIONS (AI, written 2026-11-09 before the run; development reference from 'dev' mode, results/2026-11-09_S_dev_reference.txt; development r is probably optimistic: the 112 features were screened on those faces, and ArcFace twins share folds):
 Development r, random 5-fold, boosting tier 2: CM 0.636, AM DataTang 0.644, AM internet 0.800, CF 0.616, AF internet 0.769, AF DataTang 0.705 (mean 0.695); ridge tier 2 mean 0.674; geometry boosting: 0.596, 0.569, 0.746, 0.551, 0.708, 0.619 (mean 0.631); geometry ridge mean 0.594.
 EXPECTED lockbox r, PRIMARY (no twins), centre value with +-0.08 as the 'as expected' band:
   tier 2 boosting: CM 0.60, AM DataTang 0.61, AM internet 0.76, CF 0.58, AF internet 0.73, AF DataTang 0.67 (development minus about 0.03); ridge 0.01-0.04 below boosting in every set;
   geometry boosting: CM 0.56, AM DataTang 0.54, AM internet 0.71, CF 0.52, AF internet 0.67, AF DataTang 0.59 -> H1 band 0.40-0.70 holds for four sets; AM internet and AF internet are at or above the upper edge (real risk of 'above 0.70').
 - Excluding the 50 twins changes r by -0.03 to +0.01 in a set (AM internet and AF internet have most twins).
 - H2: not supported in CM and CF 10k_US; possibly supported in the DataTang sets; H3: population gap >= 0.10 in the Asian sets, sex gap < 0.10; both mostly undecided (interval half-width about 0.10-0.17 at n 126-336).
 - H5: unique G share >= 0.25 in AM internet, AF internet, AF DataTang and about 0.00-0.15 in CM, CF and AM DataTang; unique Q <= 0.05 everywhere; unique A <= 0.05 except CM and CF 10k_US where 0.08-0.11 are expected. (The last two lines were adjusted AFTER the smoke run on development faces, which showed these values; written before the lockbox run.)
 If the mean of the six tier 2 boosting r falls below 0.60 (development minus 0.10) the claim 'explains about half of the variance inside source' is downgraded (FREEZE v1.1 point 8).

L (lookmaxxing claims; FREEZE v1.3, written 2026-11-11 before the run; claim list chosen by the author [mine], measures, rule and expectations [AI]):
 Eighteen tests for fifteen claims (C13-C15 are tested separately for men and women; C11 men, C12 women). Each claim is one measure fixed in advance (CLAIMS in the code), stated so that
 'sign * measure higher = rated higher'. Test: Spearman between the measure and the judges' z inside each lockbox set of the claim's sex group (six sets, or three for the sex-specific
 claims), face bootstrap 1000. One rule for all claims, over the sets with n >= 100: SUPPORTED if mean(sign*rho) >= 0.15 and the lower 95 % bound is above 0 in at least two thirds of the sets;
 CONTRADICTED if mean <= -0.15 and the upper bound is below 0 in two thirds; otherwise not distinguishable. C10 (symmetry): ridge fitted on the development faces with tier 2 plus the ten stored
 symmetry features versus tier 2 alone, gain in r on each lockbox set (paired face bootstrap); SUPPORTED if the mean gain >= 0.01 and the lower bound is above 0 in two thirds of the sets.
 PRIMARY = without twins, SECONDARY = all lockbox faces; both printed; all eighteen results are reported, none is dropped or chosen after seeing results.
 Measures: C01 chin-philtrum; C02 zyg_jaw (block 6); C03 mean of canthal-l and canthal-r; C04 eye-gap; C05 sclera_show_low; C06 face-wh (a width-to-height ratio, the closest existing column to fWHR, not the strict
 brow-to-lip fWHR); C07 sum over the three thirds of |share - 100/3|; C08 mean of brow-gap-l and brow-gap-r; C09 mean of z(brow_thick) and z(brow_dens); C10 ten symmetry features; C11 and C12 index M_core of block5
 (jaw, brow, eyes, nose, lips, shape; copied into COMPONENTS), femininity = -M_core; C13 jaw_turn; C14 adip_pa (perimeter^2/area of the face oval, LOWER = rounder and fuller, so higher = leaner);
 C15 jaw-w. z statistics are the development statistics of the set. Not testable and not in the list: jaw development, depth of the eye set, cheek volume (profile or 3D), hair and skin state, eye size, skin colour.
 The claims are not independent (jaw corner, jaw width, outline compactness and the jaw part of M_core are correlated, 0.8 and above for some pairs); they are associations on photographs inside a set, not causes.
 Reference: development faces, all 3737 (results/2026-11-11_L_dev_reference.txt, mode 'ldev'). The claim list was fixed with the earlier development table (results/2026-11-11_lookmax_candidates_dev.txt) in view, so
 the lockbox is a replication of directions already seen, not a blind test. Development means of sign*rho: C01 +0.13, C02 +0.29, C03 +0.26, C04 +0.30, C05 -0.22, C06 -0.17, C07 +0.17, C08 +0.36, C09 +0.15,
 C11 -0.14, C12 +0.26, C13M -0.27, C13F -0.30, C14M +0.36, C14F +0.40, C15M -0.25, C15F -0.34.
 EXPECTED lockbox verdicts, PRIMARY (the lockbox sets have 124-328 faces, so intervals are about three times wider than in development; claims near the 0.15 edge will mostly end as 'not distinguishable'):
  SUPPORTED: C02 cheekbones wide relative to the jaw, C03 canthal tilt, C04 eyes closer, C08 low-set brows, C12 feminine face in women, C14M and C14F tapering outline (higher perimeter^2/area) (in the weakest set, AM DataTang, +0.13, the bound may fail).
  CONTRADICTED: C05 less sclera below the iris (the opposite is seen), C13M and C13F sharper jaw corner, C15M and C15F wider jaw; C13M may fail through AM DataTang (-0.08).
  not distinguishable: C01 chin-philtrum, C06 width-to-height ratio (edge), C07 equal thirds (edge; mostly the height of the upper third, a noisy hairline marker), C09 thick dense brows (edge), C11 masculine in men.
  not supported: C10 symmetry (development gain -0.017 to +0.003).
 A verdict that differs from this list is recorded as a miss, not explained away.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_common as O
import scut_cm_harmony as S
import variance_partition as VP
import run_guard as RG

R = O.R
EXPECTATIONS_WRITTEN = True  # the EXPECTED block in the header holds numbers from 'dev' (2026-11-09)
EXPECTATIONS_L_WRITTEN = True  # the EXPECTED L block in the header holds verdicts written from 'ldev' (2026-11-11, v1.3)
SETS = ["CM 10k_US", "AM DataTang", "AM internet", "CF 10k_US", "AF internet", "AF DataTang"]
ETH = {"CM": "C", "CF": "C", "AM": "A", "AF": "A"}
SEX = {"CM": "M", "CF": "F", "AM": "M", "AF": "F"}
DONE = R / "data/lockbox_S_DONE"
OUT = []
RESULTS = R / "results" / "S_lockbox_run.txt"
PREDS = R / "data" / "lockbox_S_predictions.csv"
OUTO = RG.Out(None)


def say(s=""):
    OUT.append(s)
    OUTO.say(s)


def frames():
    """Raw (unz) per-group frames with all feature blocks, lock flag, setname; degenerate-landmark exclusion with development statistics."""
    cols = list(pd.read_csv(S.CSV_PATH, nrows=1).columns)
    props = cols[cols.index("face-wh"): cols.index("ala-jaw") + 1]
    lock = pd.read_csv(R / "data/lockbox.csv")[["file", "lock"]]
    age = pd.read_csv(R / "data/scut_age_mesh_all.csv")
    ext = [pd.read_csv(R / f"data/{n}.csv") for n in ("scut_colour_features", "scut_block2b_features", "scut_block2c_features", "scut_block3_symmetry", "scut_block5_dimorphism", "scut_block6_angularity")]
    base = {"CM": pd.read_csv(S.CSV_PATH).merge(pd.read_csv(R / "data/scut_cm_quality.csv"), on="file").merge(pd.read_csv(R / "data/scut_cm_source.csv"), on="file")}
    for g in ("AM", "CF", "AF"):
        base[g] = pd.read_csv(R / f"data/scut_{g.lower()}_harmony.csv")
    parts, excluded = [], []
    for g, d in base.items():
        d = d.merge(age[["file", "age_mesh"]], on="file").merge(lock, on="file")
        dev = d[~d.lock]
        med, iqr = dev[props].median(), dev[props].quantile(.75) - dev[props].quantile(.25)
        z = ((d[props] - med) / iqr).abs().max(axis=1)
        excluded += d.loc[z > 50, ["file", "lock"]].values.tolist()
        d = d[z <= 50].copy()
        for f in ext:
            d = d.merge(f, on="file", how="left")
        d["group"] = g
        d["setname"] = g + " " + d.src
        parts.append(d)
    return pd.concat(parts, ignore_index=True), props, excluded


def zfit(dev, cols):
    st = {}
    for c in cols:
        lo, hi = dev[c].quantile([0.01, 0.99])
        v = dev[c].clip(lo, hi)
        st[c] = (lo, hi, v.median(), v.mean(), v.std())
    return st


def zapply(st, d, cols):
    out = np.zeros((len(d), len(cols)))
    for j, c in enumerate(cols):
        lo, hi, med, mean, sd = st[c]
        out[:, j] = ((d[c].clip(lo, hi).fillna(med) - mean) / sd).values
    return out


def build():
    _, props, rest, tier2 = O.load()
    allf, props, excluded = frames()
    allf = allf[allf.setname.isin(SETS)].reset_index(drop=True)
    return allf, props, tier2, excluded


def twins(files_lock, files_dev):
    z = np.load(R / "data/scut_embedding.npz", allow_pickle=True)
    idx = {f: i for i, f in enumerate(z["files"])}
    E = z["emb"].astype(float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    have = np.array([f in idx for f in files_lock])
    nn = np.full(len(files_lock), np.nan)
    Xl = E[[idx[f] for f in np.array(files_lock)[have]]]
    Xd = E[[idx[f] for f in files_dev]]
    nn[have] = (Xl @ Xd.T).max(axis=1)
    return nn


def dry():
    allf, props, tier2, excluded = build()
    allf = allf.drop(columns=[c for c in ("rating_mean",) if c in allf.columns])
    say(f"dry run; rating columns dropped. Excluded as degenerate landmarks (file, lock): {excluded}")
    for s in SETS:
        d = allf[allf.setname == s]
        lk, dv = d[d.lock], d[~d.lock]
        miss = lk[tier2].isna().mean()
        say(f"  {s:12s} development {len(dv):4d}, lockbox {len(lk):4d}; lockbox missing share max {miss.max():.3f} ({miss.idxmax()}); columns absent: {[c for c in tier2 if c not in d.columns]}")
    say(f"  lockbox faces outside the six sets (lab / GuangZhou): counted by group x source in data/lockbox.csv")
    lk = allf[allf.lock]
    nn = twins(lk.file.values, allf[~allf.lock].file.values)
    say(f"  lockbox faces in the six sets: {len(lk)}; with embeddings {int(np.isfinite(nn).sum())}; twins (>0.6) {int((nn > 0.6).sum())}")
    say("done")


def models(Xd, y, cols_idx):
    out = {}
    for name, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
        m = fac()
        m.fit(Xd[:, cols_idx], y)
        out[name] = m
    return out


def dev():
    from sklearn.model_selection import KFold
    df, props, rest, tier2 = O.load()
    y = df.y.values
    say("DEV reference (development faces only): r inside each set, random 5-fold, mean of 3 seeds")
    for label, cols in (("tier2", tier2), ("geometry", props)):
        X = df[cols].values.astype(float)
        for name, fac in (("boosting", S.hgb), ("ridge", S.ridge)):
            p = np.zeros(len(y))
            for seed in O.SEEDS[:3]:
                for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
                    m = fac()
                    m.fit(X[tr], y[tr])
                    p[te] += m.predict(X[te]) / 3
            rs = {s: np.corrcoef(p[(df.setname == s).values], y[(df.setname == s).values])[0, 1] for s in SETS}
            say(f"  {label:9s} {name:9s} " + ", ".join(f"{s} {v:.3f}" for s, v in rs.items()) + f" | mean {np.mean(list(rs.values())):.3f}")
    (R / "results" / f"{date.today().isoformat()}_S_dev_reference.txt").write_text("\n".join(OUT), encoding="utf-8")


def pr2(p, y):
    """Predictive R2 on the z scale of the development statistics: 1 - SSE/SST."""
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def bootr(p, y, rng, n=1000):
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    return np.array([np.corrcoef(p[i], y[i])[0, 1] for i in idx])


def ci(a):
    return f"[{np.nanpercentile(a, 2.5):+.3f}, {np.nanpercentile(a, 97.5):+.3f}]"


def judge_matrix(files):
    a = pd.read_excel(R / "data/scut-fbp5500/SCUT-FBP5500_v2/All_Ratings.xlsx", sheet_name="ALL")
    a = a[a.Filename.isin(files)]
    return a.groupby(["Filename", "Rater"]).Rating.mean().unstack("Rater").reindex(files)


# ---------------------------------------------------------------- L: lookmaxxing claims (FREEZE v1.3)
LOOK_SETS = {"all": SETS, "M": [x for x in SETS if SEX[x[:2]] == "M"], "F": [x for x in SETS if SEX[x[:2]] == "F"]}
# (id, claim as stated in the lookmaxxing literature, measure key, sign, sets). A claim is stated so that 'sign * measure higher = rated higher'.
CLAIMS = [
    ("C01", "longer chin relative to the upper lip (short philtrum)", "chin-philtrum", +1, "all"),
    ("C02", "cheekbones wide relative to the jaw", "zyg_jaw", +1, "all"),
    ("C03", "positive canthal tilt (mean of both eyes)", "canthal", +1, "all"),
    ("C04", "eyes set closer together (gap between the eyes to eye width)", "eye-gap", -1, "all"),
    ("C05", "less white visible below the iris", "sclera_show_low", -1, "all"),
    ("C06", "wider face relative to its height", "face-wh", +1, "all"),
    ("C07", "closer to equal vertical thirds", "thirds_dev", -1, "all"),
    ("C08", "lower-set brows (small brow-to-lid gap)", "brow_gap", -1, "all"),
    ("C09", "thick and dense brows", "brow_td", +1, "all"),
    ("C10", "symmetry (gain of the 10 stored symmetry features over tier 2)", "SYM", +1, "all"),
    ("C11", "masculine face rated higher (men)", "M_core", +1, "M"),
    ("C12", "feminine face rated higher (women)", "M_core", -1, "F"),
    ("C13M", "sharper jaw corner (men)", "jaw_turn", +1, "M"),
    ("C13F", "sharper jaw corner (women)", "jaw_turn", +1, "F"),
    ("C14M", "tapering (less round) face outline: higher perimeter^2/area (men)", "adip_pa", +1, "M"),
    ("C14F", "tapering (less round) face outline: higher perimeter^2/area (women)", "adip_pa", +1, "F"),
    ("C15M", "wider jaw relative to the face (men)", "jaw-w", +1, "M"),
    ("C15F", "wider jaw relative to the face (women)", "jaw-w", +1, "F"),
]
# dimorphism index of block5_dimorphism.py (the author's masculine directions), copied so that the frozen script does not depend on that file
COMPONENTS = {
    "jaw": [("jaw-w", 1), ("jaw_turn", 1), ("chin_w", 1)],
    "brow": [("brow-gap-l", -1), ("brow-gap-r", -1), ("brow_thick", 1), ("brow_dens", 1), ("ridge_dL", 1)],
    "eyes": [("eye-face", -1)],
    "nose": [("nose-face", 1)],
    "lips": [("lip_thick", -1)],
    "shape": [("face-wh", 1)],
}
CORE = ["jaw", "brow", "eyes", "nose", "lips", "shape"]
SYMCOLS = ["sym_eyes", "sym_brows", "sym_mouth", "sym_nose", "sym_jaw", "sym_rms", "sym_h", "sym_v", "sym_axis", "sym_tilt"]
L_THRESHOLD = 0.15
L_GAIN = 0.01
L_COLS = sorted({c for v in COMPONENTS.values() for c, _ in v} | {"brow_thick", "brow_dens"})


def look_measures(d, st):
    """Per-face measures of the claims for the frame d (raw columns); st = zfit statistics of the same set (development)."""
    m = pd.DataFrame(index=d.index)
    for c in ("chin-philtrum", "zyg_jaw", "eye-gap", "sclera_show_low", "face-wh", "adip_pa", "jaw_turn", "jaw-w"):
        m[c] = d[c]
    m["canthal"] = d[["canthal-l", "canthal-r"]].mean(axis=1)
    m["brow_gap"] = d[["brow-gap-l", "brow-gap-r"]].mean(axis=1)
    m["thirds_dev"] = (d[["third-upper", "third-mid", "third-low"]] - 100 / 3).abs().sum(axis=1)
    Z = pd.DataFrame(zapply(st, d, L_COLS), columns=L_COLS, index=d.index)
    m["brow_td"] = Z[["brow_thick", "brow_dens"]].mean(axis=1)
    m["M_core"] = np.mean([np.mean([sg * Z[c] for c, sg in COMPONENTS[k]], axis=0) for k in CORE], axis=0)
    return m


def sboot(x, y, rng, n=1000):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r0 = spearmanr(x, y)[0]
    bs = np.array([spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))])
    return r0, bs


def look_verdict(vals, los, his, gain=False):
    k = len(vals)
    if k == 0:
        return "no verdict (no set with n >= 100)"
    need = int(np.ceil(2 * k / 3))
    mean = float(np.mean(vals))
    if gain:
        return "SUPPORTED" if mean >= L_GAIN and sum(lo > 0 for lo in los) >= need else "not supported"
    if mean >= L_THRESHOLD and sum(lo > 0 for lo in los) >= need:
        return "SUPPORTED"
    if mean <= -L_THRESHOLD and sum(hi < 0 for hi in his) >= need:
        return "CONTRADICTED"
    return "not distinguishable"


def lookmax(dev_all, lk_all, Xd, Xl, yd, yl, tier2, rng, ref=False):
    """L block. ref=True: development faces only (measures only, no model), used by 'ldev'. Rule (identical for every claim): over the sets of the claim with n >= 100,
    SUPPORTED if the mean of sign*Spearman >= 0.15 and the lower 95 % face-bootstrap bound is above 0 in at least two thirds of them; CONTRADICTED if the mean <= -0.15 and the
    upper bound is below 0 in at least two thirds; otherwise not distinguishable. C10: gain in r of ridge tier 2 + 10 symmetry features over tier 2, SUPPORTED if the mean gain >= 0.01
    and the lower bound is above 0 in two thirds of the sets."""
    say("\n=== L  lookmaxxing claims (FREEZE v1.3): per-face measure vs judges' z inside each set; Spearman, face bootstrap 1000; sign-adjusted so that + means 'the claim holds'")
    M = {c: np.full(len(lk_all), np.nan) for c in set(k for _, _, k, _, _ in CLAIMS) - {"SYM"}}
    for s in SETS:
        md, ml = (dev_all.setname == s).values, (lk_all.setname == s).values
        st = zfit(dev_all[md], L_COLS)
        mm = look_measures(lk_all[ml], st)
        for k in M:
            M[k][ml] = mm[k].values
    if ref:
        pops = (("ALL DEVELOPMENT FACES", np.ones(len(lk_all), bool)),)
    else:
        pops = (("PRIMARY (without twins)", ~lk_all.twin.values), ("SECONDARY (all lockbox faces)", np.ones(len(lk_all), bool)))
    summary = {}
    for pop, mask in pops:
        say(f"\n{pop}")
        for cid, text, key, sg, scope in CLAIMS:
            if key == "SYM":
                continue
            vals, los, his, parts = [], [], [], []
            for s in LOOK_SETS[scope]:
                m = mask & (lk_all.setname == s).values
                if m.sum() < 100:
                    parts.append(f"{s} n {m.sum()} (no verdict)")
                    continue
                r0, bs = sboot(M[key][m], yl[m], rng)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                if sg < 0:
                    r0, lo, hi = -r0, -hi, -lo
                vals.append(r0)
                los.append(lo)
                his.append(hi)
                parts.append(f"{s} {r0:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            v = look_verdict(vals, los, his)
            summary[(pop, cid)] = v
            say(f"  {cid:5s} {text}: " + "; ".join(parts) + f" | mean {np.mean(vals) if vals else float('nan'):+.3f} -> {v}")
        if ref:
            say("  C10 symmetry gain: not evaluated on development faces here (no held-out predictions); development reference is claim 42 in RESEARCH (gain -0.017...+0.003)")
            continue
        Zd, Zl = np.zeros((len(dev_all), len(SYMCOLS))), np.zeros((len(lk_all), len(SYMCOLS)))
        for s in SETS:
            md, ml = (dev_all.setname == s).values, (lk_all.setname == s).values
            st = zfit(dev_all[md], SYMCOLS)
            Zd[md] = zapply(st, dev_all[md], SYMCOLS)
            Zl[ml] = zapply(st, lk_all[ml], SYMCOLS)
        Xd2, Xl2 = np.hstack([Xd, Zd]), np.hstack([Xl, Zl])
        res = {}
        for name, fac in (("ridge", S.ridge), ("boosting", S.hgb)):
            m0, m1 = fac(), fac()
            m0.fit(Xd, yd)
            m1.fit(Xd2, yd)
            res[name] = (m0.predict(Xl), m1.predict(Xl2))
        for name in ("ridge", "boosting"):
            p0, p1 = res[name]
            pp, vv, ll, hh = [], [], [], []
            for s in SETS:
                m = mask & (lk_all.setname == s).values
                if m.sum() < 100:
                    pp.append(f"{s} n {m.sum()} (no verdict)")
                    continue
                a0, a1, yy = p0[m], p1[m], yl[m]
                g = np.corrcoef(a1, yy)[0, 1] - np.corrcoef(a0, yy)[0, 1]
                idx = [rng.integers(0, m.sum(), m.sum()) for _ in range(1000)]
                gb = np.array([np.corrcoef(a1[i], yy[i])[0, 1] - np.corrcoef(a0[i], yy[i])[0, 1] for i in idx])
                lo, hi = np.percentile(gb, [2.5, 97.5])
                vv.append(g)
                ll.append(lo)
                hh.append(hi)
                pp.append(f"{s} {g:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            if name == "ridge":
                v = look_verdict(vv, ll, hh, gain=True)
                summary[(pop, "C10")] = v
                say("  C10   symmetry: gain in r over tier 2, ridge (primary model of C10): " + "; ".join(pp) + f" | mean {np.mean(vv) if vv else float('nan'):+.4f} -> {v}")
            else:
                say("        symmetry gain, boosting (secondary, no verdict): " + "; ".join(pp))
    say("\nL summary: " + "; ".join(f"{pop.split()[0]} {cid} {v}" for (pop, cid), v in summary.items()))


def ldev():
    allf, props, tier2, excluded = build()
    dev_all = allf[~allf.lock].reset_index(drop=True)
    yd = np.zeros(len(dev_all))
    for s in SETS:
        md = (dev_all.setname == s).values
        yd[md] = ((dev_all.loc[md, "rating_mean"] - dev_all.loc[md, "rating_mean"].mean()) / dev_all.loc[md, "rating_mean"].std()).values
    say("L reference on DEVELOPMENT faces only (the lockbox is not read); written 2026-11-11 as the basis of the L expectations")
    lookmax(dev_all, dev_all, None, None, yd, yd, tier2, np.random.default_rng(20261111), ref=True)
    (R / "results" / f"{date.today().isoformat()}_L_dev_reference.txt").write_text("\n".join(OUT), encoding="utf-8")


def run(smoke=False, rerun=False):
    global OUTO
    if not smoke and not EXPECTATIONS_WRITTEN:
        sys.exit("REFUSED: EXPECTATIONS_WRITTEN is False (expectations must be filled from the dev reference before the run).")
    if not smoke and not EXPECTATIONS_L_WRITTEN:
        sys.exit("REFUSED: EXPECTATIONS_L_WRITTEN is False (the L expectations must be filled from the 'ldev' reference before the run).")
    if not smoke:
        OUTO = RG.Out(RG.start(DONE, RESULTS, Path(__file__), rerun, "S"))
    rng = np.random.default_rng(20261109)
    allf, props, tier2, excluded = build()
    if smoke:
        allf = allf[~allf.lock].reset_index(drop=True)
        allf["lock"] = np.random.default_rng(1).random(len(allf)) < 0.25
    dev_all = allf[~allf.lock].copy()
    lk_all = allf[allf.lock].copy()
    # z with development statistics of the same set
    Xd = np.zeros((len(dev_all), len(tier2)))
    Xl = np.zeros((len(lk_all), len(tier2)))
    yd = np.zeros(len(dev_all))
    yl = np.zeros(len(lk_all))
    for s in SETS:
        md, ml = (dev_all.setname == s).values, (lk_all.setname == s).values
        st = zfit(dev_all[md], tier2)
        Xd[md] = zapply(st, dev_all[md], tier2)
        Xl[ml] = zapply(st, lk_all[ml], tier2)
        mu, sd = dev_all.loc[md, "rating_mean"].mean(), dev_all.loc[md, "rating_mean"].std()
        yd[md] = (dev_all.loc[md, "rating_mean"] - mu) / sd
        yl[ml] = (lk_all.loc[ml, "rating_mean"] - mu) / sd
    nn = twins(lk_all.file.values, dev_all.file.values)
    lk_all["twin"] = nn > 0.6
    gi = [tier2.index(c) for c in props]
    allci = list(range(len(tier2)))
    mods = {("tier2", k): m for k, m in models(Xd, yd, allci).items()}
    mods.update({("geometry", k): m for k, m in models(Xd, yd, gi).items()})
    pred = {k: m.predict(Xl[:, gi if k[0] == "geometry" else allci]) for k, m in mods.items()}
    OUTO.stage("model fitted; H1")
    say(f"S: lockbox faces in the six sets {len(lk_all)} (twins {int(lk_all.twin.sum())}); excluded as degenerate: {[e for e in excluded if e[1]]}")

    say("\n=== H1  r inside each set (face bootstrap 1000), and predictive R2 (1 - SSE/SST on the z scale). Band 0.40-0.70 applies to geometry; judged per set on the point estimate (FREEZE v1.4), no overall word.")
    band = {"boosting": [], "ridge": []}
    for pop, mask in (("PRIMARY (without twins)", ~lk_all.twin.values), ("SECONDARY (all lockbox faces)", np.ones(len(lk_all), bool))):
        say(f"\n{pop}")
        for key in (("tier2", "boosting"), ("tier2", "ridge"), ("geometry", "boosting"), ("geometry", "ridge")):
            row = []
            for s in SETS:
                m = mask & (lk_all.setname == s).values
                if m.sum() < 100:
                    row.append(f"{s} n {m.sum()} (no verdict)")
                    continue
                b = bootr(pred[key][m], yl[m], rng)
                rr = np.corrcoef(pred[key][m], yl[m])[0, 1]
                row.append(f"{s} n {m.sum()} r {rr:.3f} {ci(b)} R2 {pr2(pred[key][m], yl[m]):+.3f}")
                if key[0] == "geometry" and pop.startswith("PRIMARY"):
                    band[key[1]].append(("below" if rr < 0.40 else "above" if rr > 0.70 else "inside"))
            pooled = np.corrcoef(pred[key][mask], yl[mask])[0, 1]
            say(f"  {key[0]:8s} {key[1]:8s} " + "; ".join(row) + f" | pooled (display only) {pooled:.3f}")

    say("\nH1 count, geometry, PRIMARY, point estimates vs band 0.40-0.70: " + "; ".join(f"{k}: {v.count('inside')} inside, {v.count('above')} above, {v.count('below')} below" for k, v in band.items()))
    OUTO.stage("H2")
    h2n = {"PRIMARY": [0, 0], "SECONDARY": [0, 0]}
    say("\n=== H2  model r with the mean of the other 59 judges minus single judge r with the same mean (boosting tier 2; mean over 60 judges; paired bootstrap)")
    for pop, mask in (("PRIMARY", ~lk_all.twin.values), ("SECONDARY", np.ones(len(lk_all), bool))):
        for s in SETS:
            m = mask & (lk_all.setname == s).values
            if m.sum() < 100:
                continue
            M = judge_matrix(lk_all.file.values[m]).values
            p = pred[("tier2", "boosting")][m]

            def stat(i):
                Mi = M[i]
                ti, ci_ = np.nansum(Mi, axis=1, keepdims=True), np.sum(np.isfinite(Mi), axis=1, keepdims=True)
                others = (ti - np.nan_to_num(Mi)) / (ci_ - np.isfinite(Mi))
                gm, js = [], []
                for j in range(Mi.shape[1]):
                    ok = np.isfinite(Mi[:, j])
                    gm.append(np.corrcoef(p[i][ok], others[ok, j])[0, 1])
                    js.append(np.corrcoef(Mi[ok, j], others[ok, j])[0, 1])
                return np.nanmean(gm) - np.nanmean(js), np.nanmean(gm), np.nanmean(js)

            full = stat(np.arange(m.sum()))
            bs = np.array([stat(rng.integers(0, m.sum(), m.sum()))[0] for _ in range(1000)])
            say(f"  {pop:9s} {s:12s} n {m.sum():3d}: model {full[1]:.3f}, single judge {full[2]:.3f}, difference {full[0]:+.3f} {ci(bs)}")
            h2n[pop][0] += int(np.percentile(bs, 2.5) > 0)
            h2n[pop][1] += 1
    say("H2 count (interval above 0; 'supported overall' needs at least 4 of 6, FREEZE v1.4): " + "; ".join(f"{k} {v[0]} of {v[1]}" for k, v in h2n.items()))
    OUTO.stage("H3")
    h3n = {"PRIMARY": [0, 0, 0], "SECONDARY": [0, 0, 0]}

    say("\n=== H3  size-matched training pools, boosting tier 2 (population gap = own set minus other population; sex gap = own set minus same population other sex)")
    for pop, mask in (("PRIMARY", ~lk_all.twin.values), ("SECONDARY", np.ones(len(lk_all), bool))):
        for s in SETS:
            m = mask & (lk_all.setname == s).values
            if m.sum() < 100:
                continue
            g = s[:2]
            pool = {
                "own": (dev_all.setname == s).values,
                "other_pop": np.array([ETH[x] != ETH[g] for x in dev_all.group]),
                "other_sex": np.array([ETH[x] == ETH[g] and SEX[x] != SEX[g] for x in dev_all.group]),
            }
            n_m = min(v.sum() for v in pool.values())
            pr = {}
            for k, v in pool.items():
                ix = np.where(v)[0]
                acc = np.zeros(m.sum())
                for _ in range(20):
                    sub = rng.choice(ix, n_m, replace=False)
                    mm = S.hgb()
                    mm.fit(Xd[sub], yd[sub])
                    acc += mm.predict(Xl[m]) / 20
                pr[k] = acc
            yy = yl[m]
            idx = [rng.integers(0, m.sum(), m.sum()) for _ in range(1000)]
            gp = np.array([np.corrcoef(pr["own"][i], yy[i])[0, 1] - np.corrcoef(pr["other_pop"][i], yy[i])[0, 1] for i in idx])
            gs = np.array([np.corrcoef(pr["own"][i], yy[i])[0, 1] - np.corrcoef(pr["other_sex"][i], yy[i])[0, 1] for i in idx])
            r = {k: np.corrcoef(v, yy)[0, 1] for k, v in pr.items()}
            h3n[pop][0] += int((r['own'] - r['other_pop']) >= 0.10 and np.percentile(gp, 2.5) > 0)
            h3n[pop][1] += int((r['own'] - r['other_sex']) >= 0.10 and np.percentile(gs, 2.5) > 0)
            h3n[pop][2] += 1
            say(f"  {pop:9s} {s:12s} n {m.sum():3d}, pool {n_m}: own {r['own']:.3f}, other pop {r['other_pop']:.3f}, other sex {r['other_sex']:.3f}; population gap {r['own'] - r['other_pop']:+.3f} {ci(gp)}; sex gap {r['own'] - r['other_sex']:+.3f} {ci(gs)}")

    say("H3 count (gap >= 0.10 with the interval above 0; 'supported overall' needs at least 4 of 6): " + "; ".join(f"{k}: population {v[0]} of {v[2]}, sex {v[1]} of {v[2]}" for k, v in h3n.items()))
    say("H3 note: the 'other population' pool differs in source and sex mix as well as population (FREEZE v1.4 disclosure).")
    OUTO.stage("H5")
    say("\n=== H5  commonality inside each lockbox set (ridge; G geometry, Q capture, A age; 30 subsamples of 80 %; unique shares, mean and 2.5-97.5 percentile)")
    for pop, mask in (("PRIMARY", ~lk_all.twin.values), ("SECONDARY", np.ones(len(lk_all), bool))):
        for s in SETS:
            m = mask & (lk_all.setname == s).values
            if m.sum() < 100:
                continue
            d = lk_all[m].copy().reset_index(drop=True)
            d["rating_mean"] = yl[m]
            for c in VP.QUALITY + ["age_mesh"]:
                lo, hi = dev_all.loc[dev_all.setname == s, c].quantile([0.01, 0.99])
                v = d[c].clip(lo, hi)
                dv = dev_all.loc[dev_all.setname == s, c].clip(lo, hi)
                d[c] = (v.fillna(dv.median()) - dv.mean()) / dv.std()
            Z = {c: Xl[m][:, tier2.index(c)] for c in props}
            for c in props:
                d[c] = Z[c]
            groups = {"G": props, "Q": VP.QUALITY, "A": ["age_mesh"]}
            sh = {k: [] for k in (("G",), ("Q",), ("A",))}
            for _ in range(30):
                sub = d.iloc[rng.choice(len(d), int(0.8 * len(d)), replace=False)].reset_index(drop=True)
                res = VP.partition(sub, groups, S.ridge, repeats=2)
                for k in sh:
                    sh[k].append(res["comm"][k])
            say(f"  {pop:9s} {s:12s} n {m.sum():3d}: " + ", ".join(f"unique {k[0]} {np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]" for k, v in sh.items()))

    say("H5 is descriptive: no verdict (FREEZE v1.4).")
    OUTO.stage("L")
    lookmax(dev_all, lk_all, Xd, Xl, yd, yl, tier2, rng)
    say("L note: the jaw-related tests (C02, C13, C14, C15) describe one dimension (a tapering lower face) and are presented as one group; no count of supported claims is reported (FREEZE v1.4).")

    if not smoke:
        fr_all, _, _ = frames()
        out_sets = fr_all[fr_all.lock & ~fr_all.setname.isin(SETS)].setname.value_counts()
        say("Lockbox faces outside the six sets (no model, no report): " + ", ".join(f"{k} {v}" for k, v in out_sets.items()))
    if smoke:
        say("\nsmoke finished (development faces posing as lockbox; nothing written)")
    else:
        pdf = lk_all[["file", "setname", "twin"]].copy().reset_index(drop=True)
        pdf["y"] = yl
        for k, v in pred.items():
            pdf[f"pred_{k[0]}_{k[1]}"] = v
        pdf.to_csv(PREDS, index=False)
        say("\ndone (the lockbox run S is finished; it is not repeated)")
        OUTO.finish(RESULTS)
        print("[done] results/S_lockbox_run.txt written; predictions in data/lockbox_S_predictions.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"dry": dry, "dev": dev, "ldev": ldev, "run": lambda: run(False, "--rerun" in sys.argv), "smoke": lambda: run(True)}[sys.argv[1]]()
