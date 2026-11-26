"""Fills the computed keys of page/data.json from DEVELOPMENT faces only and mirrors the file to data.js (for opening the page from disk).
The lockbox is never analysed: overnight_common.load() drops its faces before anything is computed, and MEBeauty is not read at all.
Hand-written keys of data.json (labels, quotes, lockbox, mebeauty, not_claimed, lookmaxxing) are kept as they are.
Usage: python page/build_data.py          recompute
       python page/build_data.py --remove  recompute only the `remove` ladder
       python page/build_data.py --sync   only copy data.json -> data.js after a hand edit
Everything computed here is exploratory and descriptive; shapes of the keys are in DATA_SCHEMA.md."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PAGE.parent / "scripts"))
COMPUTED = ["meta", "sets", "remove", "clusters", "commonality", "ceiling_scut", "face"]
SETS = ["CM 10k_US", "CF 10k_US", "AM DataTang", "AM internet", "AF DataTang", "AF internet"]
GRID = np.linspace(-2, 2, 9)
RNG = np.random.default_rng(20261110)

# MediaPipe indices drawn by the page; the first block is every point js/metrics.js measures.
MODEL_PTS = {"forehead": 10, "menton": 152, "nasion": 168, "pronasale": 1, "subnasale": 2, "bridgeUpper": 6, "bridgeMid": 197, "bridgeLower": 195,
             "alarL": 98, "alarR": 327, "nostrilTopL": 45, "nostrilLowL": 48, "nostrilTopR": 275, "nostrilLowR": 278,
             "eyeOuterL": 33, "eyeInnerL": 133, "eyeTopL": 159, "eyeBotL": 145, "irisL": 468, "eyeOuterR": 263, "eyeInnerR": 362, "eyeTopR": 386, "eyeBotR": 374, "irisR": 473,
             "browOuterL": 46, "browInnerL": 107, "browOuterR": 276, "browInnerR": 336, "mouthL": 61, "mouthR": 291, "cupid": 0, "cupidL": 37, "cupidR": 267,
             "lipUpper": 13, "lipLower": 14, "lipBot": 17, "zygionL": 234, "zygionR": 454, "gonionL": 172, "gonionR": 397, "templeL": 54, "templeR": 284}
PATHS = {"oval": [10, 297, 284, 389, 454, 361, 397, 379, 400, 152, 176, 150, 172, 132, 234, 162, 54, 67],
         "browL": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46], "browR": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
         "eyeL": [33, 160, 159, 158, 133, 153, 145, 144], "eyeR": [263, 387, 386, 385, 362, 380, 374, 373],
         "bridge": [168, 6, 197, 195, 5, 4], "noseBase": [48, 64, 98, 2, 327, 294, 278],
         "lipOuter": [61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181], "lipInner": [78, 82, 13, 312, 308, 317, 14, 87]}
IDX = sorted(set(MODEL_PTS.values()) | {i for p in PATHS.values() for i in p})


def rnd(a, k=3):
    return np.round(np.asarray(a, float), k).tolist()


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def resid(v, C):
    C1 = np.column_stack([np.ones(len(v)), C])
    return v - C1 @ np.linalg.lstsq(C1, v, rcond=None)[0]


def boot(n, reps=1000):
    return RNG.integers(0, n, (reps, n))


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def set_views(df, y, oof):
    """Per set: r with interval, Spearman, a 2D count grid and eight prediction bins, for each model."""
    out = {}
    edges_x, edges_y = np.linspace(-2.4, 2.4, 17), np.linspace(-3, 3, 17)
    for s in SETS:
        m = (df.setname == s).values
        ys, entry = y[m], {"n": int(m.sum())}
        B = boot(len(ys))
        for fam, P in oof.items():
            Ps, p = P[:, m], P[:, m].mean(axis=0)
            r = float(np.mean([corr(q, ys) for q in Ps]))
            rb = [np.mean([corr(q[i], ys[i]) for q in Ps[:3]]) for i in B]
            q = np.quantile(p, np.linspace(0, 1, 9))
            b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 7)
            bins = [[p[b == k].mean(), ys[b == k].mean(), 1.96 * ys[b == k].std(ddof=1) / np.sqrt((b == k).sum()),
                     np.percentile(ys[b == k], 10), np.percentile(ys[b == k], 90), int((b == k).sum())] for k in range(8)]
            H = np.histogram2d(np.clip(p, -2.39, 2.39), np.clip(ys, -2.99, 2.99), [edges_x, edges_y])[0].astype(int)
            entry[fam] = {"r": round(r, 3), "ci": rnd(ci(rb)), "rho": round(float(spearmanr(p, ys)[0]), 3), "grid": H.T.tolist(), "bins": rnd(bins),
                          "pred_quantiles": rnd(np.quantile(p, np.linspace(0, 1, 21)))}
        out[s] = entry
    return out, {"x": [-2.4, 2.4], "y": [-3, 3], "cells": 16}


def remove_levers(df, y, P, raw, Praw, cap, age):
    """r that remains for each combination of removed levers; every r is the mean over seeds of the out-of-fold r (P, Praw: seeds x faces).
    Source removed = judged inside a set (z target); kept = raw 1-5 ratings, six sets mixed, model told the set. Age / capture removed =
    partial correlation given perceived age / six capture statistics (linear, inside the set when source is removed). Intervals: face bootstrap, 3 seeds."""
    out = {}
    setm = [(df.setname == s).values for s in SETS]
    for key in ["", "age", "capture", "age,capture"]:
        cols = ([age] if "age" in key else []) + (cap if "capture" in key else [])
        C = df[cols].values if cols else None
        f = (lambda v, i: resid(v[i], C[i])) if cols else (lambda v, i: v[i])
        alli = np.arange(len(y))
        B = boot(len(y), 400)
        r = lambda t, Q, i: float(np.mean([corr(f(t, i), f(q, i)) for q in Q]))
        out[key or "none"] = {"pooled": round(r(raw, Praw, alli), 3), "ci": rnd(ci([r(raw, Praw[:3], i) for i in B]))}
        per, bs = {}, np.zeros((400, len(SETS)))
        for k, m in enumerate(setm):
            ii = np.where(m)[0]
            per[SETS[k]] = round(r(y, P, ii), 3)
            bs[:, k] = [r(y, P[:3], ii[j]) for j in boot(len(ii), 400)]
        out[("source," + key).rstrip(",")] = {"pooled": round(float(np.mean(list(per.values()))), 3), "ci": rnd(ci(bs.mean(axis=1))), "sets": per}
    return out


def importance(df, y, tier2, defs, S):
    """Held-out R2 lost when a cluster (or one feature) is shuffled inside the test fold; boosting, 5 folds x 2 repeats x 4 shuffles."""
    from sklearn.model_selection import RepeatedKFold
    X = df[tier2].values
    col = {c: j for j, c in enumerate(tier2)}
    groups = {d["id"]: [col[c] for c in d["features"]] for d in defs}
    drop = {k: [] for k in list(groups) + tier2}
    r2 = lambda t, q: 1 - np.sum((t - q) ** 2) / np.sum((t - t.mean()) ** 2)
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=2, random_state=S.SEED).split(X):
        mdl = S.hgb().fit(X[tr], y[tr])
        base = r2(y[te], mdl.predict(X[te]))
        for k, cols in list(groups.items()) + [(c, [col[c]]) for c in tier2]:
            d = []
            for _ in range(4):
                Xp = X[te].copy()
                Xp[:, cols] = Xp[RNG.permutation(len(te))][:, cols]
                d.append(base - r2(y[te], mdl.predict(Xp)))
            drop[k].append(np.mean(d))
    drop = {k: float(np.mean(v)) for k, v in drop.items()}
    total = sum(max(drop[d["id"]], 0) for d in defs)
    rho = {c: float(np.mean([spearmanr(df.loc[df.setname == s, c], y[(df.setname == s).values])[0] for s in SETS])) for c in tier2}
    out = []
    for d in defs:
        inner = sum(max(drop[c], 0) for c in d["features"]) or 1
        out.append({"id": d["id"], "drop": round(drop[d["id"]], 4), "share": round(max(drop[d["id"]], 0) / total, 3),
                    "features": [{"id": c, "drop": round(drop[c], 4), "within": round(max(drop[c], 0) / inner, 3), "rho": round(rho[c], 3)} for c in d["features"]]})
    return out


def judges(df, y, p):
    """One judge against the mean of the other 59, and the reliability ceiling of the 60-judge mean, inside each set."""
    a = pd.read_excel(PAGE.parent / "data/scut-fbp5500/SCUT-FBP5500_v2/All_Ratings.xlsx", sheet_name="ALL")
    W = a[a.Filename.isin(df.file)].groupby(["Filename", "Rater"]).Rating.mean().unstack("Rater")
    out = {}
    for s in SETS:
        m = (df.setname == s).values
        w = W.reindex(df.file[m]).values
        rj = [corr(w[:, j], np.delete(w, j, axis=1).mean(axis=1)) for j in range(w.shape[1])]
        rel = 1 - (w.var(axis=1, ddof=1).mean() / w.shape[1]) / w.mean(axis=1).var(ddof=1)
        out[s] = {"one_judge": round(float(np.mean(rj)), 3), "one_judge_range": rnd([np.percentile(rj, 10), np.percentile(rj, 90)]),
                  "ceiling": round(float(np.sqrt(rel)), 3)}
    return out


def procrustes(Xs):
    """Generalised Procrustes (similarity, no reflection); returns aligned shapes."""
    Xs = Xs - Xs.mean(axis=1, keepdims=True)
    Xs = Xs / np.linalg.norm(Xs, axis=(1, 2), keepdims=True)
    ref = Xs[0]
    for _ in range(4):
        for i, x in enumerate(Xs):
            u, _, vt = np.linalg.svd(x.T @ ref)
            if np.linalg.det(u @ vt) < 0:
                u[:, -1] *= -1
            Xs[i] = x @ (u @ vt)
        ref = Xs.mean(axis=0)
        ref /= np.linalg.norm(ref)
    return Xs


def face(df, raw, props, tier2, y, S):
    """Mean shape per set and how it moves per 1 z of each proportion (ridge regression of aligned landmarks on the 42 z-scores, alpha 1),
    plus the pooled models in browser form: ridge slopes and boosting partial-dependence curves on GRID."""
    lm = {}
    for f in ["scut_cm_landmarks", "scut_am_landmarks_web", "scut_cf_landmarks_web", "scut_af_landmarks_web"]:
        for line in open(PAGE.parent / f"data/{f}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["file"] in set_files and r.get("trichion_found") and r.get("lm") and len(r["lm"]) >= 478:
                lm[r["file"]] = np.array([r["lm"][i] for i in IDX] + [r["trichion"], r["glabella"]])
    X = df[tier2].values
    rm = S.ridge().fit(X, y)
    coef = rm.named_steps["model"].coef_ / rm.named_steps["scale"].scale_
    zero = np.zeros((1, len(tier2)))
    hm = S.hgb().fit(X, y)
    pd_curves = {}
    for j, c in enumerate(props):
        Xg = X.copy()
        vals = []
        for g in GRID:
            Xg[:, j] = g
            vals.append(hm.predict(Xg).mean())
        pd_curves[c] = rnd(np.array(vals) - vals[4], 4)
    sets, Ys = {}, {}
    for s in SETS:
        d = df[(df.setname == s) & df.file.isin(lm)]
        A = procrustes(np.stack([lm[f] for f in d.file]))
        mean = A.mean(axis=0)
        t, mn = mean[-2], mean[IDX.index(152)]
        ax = (mn - t) / np.linalg.norm(mn - t)
        R = np.array([[ax[1], ax[0]], [-ax[0], ax[1]]])  # hairline-to-chin axis points down, its length = 1
        Ys[s] = ((A - (t + mn) / 2) @ R / np.linalg.norm(mn - t)).reshape(len(A), -1)
        rs = raw[raw.file.isin(df.file[df.setname == s])]
        stats = {}
        for c in props:
            v = rs[c].clip(*rs[c].quantile([0.01, 0.99]))
            stats[c] = [round(float(v.mean()), 4), round(float(v.std()), 4)]
        sets[s] = {"n": len(d), "mean": rnd(Ys[s].mean(axis=0).reshape(-1, 2), 4), "stats": stats}
    # Morph: for +1 z of one proportion and 0 for the other 41, the landmark move that is smallest in the metric of the set's own landmark covariance
    # (d = C J' (J C J' + 0.01 I)^-1 e). J comes from the metric code itself. Proportions share landmarks (the thirds sum to 100 %, mouth/nose = mouth/face : nose/face),
    # so not every request is geometrically possible; "response" says what the drawn face really measures, and the page feeds that to the models.
    tmp = PAGE / "_jac_in.json"
    tmp.write_text(json.dumps({"index": IDX + ["trichion", "glabella"], "model": MODEL_PTS, "props": props, "sets": {s: sets[s]["mean"] for s in SETS}}))
    subprocess.run([str(S.NODE), str(PAGE / "jacobian.mjs"), str(tmp), str(PAGE / "_jac_out.json")], check=True)
    jac = json.loads((PAGE / "_jac_out.json").read_text())
    tmp.unlink(), (PAGE / "_jac_out.json").unlink()
    for s in SETS:
        sd = np.array([sets[s]["stats"][c][1] for c in props])
        J = np.array(jac["J"][s]) / sd[:, None]
        C = np.cov(Ys[s].T)
        G = J @ C @ J.T
        Bm = np.linalg.solve(G + 0.01 * np.eye(len(props)), J @ C)
        resp = J @ Bm.T
        sets[s]["morph"] = {c: rnd(Bm[j].reshape(-1, 2), 4) for j, c in enumerate(props)}
        sets[s]["response"] = rnd(resp)  # [k][j]: z of proportion k that the drawn face measures per +1 z asked of proportion j
    return {"points": {"index": IDX + ["trichion", "glabella"], "model": MODEL_PTS, "paths": PATHS}, "props": props, "grid": GRID.tolist(), "range": 2,
            "ridge": {"intercept": round(float(rm.predict(zero)[0]), 4), "coef": {c: round(float(coef[j]), 4) for j, c in enumerate(props)}},
            "boosting": {"base": round(float(hm.predict(zero)[0]), 4), "pd": pd_curves}, "sets": sets}


def write(data):
    body = ",\n".join(f'"{k}": ' + (json.dumps(v, ensure_ascii=False, separators=(",", ":")) if k in COMPUTED else json.dumps(v, ensure_ascii=False, indent=1))
                      for k, v in data.items())
    (PAGE / "data.json").write_text("{\n" + body + "\n}\n", encoding="utf-8")
    sync()


def sync():
    data = json.loads((PAGE / "data.json").read_text(encoding="utf-8"))
    (PAGE / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")


def main(only_remove=False):
    global set_files
    import exploration_v1 as E
    import overnight_common as O
    import scut_cm_harmony as S
    import variance_partition as VP
    data = json.loads((PAGE / "data.json").read_text(encoding="utf-8"))
    df, props, rest, tier2 = O.load()
    df = df[df.setname.isin(SETS)].reset_index(drop=True)
    set_files = set(df.file)
    y = df.y.values
    sets_raw, _ = E.load()
    raw = pd.concat([d[["file", "rating_mean"] + props] for d in sets_raw.values()])
    rating = df[["file"]].merge(raw[["file", "rating_mean"]], on="file").rating_mean.values
    oof = {fam: O.oof_seeds(df, tier2, y, fam, "page") for fam in ("boosting", "ridge")}
    onehot = pd.get_dummies(df.setname).values.astype(float)
    Xr = np.column_stack([df[tier2].values, onehot])
    p_raw = np.array([S.oof_predict(S.hgb, Xr, rating, 2, s) for s in O.SEEDS[:3]])

    if only_remove:
        data["remove"] = remove_levers(df, y, oof["boosting"], rating, p_raw, VP.QUALITY, "age_mesh")
        write({k: data[k] for k in COMPUTED + [k for k in data if k not in COMPUTED]})
        print({k: v["pooled"] for k, v in data["remove"].items()})
        return
    data["sets"], axes = set_views(df, y, oof)
    data["meta"] = {"built": str(date.today()), "sample": "development faces only (lockbox excluded by the loader)", "n": len(df), "density_axes": axes,
                    "model": "tier 2: 42 proportions + 28 image features; out-of-fold, 5 folds x 2 repeats x 10 seeds",
                    "mean_r": {fam: round(float(np.mean([data["sets"][s][fam]["r"] for s in SETS])), 3) for fam in oof}}
    data["remove"] = remove_levers(df, y, oof["boosting"], rating, p_raw, VP.QUALITY, "age_mesh")
    data["clusters"] = importance(df, y, tier2, data["cluster_defs"], S)
    vp = json.loads((PAGE.parent / "data/cache_expl/vp.json").read_text())
    data["commonality"] = {k.replace("|", " "): {p: rnd(v[:3]) for p, v in e["stats"].items()} for k, e in vp.items() if k.replace("|", " ") in SETS}
    data["ceiling_scut"] = judges(df, y, oof["boosting"].mean(axis=0))
    data["face"] = face(df, raw, props, tier2, y, S)
    write({k: data[k] for k in COMPUTED + [k for k in data if k not in COMPUTED]})
    print("mean r", data["meta"]["mean_r"])


if __name__ == "__main__":
    sync() if "--sync" in sys.argv else main("--remove" in sys.argv)
