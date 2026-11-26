"""Post hoc comparison on the SCUT and MEBeauty test samples in the manuscript.

Reuse the saved test predictions and the original training pipeline. Recreate
the SCUT raw-target fit on development data and fit one new MEBeauty-only
model on folds 1–4. Write aggregate statistics and a diagnostic chart only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BOOT_DRAWS = 5000
BOOT_SEED = 20261119


def pearson(pred, target):
    p = pred - pred.mean(axis=-1, keepdims=True)
    y = target - target.mean(axis=-1, keepdims=True)
    return (p * y).sum(axis=-1) / np.sqrt((p * p).sum(axis=-1) * (y * y).sum(axis=-1))


def score_error(pred, target):
    mse = ((pred - target) ** 2).mean(axis=-1)
    benchmark = target.var(axis=-1)
    return 1 - mse / benchmark, mse - benchmark


def interval(draws):
    return np.quantile(draws, [0.025, 0.975])


def fmt(value):
    return f"{value:.3f}"


def fmt_interval(value, draws):
    lo, hi = interval(draws)
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def scut_data(sl):
    saved = pd.read_csv(sl.PREDS)
    frame, _, tier2, _ = sl.build()
    dev = frame.loc[~frame.lock]
    test = frame.loc[frame.lock].copy()
    assert (len(dev), len(test), len(tier2)) == (3737, 1246, 70)
    assert test.file.is_unique and saved.file.is_unique
    assert test.file.tolist() == saved.file.tolist()
    assert test.setname.tolist() == saved.setname.tolist()
    twins = sl.twins(test.file.to_numpy(), dev.file.to_numpy()) > 0.6
    np.testing.assert_array_equal(twins, saved.twin)

    train_x = np.empty((len(dev), len(tier2)))
    test_x = np.empty((len(test), len(tier2)))
    for group in sl.SETS:
        in_train = dev.setname.eq(group)
        in_test = test.setname.eq(group)
        stats = sl.zfit(dev.loc[in_train], tier2)
        train_x[in_train] = sl.zapply(stats, dev.loc[in_train], tier2)
        test_x[in_test] = sl.zapply(stats, test.loc[in_test], tier2)

    train_raw = dev.rating_mean.to_numpy()
    group_stats = dev.groupby("setname").rating_mean.agg(["mean", "std"])
    test_stats = group_stats.reindex(test.setname)
    group_means = test_stats["mean"].to_numpy()
    group_scales = test_stats["std"].to_numpy()
    raw = test.rating_mean.to_numpy()
    np.testing.assert_allclose((raw - group_means) / group_scales, saved.y, atol=1e-12, rtol=0)

    dev_dummies = pd.get_dummies(dev.setname).reindex(columns=sl.SETS).to_numpy(dtype=float)
    test_dummies = pd.get_dummies(test.setname).reindex(columns=sl.SETS).to_numpy(dtype=float)
    raw_model = sl.S.hgb().fit(np.column_stack([train_x, dev_dummies]), train_raw)
    saved["pred_raw"] = raw_model.predict(np.column_stack([test_x, test_dummies]))
    saved["pred_group_mean"] = group_means
    saved["pred_back_transformed"] = saved.pred_tier2_boosting * group_scales + group_means
    saved["y_raw"] = raw
    primary = saved.loc[~saved.twin].reset_index(drop=True)
    assert len(primary) == 1196
    assert round(pearson(primary.pred_raw.to_numpy(), primary.y_raw.to_numpy()), 3) == 0.823
    return primary, sl.SETS


def scut_report(data, groups, rng):
    indices = [np.flatnonzero(data.setname.eq(group)) for group in groups]
    samples = [idx[rng.integers(len(idx), size=(BOOT_DRAWS, len(idx)))] for idx in indices]
    pooled_sample = np.concatenate(samples, axis=1)
    lines = [
        "## Table 1. SCUT | Same 1,196 test photographs",
        "",
        "| Model and target scale | Pooled r | Macro r | Pooled minus macro r [95% interval] | Pooled R² |",
        "|---|---:|---:|---:|---:|",
    ]
    detail = {}
    for label, pred_col, target_col in (
        ("70 features, within-group z scale", "pred_tier2_boosting", "y"),
        ("70 features, back-transformed to 1–5", "pred_back_transformed", "y_raw"),
        ("70 features + group indicators, 1–5", "pred_raw", "y_raw"),
        ("Development group mean only, 1–5", "pred_group_mean", "y_raw"),
    ):
        pred, target = data[pred_col].to_numpy(), data[target_col].to_numpy()
        pooled = pearson(pred, target)
        r2, _ = score_error(pred, target)
        if pred_col == "pred_group_mean":
            macro, gap = "Undefined", "—"
        else:
            group_r = [pearson(pred[idx], target[idx]) for idx in indices]
            mean_r = np.mean(group_r)
            boot_macro = np.mean([pearson(pred[idx], target[idx]) for idx in samples], axis=0)
            boot_pooled = pearson(pred[pooled_sample], target[pooled_sample])
            macro = fmt(mean_r)
            gap = fmt_interval(pooled - mean_r, boot_pooled - boot_macro)
            detail[pred_col] = group_r
        lines.append(f"| {label} | {pooled:.3f} | {macro} | {gap} | {r2:.3f} |")

    np.testing.assert_allclose(detail["pred_tier2_boosting"], detail["pred_back_transformed"], atol=1e-12)
    assert round(np.mean(detail["pred_tier2_boosting"]), 3) == 0.678
    assert round(pearson(data.pred_tier2_boosting.to_numpy(), data.y.to_numpy()), 3) == 0.672
    lines.extend([
        "",
        "| Group | n | r, 70 features (same inside group on either scale) | r, 70 features + group indicators |",
        "|---|---:|---:|---:|",
    ])
    for j, group in enumerate(groups):
        lines.append(f"| {group} | {len(indices[j])} | {detail['pred_tier2_boosting'][j]:.3f} | "
                     f"{detail['pred_raw'][j]:.3f} |")

    pred = data.pred_raw.to_numpy()
    target = data.y_raw.to_numpy()
    centered = data[["pred_raw", "y_raw"]] - data.groupby("setname")[["pred_raw", "y_raw"]].transform("mean")
    total_cov = ((pred - pred.mean()) * (target - target.mean())).mean()
    within_cov = (centered.pred_raw * centered.y_raw).mean()
    between_share = (total_cov - within_cov) / total_cov
    centered_r = pearson(centered.pred_raw.to_numpy(), centered.y_raw.to_numpy())
    lines.extend([
        "",
        f"For the raw-target model, pooled r after centering both columns within each test group is {centered_r:.3f}. "
        f"Between-group means account for {between_share:.1%} of its pooled covariance numerator "
        f"(total {total_cov:.5f}, within {within_cov:.5f}). The variance terms in Pearson's denominator also change.",
        "",
        "Macro r gives equal weight to each group. A constant prediction within each group makes its r undefined. "
        "The back-transformation applies the development mean and standard deviation of each group to the saved "
        "predictions and requires no refit. Pooled R² uses the mean of the 1,196 test targets on that row's scale; "
        "compare R² values on the same scale. Intervals resample photographs within groups and reuse the same draws "
        "for the pooled and macro metrics.",
    ])
    return lines


def mebeauty_data(transfer, training, model):
    saved = pd.read_csv(model.PREDS)
    _, _, _, tier2 = training.load()
    features = transfer.mb_features(tier2)
    labels = pd.concat([
        pq.read_table(part, columns=["image_id", "beauty_score"], filters=[("cv_fold", "!=", 0)]).to_pandas()
        for part in sorted((transfer.MB / "fbp_extended").glob("*.parquet"))
    ], ignore_index=True)
    train = features.loc[features.cv_fold.ne(0)].merge(labels, on="image_id", validate="one_to_one")
    target = train.groupby("group").beauty_score.transform(lambda v: (v - v.mean()) / v.std())
    test = features.loc[features.cv_fold.eq(0)]
    assert (len(train), len(test), len(saved)) == (1968, 492, 492)
    assert set(train.cv_fold.unique()) == {1, 2, 3, 4}
    assert train.image_id.is_unique and test.image_id.is_unique and saved.image_id.is_unique
    assert set(train.image_id).isdisjoint(test.image_id)
    assert set(test.image_id) == set(saved.image_id)
    boost = training.S.hgb().fit(train[tier2].to_numpy(dtype=float), target.to_numpy())
    new_predictions = pd.Series(boost.predict(test[tier2].to_numpy(dtype=float)), index=test.image_id)
    assert saved.group.tolist() == test.set_index("image_id").group.reindex(saved.image_id).tolist()
    saved["target_only"] = saved.image_id.map(new_predictions)
    return saved, transfer.PRIMARY


def mebeauty_report(data, groups, rng):
    columns = ["A_boosting", "B_boosting", "target_only"]
    group_results = []
    lines = [
        "## Table 2. MEBeauty | Same fold 0 test photographs",
        "",
        "| Group | n | r A [95% interval] | R² A [95% interval] | ΔMSE A [95% interval] | r B | R² B | r C | R² C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        part = data.loc[data.group.eq(group)]
        y = part.y.to_numpy()
        sample = rng.integers(len(part), size=(BOOT_DRAWS, len(part)))
        results = {}
        for col in columns:
            pred = part[col].to_numpy()
            r2, gap = score_error(pred, y)
            boot_r2, boot_gap = score_error(pred[sample], y[sample])
            results[col] = dict(
                r=pearson(pred, y), r2=r2, gap=gap, boot_r=pearson(pred[sample], y[sample]),
                boot_r2=boot_r2, boot_gap=boot_gap,
            )
        a, b, c = [results[col] for col in columns]
        group_results.append((group, len(part), results))
        lines.append(f"| {group} | {len(part)} | {fmt_interval(a['r'], a['boot_r'])} | "
                     f"{fmt_interval(a['r2'], a['boot_r2'])} | "
                     f"{fmt_interval(a['gap'], a['boot_gap'])} | {b['r']:.3f} | {b['r2']:.3f} | "
                     f"{c['r']:.3f} | {c['r2']:.3f} |")

    aggregate = {
        col: {key: np.mean([res[col][key] for _, _, res in group_results], axis=0) for key in ("r", "r2", "boot_r", "boot_r2")}
        for col in columns
    }
    a, b, c = [aggregate[col] for col in columns]
    assert round(a["r"], 3) == 0.337 and round(b["r"], 3) == 0.431
    np.testing.assert_array_equal(
        np.round([res["A_boosting"]["r2"] for _, _, res in group_results], 3),
        [-0.018, -0.158, -0.047, 0.436],
    )
    primary = data.loc[data.group.isin(groups)]
    assert len(primary) == 261
    centered_a = primary.A_boosting - primary.groupby("group").A_boosting.transform("mean")
    assert round(pearson(centered_a.to_numpy(), primary.y.to_numpy()), 3) == 0.263
    lines.append(
        f"| Equal-group mean | {len(primary)} | {fmt_interval(a['r'], a['boot_r'])} | "
        f"{fmt_interval(a['r2'], a['boot_r2'])} | — | "
        f"{b['r']:.3f} | {b['r2']:.3f} | {c['r']:.3f} | {c['r2']:.3f} |"
    )
    lines.extend([
        "",
        "A = SCUT only. B = SCUT + MEBeauty folds 1–4. C = MEBeauty folds 1–4 only. "
        "A and B use saved boosting predictions. C uses the same 70 features, estimator settings, "
        "and MEBeauty training pre-processing as B. The score is evaluated for all three against the saved "
        "group-standardized test target (scale fitted on all MEBeauty folds in step B).",
        "",
        "ΔMSE = model MSE minus the MSE of a constant fitted to the *test group mean*. Positive is worse. "
        "The constant is an evaluation reference and uses the observed test outcomes. "
        "It is refitted within each paired bootstrap sample. This comparison quantifies error on the reported scale "
        "with the models fixed. A prospective comparison would fix the baseline using training data.",
        "",
        "| Paired contrast | Δ macro r [95% interval] | Δ mean group R² [95% interval] |",
        "|---|---:|---:|",
    ])
    for label, left, right in (("B − A", b, a), ("C − A", c, a), ("B − C", b, c)):
        r = fmt_interval(left["r"] - right["r"], left["boot_r"] - right["boot_r"])
        r2 = fmt_interval(left["r2"] - right["r2"], left["boot_r2"] - right["boot_r2"])
        lines.append(f"| {label} | {r} | {r2} |")
    lines.extend([
        "",
        "Intervals use paired resampling within each group. Small groups retain the same weight as large groups. "
        "The intervals describe variation across test photographs for the fixed models. They are descriptive and "
        "unadjusted for multiple comparisons.",
    ])
    return lines, group_results


def draw_mebeauty(groups, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{group.replace('_', ' ').capitalize()}  (n = {count})" for group, count, _ in groups]
    positions = np.arange(len(groups))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.75, bottom=0.24, wspace=0.30)
    fig.text(0.24, 0.92, "SCUT → MEBeauty: agreement and error on the same photographs",
             fontsize=15, weight="bold", color="#142535")
    fig.text(0.24, 0.85, "Model A · Fold 0 · Four primary groups", fontsize=10, color="#415564")
    for axis, key, draws, title, color in (
        (left, "r", "boot_r", "Pearson r", "#176785"),
        (right, "r2", "boot_r2", "Predictive R²", "#974826"),
    ):
        values = np.array([res["A_boosting"][key] for _, _, res in groups])
        intervals = np.array([interval(res["A_boosting"][draws]) for _, _, res in groups])
        axis.hlines(positions, intervals[:, 0], intervals[:, 1], color=color, linewidth=2)
        axis.scatter(values, positions, color=color, s=42, zorder=3)
        axis.axvline(0, color="#758594", linestyle="--", linewidth=1)
        axis.grid(axis="x", color="#e8edf0")
        axis.set_axisbelow(True)
        axis.set_title(title, loc="left", fontsize=11, weight="bold", pad=11)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#c0cbd2")
        axis.tick_params(axis="both", length=0, labelsize=9, pad=6, colors="#263b49")
    left.set_yticks(positions, labels)
    left.invert_yaxis()
    left.set_xlabel("Higher = stronger positive agreement", fontsize=9)
    right.set_xlabel("Negative = more error than the test-group mean", fontsize=9)
    fig.text(0.24, 0.07, "Dots are estimates. Lines are 95% intervals from resampling photographs within groups.",
             fontsize=9, color="#415564")
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.project.resolve() / "scripts"))
    import mebeauty_B as mb
    import mebeauty_transfer as transfer
    import overnight_common as training
    import run_guard
    import scut_lockbox_S as scut

    run_guard.verify()
    rng = np.random.default_rng(BOOT_SEED)
    scut_primary, scut_groups = scut_data(scut)
    mb_test, mb_groups = mebeauty_data(transfer, training, mb)
    mb_lines, plot_data = mebeauty_report(mb_test, mb_groups, rng)
    lines = [
        "# Evaluation diagnostics | SCUT and MEBeauty",
        "",
        "2026-11-19. Post hoc analyses on test samples already used in the manuscript. "
        "The original S/B predictions, reports, and protocol are source inputs. "
        f"Bootstrap uses {BOOT_DRAWS:,} draws and seed {BOOT_SEED}. The original input hashes and library "
        "versions were verified before reconstructing the raw SCUT fit and fitting the new MEBeauty-only model.",
        "",
        *scut_report(scut_primary, scut_groups, rng),
        "",
        *mb_lines,
        "",
        "The inputs are `data/lockbox_S_predictions.csv`, `data/mebeauty_B_predictions.csv`, "
        "`scripts/scut_lockbox_S.py`, `scripts/posthoc_ladder_lockbox.py`, and `scripts/mebeauty_B.py`. "
        "Existing headline point estimates were reproduced from the saved predictions or the reconstructed fit. "
        "New intervals use the bootstrap specified above.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.figure:
        draw_mebeauty(plot_data, args.figure)


if __name__ == "__main__":
    main()
