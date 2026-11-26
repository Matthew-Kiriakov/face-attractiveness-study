"""Post hoc follow-up to editorial v4 using the original inputs and test splits."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def scut_check(scut, rng):
    frame, _, _ = scut.frames()
    _, _, _, columns = scut.O.load()
    new_groups = ["CM internet", "CF internet"]
    groups = scut.SETS + new_groups
    use = frame.loc[frame.setname.isin(groups)]
    train = use.loc[~use.lock]
    test = use.loc[use.lock].copy()
    old = pd.read_csv(scut.PREDS).set_index("file")
    assert test.file.is_unique and old.index.is_unique
    assert set(test.loc[test.setname.isin(scut.SETS), "file"]) == set(old.index)
    train_x = np.empty((len(train), len(columns)))
    test_x = np.empty((len(test), len(columns)))
    train_y = np.empty(len(train))
    test_y = np.empty(len(test))
    for group in groups:
        tr = train.setname.eq(group)
        te = test.setname.eq(group)
        stats = scut.zfit(train.loc[tr], columns)
        train_x[tr], test_x[te] = scut.zapply(stats, train.loc[tr], columns), scut.zapply(stats, test.loc[te], columns)
        ratings = train.loc[tr, "rating_mean"]
        train_y[tr] = (ratings - ratings.mean()) / ratings.std()
        test_y[te] = (test.loc[te, "rating_mean"] - ratings.mean()) / ratings.std()
    test["y"] = test_y
    test["pred8"] = scut.S.hgb().fit(train_x, train_y).predict(test_x)
    test["pred6"] = test.file.map(old.pred_tier2_boosting)
    in_six = test.setname.isin(scut.SETS)
    np.testing.assert_allclose(test.loc[in_six, "y"], test.loc[in_six, "file"].map(old.y), atol=1e-12, rtol=0)
    new_twin = scut.twins(test.file.to_numpy(), train.file.to_numpy()) > .60
    old_twin = test.file.isin(old.index[old.twin])
    assert not (old_twin & ~new_twin).any()
    common = test.loc[~new_twin & ~old_twin]
    rows, old_r, new_r, old_boot, new_boot = [], [], [], [], []
    for group in groups:
        part = common.loc[common.setname.eq(group)]
        y = part.y.to_numpy()
        idx = rng.integers(len(part), size=(5000, len(part)))
        p8 = part.pred8.to_numpy()
        new_r.append(pearson(p8, y))
        new_boot.append(pearson(p8[idx], y[idx]))
        earlier = "—"
        if group in scut.SETS:
            p6 = part.pred6.to_numpy()
            old_r.append(pearson(p6, y))
            old_boot.append(pearson(p6[idx], y[idx]))
            earlier = fmt_interval(old_r[-1], old_boot[-1])
        rows.append(f"| {group} | {train.setname.eq(group).sum()} | {test.setname.eq(group).sum()} | {len(part)} | "
                    f"{earlier} | {fmt_interval(new_r[-1], new_boot[-1])} |")
    mean6 = np.mean(old_r)
    fit8_six = np.mean(new_r[:6])
    fit8_eight = np.mean(new_r)
    b6 = np.mean(old_boot, axis=0)
    b86 = np.mean(new_boot[:6], axis=0)
    b88 = np.mean(new_boot, axis=0)
    return f"""## SCUT | Sensitivity to two excluded groups

Fit one booster with the original 70 features and fixed hyperparameters on all eight development groups. Each group uses its own development statistics. Retain the original split and reapply the original ArcFace threshold to the expanded training sample. Both models are compared on identical photographs in the six original groups. There are {in_six.sum()} original-six test photographs before filtering, {sum(in_six & ~old_twin)} under the original filter, and {sum(common.setname.isin(scut.SETS))} on the matched sample. This analysis uses previously opened test data.

| Group | Training n | Test before filtering | Test after filtering | Original six-group fit r [95%] | Eight-group fit r [95%] |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- Original fit, six groups on the matched sample: **{mean6:.3f}**.
- Expanded fit, same six groups and same photographs: **{fit8_six:.3f}**. Change from the original fit: {fmt_interval(fit8_six - mean6, b86 - b6)}.
- Expanded fit, eight groups: **{fit8_eight:.3f}**. Change in the macro average from reporting eight instead of six with the expanded fit: {fmt_interval(fit8_eight - fit8_six, b88 - b86)}.

Intervals use 5,000 paired bootstrap samples of photographs within groups with fixed fits and group membership. The expanded sample is a sensitivity scenario for the original six-group result.
"""


def mebeauty_check(mb, transfer, rng):
    features = pd.read_csv(transfer.MB / "mebeauty_features.csv",
                           usecols=["image_id", "cv_fold", "ethnicity", "gender"])
    labels = pd.concat([
        pq.read_table(p, columns=["image_id", "beauty_score"]).to_pandas()
        for p in sorted((transfer.MB / "fbp_extended").glob("*.parquet"))
    ], ignore_index=True)
    data = features.merge(labels, on="image_id", validate="one_to_one")
    data["group"] = data.ethnicity + "_" + data.gender
    saved = pd.read_csv(mb.PREDS)
    train = data.loc[data.cv_fold.ne(0)]
    test = data.loc[data.cv_fold.eq(0)].merge(saved[["image_id", "group", "y", "A_boosting", "B_boosting"]],
                                            on=["image_id", "group"], validate="one_to_one")
    assert (len(train), len(test)) == (1968, 492)
    original = data.groupby("group").beauty_score.agg(["mean", "std"])
    original_test = test.join(original, on="group")
    np.testing.assert_allclose((original_test.beauty_score - original_test["mean"]) / original_test["std"],
                               test.y, atol=1e-12, rtol=0)
    reference = train.groupby("group").beauty_score.agg(["mean", "std"])
    test = test.join(reference, on="group")
    rows = []
    for group in transfer.PRIMARY:
        part = test.loc[test.group.eq(group)]
        y = ((part.beauty_score - part["mean"]) / part["std"]).to_numpy()
        p = part.A_boosting.to_numpy()
        idx = rng.integers(len(part), size=(5000, len(part)))
        baseline_error = y ** 2
        gap = (p - y) ** 2 - baseline_error
        np.testing.assert_allclose(pearson(p, y), pearson(p, part.y.to_numpy()), atol=1e-12, rtol=0)
        skill = -gap.mean() / baseline_error.mean()
        rows.append(f"| {group} | {len(part)} | {pearson(p, y):.3f} | {(gap + baseline_error).mean():.3f} | "
                    f"{baseline_error.mean():.3f} | {fmt_interval(gap.mean(), gap[idx].mean(axis=1))} | {skill:.3f} |")
    return f"""## MEBeauty | Error against a constant fixed before test scoring

For every group, use the mean and sample standard deviation of the MEBeauty scores in folds 1–4, restricted to photographs with features as in step B. Score the saved A predictions as z-scores on that reference scale. Equivalently, interpret a prediction in raw beauty-score units as the group mean plus the saved prediction times the group standard deviation. Both the scale and the constant (zero on this z scale) are fixed by folds 1–4. The SCUT model weights are fixed and the interpretation requires group metadata and labelled reference data from MEBeauty. This post hoc calculation reuses part 0.

| Group | n | r | Model MSE | Fixed-mean MSE | Model minus fixed-mean MSE [95%] | MSE skill vs fixed mean |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Positive error differences favour the constant. Intervals use 5,000 paired draws of test photographs within each group with a fixed constant. MSE skill is 1 − model MSE / fixed-mean MSE. The original manuscript R² uses test-group means and the original all-fold scale.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve() / "scripts"))
    import mebeauty_B
    import mebeauty_transfer
    import run_guard
    import scut_lockbox_S
    from evaluation_diagnostics import fmt_interval, pearson

    run_guard.verify()
    rng = np.random.default_rng(20261120)
    scut_report = scut_check(scut_lockbox_S, rng)
    mb_report = mebeauty_check(mebeauty_B, mebeauty_transfer, rng)
    args.report.write_text("# Review follow-up on editorial v4\n\n" + scut_report + "\n" + mb_report, encoding="utf-8")
