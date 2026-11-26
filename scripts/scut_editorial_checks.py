"""Re-express saved SCUT predictions and cross-fit development group means."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def pooled(pred, target):
    return float(np.corrcoef(pred, target)[0, 1])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--figure", type=Path)
    args = cli.parse_args()
    sys.path.insert(0, str(args.project.resolve() / "scripts"))
    import run_guard
    import scut_lockbox_S as scut

    run_guard.verify()
    frame, _, _ = scut.frames()
    dev = frame.loc[~frame.lock].copy()
    assert dev.file.is_unique
    stats = dev[dev.setname.isin(scut.SETS)].groupby("setname").rating_mean.agg(["mean", "std"])
    saved = pd.read_csv(scut.PREDS).query("not twin").copy()
    assert len(saved) == 1196 and saved.file.is_unique
    data = saved.join(stats, on="setname", validate="many_to_one")
    original = frame.set_index("file").rating_mean.reindex(data.file).to_numpy()
    np.testing.assert_allclose(data.y * data["std"] + data["mean"], original, atol=1e-12, rtol=0)
    z = data[["pred_tier2_boosting", "y"]]
    centered_z = z - z.groupby(data.setname).transform("mean")
    centered_rescaled = centered_z.mul(data["std"], axis=0)
    raw = z.mul(data["std"], axis=0).add(data["mean"], axis=0)
    trace = [pooled(*pair.to_numpy().T) for pair in (z, centered_z, centered_rescaled, raw)]
    group_r = [pooled(*group.to_numpy().T) for _, group in z.groupby(data.setname)]
    macro = float(np.mean(group_r))
    baseline = pooled(data["mean"], original)
    assert [round(x, 3) for x in (trace[0], trace[-1], macro, baseline)] == [.672, .824, .678, .608]

    dev = dev.reset_index(drop=True)
    target = dev.rating_mean.to_numpy()
    assert np.isfinite(target).all()
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20261120).split(dev, dev.group))
    cv = {}
    for label in ("group", "setname"):
        predictions = np.full(len(dev), np.nan)
        fallback = 0
        for fit, test in splits:
            train, held_out = dev.iloc[fit], dev.iloc[test]
            means = train.groupby(label).rating_mean.mean()
            predicted = held_out[label].map(means)
            fallback += int(predicted.isna().sum())
            predictions[test] = predicted.fillna(held_out.group.map(train.groupby("group").rating_mean.mean()))
        assert np.isfinite(predictions).all()
        cv[label] = {
            "r": pooled(predictions, target),
            "r2": float(1 - np.sum((predictions - target) ** 2) / np.sum((target - target.mean()) ** 2)),
            "fallback_to_sheet_mean": fallback,
        }
    result = {
        "scut": {
            "n": len(saved), "pooled_z": trace[0], "pooled_centered_z": trace[1],
            "pooled_centered_rescaled": trace[2], "pooled_1_to_5": trace[3],
            "macro_r": macro, "group_mean_pooled_r": baseline,
        },
        "development": {"n": len(dev), "set_counts": dev.setname.value_counts().to_dict(), "five_fold": cv},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.figure:
        fig, ax = plt.subplots(figsize=(9.6, 3.3))
        fig.subplots_adjust(left=.34, right=.95, bottom=.23, top=.93)
        ax.set_yticks([2, 1, 0], [
            "Одна модель\nНормированная шкала",
            "Та же модель\nИсходная шкала 1–5",
            "Среднее группы\nИз рабочей выборки",
        ])
        for y, point in zip([2, 1, 0], [trace[0], trace[3], baseline]):
            ax.scatter(point, y + .13, color="#116b8b", s=65, zorder=3)
            ax.text(point + .026, y + .13, f"{point:.3f}".replace(".", ","), va="center", fontsize=10)
        for y in (2, 1):
            ax.scatter(macro, y - .15, color="#9a4a27", marker="D", s=54, zorder=3)
            ax.text(macro + .026, y - .15, f"{macro:.3f}".replace(".", ","), va="center", fontsize=10)
        ax.text(.03, -.17, "Внутри группы прогноз постоянен", va="center", color="#485963", fontsize=9)
        ax.set(xlim=(0, 1), ylim=(-.42, 2.45), xlabel="Корреляция Пирсона")
        ticks = np.linspace(0, 1, 6)
        ax.set_xticks(ticks, [f"{v:.1f}".replace(".", ",") for v in ticks])
        ax.grid(axis="x", color="#dde4e8")
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.scatter([], [], color="#116b8b", s=55, label="Общая")
        ax.scatter([], [], color="#9a4a27", marker="D", s=47, label="Средняя по группам")
        ax.legend(loc="lower center", bbox_to_anchor=(.5, -.47), ncol=2, frameon=False, fontsize=9)
        fig.savefig(args.figure, dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
