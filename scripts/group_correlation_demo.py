"""Compute pooled and group-wise Pearson r and a group-mean reference.

Run with no arguments for a deterministic synthetic example. To use your data,
pass a CSV with columns split, group, target, prediction. The split values are
train and test. All targets and predictions must use the same rating scale.
Prediction is required only on test rows. Group means are estimated on train.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import StatisticsError, correlation, mean


def pearson(rows):
    actual, predicted = zip(*rows)
    try:
        return correlation(actual, predicted)
    except StatisticsError:
        return None


def summarize(rows):
    train = defaultdict(list)
    test = defaultdict(list)
    for row in rows:
        group = row["group"].strip()
        split = row["split"].strip()
        actual = float(row["target"])
        if not group or not math.isfinite(actual):
            raise ValueError("Each row needs a group and a finite target.")
        if split == "train":
            train[group].append(actual)
        elif split == "test":
            pred = float(row["prediction"])
            if not math.isfinite(pred):
                raise ValueError("Test predictions must be finite.")
            test[group].append((actual, pred))
        else:
            raise ValueError(f"Unknown split {split!r}. Use train or test.")
    if not test:
        raise ValueError("Provide at least one test row.")
    if test.keys() - train.keys():
        raise ValueError("Every test group must have a training group mean.")
    grouped = [(g, len(pairs), pearson(pairs)) for g, pairs in sorted(test.items())]
    pooled = pearson([pair for pairs in test.values() for pair in pairs])
    group_rs = [r for _, _, r in grouped if r is not None]
    macro = mean(group_rs) if group_rs else None
    training_means = {g: mean(train[g]) for g in test}
    reference = pearson([(y, training_means[g]) for g, pairs in test.items() for y, _ in pairs])
    return pooled, macro, reference, grouped


def show(value):
    return "undefined" if value is None else f"{value:.3f}"


def example(separation):
    # Zero covariance inside each group. The same group offset is then added
    # to each target and prediction, leaving the within-group correlation at 0.
    for group, sign in (("A", -1), ("B", 1)):
        center = separation * sign
        yield dict(split="train", group=group, target=center, prediction="")
        for actual, pred in zip((-1, -1, 1, 1), (-1, 1, -1, 1)):
            yield dict(split="test", group=group, target=center + actual, prediction=center + pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path)
    args = parser.parse_args()
    if args.csv is None:
        print("Group separation | Pooled r | Macro r | Training group mean r")
        for separation in (0, .5, 1, 2, 4):
            pooled, macro, baseline, _ = summarize(example(separation))
            assert math.isclose(pooled, separation ** 2 / (1 + separation ** 2), abs_tol=1e-12)
            assert math.isclose(macro, 0, abs_tol=1e-12)
            print(f"{2 * separation:>16g} | {show(pooled):>8} | {show(macro):>7} | {show(baseline)}")
        return
    with args.csv.open(encoding="utf-8-sig", newline="") as source:
        pooled, macro, baseline, groups = summarize(csv.DictReader(source))
    for group, size, r in groups:
        print(f"{group}\tn={size}\tr={show(r)}")
    defined = sum(r is not None for _, _, r in groups)
    print(f"Pooled r: {show(pooled)}\nMacro r ({defined}/{len(groups)} groups): {show(macro)}")
    print(f"Pooled r of training group means: {show(baseline)}")


if __name__ == "__main__":
    main()
