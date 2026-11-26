"""SCUT-FBP5500 Caucasian men: frontal harmony features and a first prediction.

Landmarks are MediaPipe Face Landmarker, the same model the page uses.
Measurements are computed by scripts/measure_batch.mjs, which calls the
page's metrics.js and gates.js. Symmetry is stored in the CSV and left
out of the predictor. The published 60/40 and 5-fold file lists are not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "scut-fbp5500" / "SCUT-FBP5500_v2" / "Images"
RATINGS = ROOT / "data" / "scut-fbp5500" / "SCUT-FBP5500_v2" / "All_Ratings.xlsx"
MODEL = ROOT / "models" / "face_landmarker.task"
CACHE = ROOT / "data" / "scut_cm_landmarks.jsonl"
MEASURED = ROOT / "data" / "scut_cm_measured.json"
CSV_PATH = ROOT / "data" / "scut_cm_harmony.csv"
FIT_PATH = ROOT / "data" / "scut_cm_harmony_fit.json"
NODE = Path("node")
MEASURE_JS = ROOT / "scripts" / "measure_batch.mjs"

SEED = 20261023
COLOR_GAP = 32
LUMA_DROP = 14
HAIR_RUN = 5


def js_round(value: float) -> int:
    return int(math.floor(value + 0.5))


def luma(pixel) -> float:
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]


def estimate_trichion(rgb: np.ndarray, landmarks: list[tuple[float, float]]):
    """Port of app.js estimateTrichion. Returns normalized (x, y) or None."""
    height, width = rgb.shape[:2]
    forehead = landmarks[10]
    nasion = landmarks[168]
    menton = landmarks[152]
    dx = menton[0] - nasion[0]
    dy = menton[1] - nasion[1]
    length = math.hypot(dx, dy) or 1.0
    upx = -dx / length
    upy = -dy / length
    px = -upy
    py = upx

    def read(nx: float, ny: float):
        x = js_round(nx * width)
        y = js_round(ny * height)
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        red = green = blue = 0.0
        count = 0
        for offset in range(-4, 5):
            xx = js_round(x + px * offset * 2)
            yy = js_round(y + py * offset * 2)
            if xx < 0 or yy < 0 or xx >= width or yy >= height:
                continue
            pixel = rgb[yy, xx]
            red += float(pixel[0])
            green += float(pixel[1])
            blue += float(pixel[2])
            count += 1
        if not count:
            return None
        return (red / count, green / count, blue / count)

    skins = []
    for step_index in range(6):
        t = 0.25 + step_index * 0.1
        sample = read(
            forehead[0] + (nasion[0] - forehead[0]) * t,
            forehead[1] + (nasion[1] - forehead[1]) * t,
        )
        if sample:
            skins.append(sample)
    if not skins:
        return None
    skin = tuple(sum(sample[channel] for sample in skins) / len(skins) for channel in range(3))
    pixels_per_unit = math.hypot(upx * width, upy * height) or 1.0
    step = 2 / pixels_per_unit
    max_steps = max(8, js_round((length * 0.55) / step))
    skin_luma = luma(skin)
    run = 0
    start = None
    for i in range(1, max_steps + 1):
        x = forehead[0] + upx * step * i
        y = forehead[1] + upy * step * i
        if x < 0.004 or y < 0.004 or x > 0.996 or y > 0.996:
            break
        color = read(x, y)
        if color is None:
            break
        unlike = math.hypot(*(color[c] - skin[c] for c in range(3))) > COLOR_GAP and skin_luma - luma(color) > LUMA_DROP
        if unlike:
            if run == 0:
                start = (x, y)
            run += 1
            if run >= HAIR_RUN:
                return start
        else:
            run = 0
            start = None
    return None


def fill_iris(landmarks: list):
    def center(indices):
        xs = []
        ys = []
        for index in indices:
            point = landmarks[index] if index < len(landmarks) else None
            if point is None:
                return None
            xs.append(point[0])
            ys.append(point[1])
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    while len(landmarks) <= 473:
        landmarks.append(None)
    if landmarks[468] is None:
        landmarks[468] = center([33, 133, 159, 145])
    if landmarks[473] is None:
        landmarks[473] = center([263, 362, 386, 374])
    return landmarks


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["file"]] = row
    return done


def detect(limit: int) -> None:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.core.image import Image as MpImage
    from mediapipe.tasks.python.vision.core.image import ImageFormat

    files = sorted(IMAGES.glob("CM*.jpg"), key=lambda path: int(path.stem[2:]))
    if limit:
        files = files[:limit]
    done = load_done(CACHE)
    pending = [path for path in files if path.name not in done]
    print(f"images {len(files)}, cached {len(files) - len(pending)}, to detect {len(pending)}", flush=True)
    if not pending:
        return

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with CACHE.open("a", encoding="utf-8") as handle:
            for index, path in enumerate(pending, start=1):
                rgb = np.asarray(Image.open(path).convert("RGB"))
                if not rgb.flags.c_contiguous:
                    rgb = np.ascontiguousarray(rgb)
                height, width = rgb.shape[:2]
                try:
                    result = landmarker.detect(MpImage(image_format=ImageFormat.SRGB, data=rgb))
                except Exception as error:
                    row = {"file": path.name, "error": str(error)}
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(f"{index}/{len(pending)} {path.name} ERROR {error}", flush=True)
                    continue
                if not result.face_landmarks:
                    row = {"file": path.name, "error": "no face"}
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(f"{index}/{len(pending)} {path.name} no face", flush=True)
                    continue
                raw = result.face_landmarks[0]
                landmarks = [(point.x, point.y) for point in raw]
                landmarks = fill_iris(landmarks)
                if any(point is None for point in landmarks[:474]):
                    row = {"file": path.name, "error": "incomplete mesh"}
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    continue
                found = estimate_trichion(rgb, landmarks)
                trichion = found if found is not None else landmarks[10]
                glabella = (
                    (landmarks[107][0] + landmarks[336][0]) / 2,
                    (landmarks[107][1] + landmarks[336][1]) / 2,
                )
                row = {
                    "file": path.name,
                    "w": width,
                    "h": height,
                    "n_lm": len(raw),
                    "trichion_found": found is not None,
                    "trichion": [round(trichion[0], 7), round(trichion[1], 7)],
                    "glabella": [round(glabella[0], 7), round(glabella[1], 7)],
                    "lm": [[round(point[0], 7), round(point[1], 7)] for point in landmarks],
                }
                handle.write(json.dumps(row) + "\n")
                if index % 25 == 0 or index == len(pending):
                    handle.flush()
                    print(f"{index}/{len(pending)} {path.name}", flush=True)
    finally:
        landmarker.close()


def measure() -> dict:
    subprocess.run(
        [str(NODE), str(MEASURE_JS), str(CACHE), str(MEASURED)],
        check=True,
    )
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def load_ratings() -> dict[str, list[tuple]]:
    book = load_workbook(RATINGS, read_only=True, data_only=True)
    sheet = book["Caucasian_Male"]
    rows = sheet.iter_rows(values_only=True)
    header = [str(cell).strip() if cell is not None else "" for cell in next(rows)]
    columns = {name.lower(): index for index, name in enumerate(header)}
    i_rater = columns["rater"]
    i_file = columns["filename"]
    i_rating = columns["rating"]
    buckets: dict[str, list[tuple]] = {}
    for row in rows:
        if not row or row[i_file] is None or row[i_rating] is None:
            continue
        name = Path(str(row[i_file]).strip()).name
        if not name.upper().startswith("CM"):
            continue
        buckets.setdefault(name, []).append((row[i_rater], float(row[i_rating])))
    book.close()
    return buckets


def pearson(left, right) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom == 0:
        return None
    return float((x * y).sum() / denom)


def spearman(left, right) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    return pearson(rank(x[mask]), rank(y[mask]))


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average ties
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            mean_rank = float(ranks[order[start:end]].mean())
            ranks[order[start:end]] = mean_rank
        start = end
    return ranks


def r2_score(actual, predicted) -> float | None:
    y = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return None
    residual = float(((y - pred) ** 2).sum())
    return 1 - residual / total


def mae(actual, predicted) -> float:
    y = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(y - pred)))


def hgb():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=200,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=0,
    )


def ridge():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def oof_predict(factory, features: np.ndarray, target: np.ndarray, n_repeats: int, seed: int) -> np.ndarray:
    splitter = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    total = np.zeros(len(target), dtype=float)
    count = np.zeros(len(target), dtype=float)
    for train, test in splitter.split(features):
        model = factory()
        model.fit(features[train], target[train])
        total[test] += model.predict(features[test])
        count[test] += 1
    return total / count


def pack(name: str, actual, predicted) -> dict:
    return {
        "pearson_r": pearson(actual, predicted),
        "spearman_r": spearman(actual, predicted),
        "r2": r2_score(actual, predicted),
        "mae": mae(actual, predicted),
        "n": int(len(actual)),
        "model": name,
    }


def split_half(buckets: dict[str, list[tuple]], files: list[str]) -> dict:
    raters = sorted({rater for name in files for rater, _rating in buckets[name]}, key=lambda value: str(value))
    even = set(raters[::2])
    left = []
    right = []
    for name in files:
        a = [rating for rater, rating in buckets[name] if rater in even]
        b = [rating for rater, rating in buckets[name] if rater not in even]
        if a and b:
            left.append(sum(a) / len(a))
            right.append(sum(b) / len(b))
    half_r = pearson(left, right)
    reliability = None if half_r is None or half_r <= -1 else (2 * half_r) / (1 + half_r)
    ceiling = None if reliability is None or reliability <= 0 else math.sqrt(reliability)
    return {
        "n_faces": len(left),
        "n_raters": len(raters),
        "split": "even/odd after sorting rater ids",
        "half_pearson_r": half_r,
        "spearman_brown_reliability": reliability,
        "correlation_ceiling_sqrt_reliability": ceiling,
    }


def trichion_above(cache_row: dict | None) -> float | None:
    if not cache_row or cache_row.get("error") or "lm" not in cache_row:
        return None
    forehead = cache_row["lm"][10]
    menton = cache_row["lm"][152]
    nasion = cache_row["lm"][168]
    trichion = cache_row["trichion"]
    span = menton[1] - nasion[1]
    if span <= 1e-6:
        return None
    return (forehead[1] - trichion[1]) / span


def write_table(catalog: list[dict], measured_rows: list[dict], buckets: dict[str, list[tuple]]) -> list[dict]:
    metric_ids = [item["id"] for item in catalog]
    cached = load_done(CACHE)
    records = []
    for row in measured_rows:
        ratings = buckets.get(row["file"])
        if not ratings or row.get("error") or not row.get("values"):
            continue
        scores = [rating for _rater, rating in ratings]
        values = row["values"]
        record = {
            "file": row["file"],
            "rating_mean": sum(scores) / len(scores),
            "rating_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "rating_n": len(scores),
            "trichion_found": int(row["trichion_found"]),
            "trichion_above": trichion_above(cached.get(row["file"])),
            "roll_deg": row["roll"],
            "half_ratio": row["half"],
            "ticks": row["ticks"],
            "ticks_of": row["ticks_of"],
        }
        for metric_id in metric_ids:
            record[metric_id] = values.get(metric_id)
        records.append(record)
    records.sort(key=lambda record: int(Path(record["file"]).stem[2:]))
    fieldnames = [
        "file",
        "rating_mean",
        "rating_sd",
        "rating_n",
        "trichion_found",
        "trichion_above",
        "roll_deg",
        "half_ratio",
        "ticks",
        "ticks_of",
        *metric_ids,
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return records


def matrix(records: list[dict], feature_ids: list[str]):
    target = np.array([record["rating_mean"] for record in records], dtype=float)
    columns = []
    for feature_id in feature_ids:
        column = []
        for record in records:
            value = record[feature_id]
            column.append(np.nan if value is None else float(value))
        columns.append(column)
    return np.column_stack(columns), target


def univariate(records: list[dict], feature_ids: list[str], names: dict[str, str]) -> list[dict]:
    target = [record["rating_mean"] for record in records]
    rows = []
    for feature_id in feature_ids:
        column = [record[feature_id] for record in records]
        rows.append(
            {
                "id": feature_id,
                "name": names[feature_id],
                "pearson_r": pearson(column, target),
                "n": int(np.isfinite(np.array(column, dtype=float)).sum()),
            }
        )
    rows.sort(key=lambda item: abs(item["pearson_r"] or 0), reverse=True)
    return rows


def fit_block(records: list[dict], feature_ids: list[str], label: str, nulls: bool) -> dict:
    features, target = matrix(records, feature_ids)
    print(f"fit {label} n={len(records)}", flush=True)
    hgb_pred = oof_predict(hgb, features, target, n_repeats=5, seed=SEED)
    ridge_pred = oof_predict(ridge, features, target, n_repeats=5, seed=SEED)
    block = {
        "label": label,
        "n": len(records),
        "n_features": len(feature_ids),
        "rating_mean": float(target.mean()),
        "rating_sd": float(target.std(ddof=1)),
        "hist_gradient_boosting": pack("HistGradientBoosting max_depth=3 min_leaf=20 l2=1", target, hgb_pred),
        "ridge_alpha_10": pack("Ridge alpha=10, median impute, standardized", target, ridge_pred),
    }
    if not nulls:
        return block
    rng = np.random.default_rng(SEED)
    null = []
    for draw in range(30):
        shuffled = rng.permutation(target)
        pred = oof_predict(hgb, features, shuffled, n_repeats=1, seed=SEED + 1 + draw)
        null.append(pearson(shuffled, pred))
        if (draw + 1) % 10 == 0:
            print(f"null {draw + 1}/30", flush=True)
    observed = block["hist_gradient_boosting"]["pearson_r"]
    null_abs = [abs(value) for value in null if value is not None]
    beaten = sum(1 for value in null_abs if observed is not None and value >= abs(observed))
    block["null_hgb_5fold_30_shuffles"] = {
        "mean_abs_r": float(np.mean(null_abs)) if null_abs else None,
        "max_abs_r": float(np.max(null_abs)) if null_abs else None,
        "n_null_ge_observed": beaten,
        "n_null": len(null_abs),
    }
    return block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--detect-only", action="store_true")
    args = parser.parse_args()
    if not IMAGES.exists():
        sys.exit(f"missing images: {IMAGES}")
    if not MODEL.exists():
        sys.exit(f"missing model: {MODEL}")
    detect(args.limit)
    if args.detect_only:
        return
    if args.limit:
        sys.exit("limit is only for a detection smoke test; rerun without it to measure and fit")
    measured = measure()
    buckets = load_ratings()
    records = write_table(measured["catalog"], measured["rows"], buckets)
    names = {item["id"]: item["name"] for item in measured["catalog"]}
    harmony_ids = [item["id"] for item in measured["catalog"] if item["group"] != "Симметрия"]
    symmetry_ids = [item["id"] for item in measured["catalog"] if item["group"] == "Симметрия"]
    usable = [record for record in records if record["file"] in buckets]
    frontal = [
        record
        for record in usable
        if record["roll_deg"] is not None
        and abs(record["roll_deg"]) <= 7
        and record["half_ratio"] is not None
        and 0.85 <= record["half_ratio"] <= 1.18
    ]
    hair = [record for record in usable if record["trichion_found"] == 1]
    blocks = [fit_block(usable, harmony_ids, "all detected faces", nulls=True)]
    if len(frontal) >= 200 and len(frontal) != len(usable):
        blocks.append(fit_block(frontal, harmony_ids, "roll<=7 and half-width 0.85-1.18", nulls=False))
    if len(hair) >= 200 and len(hair) != len(usable):
        blocks.append(fit_block(hair, harmony_ids, "hairline found, not forehead fallback", nulls=False))
    ceiling = split_half(buckets, [record["file"] for record in usable])
    associations = univariate(usable, harmony_ids, names)
    tick_r = pearson([record["ticks"] for record in usable], [record["rating_mean"] for record in usable])
    failed = [row["file"] for row in measured["rows"] if row.get("error") or not row.get("values")]
    missing_rating = [row["file"] for row in measured["rows"] if row["file"] not in buckets and not row.get("error")]
    report = {
        "dataset": "SCUT-FBP5500 v2, Caucasian_Male sheet, scale 1-5",
        "n_images": 750,
        "n_csv": len(records),
        "failed_files": failed,
        "n_missing_rating": missing_rating,
        "trichion_found": sum(record["trichion_found"] for record in usable),
        "trichion_above_median": float(np.median([record["trichion_above"] for record in usable if record["trichion_above"] is not None])),
        "trichion_barely_moved": sum(1 for record in usable if record["trichion_above"] is not None and record["trichion_above"] < 0.03),
        "frontal_qc_pass": len(frontal),
        "predictor": "harmony metrics only; symmetry group excluded",
        "symmetry_excluded": symmetry_ids,
        "features": harmony_ids,
        "validation": "RepeatedKFold 5 splits x 5 repeats, seed 20261023. Not the dataset's 60/40 or 5 published folders.",
        "hyperparameters_fixed_before_fit": {
            "hist_gradient_boosting": {
                "learning_rate": 0.06,
                "max_iter": 200,
                "max_depth": 3,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
            },
            "ridge": {"alpha": 10.0},
        },
        "rater_ceiling": ceiling,
        "ticks_vs_rating_pearson": tick_r,
        "blocks": blocks,
        "univariate_pearson_desc": associations,
        "csv": str(CSV_PATH),
    }
    FIT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_csv", "trichion_found", "frontal_qc_pass", "rater_ceiling", "ticks_vs_rating_pearson", "blocks")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
