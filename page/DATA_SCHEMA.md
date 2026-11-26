# data.json: what the page reads

`index.html` reads `data.json` when served over http. Opened from disk (`file://`) browsers refuse that, so the page falls back to `data.js`, an exact mirror. After editing `data.json` by hand run `python page/build_data.py --sync` to refresh the mirror.

Two kinds of keys:

- **Computed** (one long line each, rewritten by `python page/build_data.py`): `meta`, `sets`, `remove`, `clusters`, `commonality`, `ceiling_scut`, `face`. Development faces only (3737); the loader drops the lockbox before anything is computed; MEBeauty is not read.
- **Hand-written** (indented, kept as they are by the build): `set_labels`, `cluster_defs`, `feature_labels`, `quotes`, `remove_notes`, `mebeauty`, `lockbox`, `heldout`, `not_claimed`, `lookmaxxing`. Every text has `en` and `ru`.

All correlations are Pearson r unless named otherwise; intervals are 95 % face bootstraps `[lo, hi]`.

## Placeholders to fill

| Key | Now | Fill with |
|---|---|---|
| `lockbox.scut.result` | `null` | after run S: `{"CM 10k_US": [r, lo, hi], ...}` for the six sets, tier 2 boosting, primary (no twins). The panel then opens the lock and draws the result over the written band. |
| `lockbox.mebeauty_B.result` | `null` | after run B: `[mean of four, lo, hi]` |
| `lockbox.status` | `"sealed"` | informational |
| `lookmaxxing` | `status: "PLACEHOLDER"`, one line | the frozen claim list, when it exists (the section will need a design then) |

The fourth ladder rung ("held-out set") and the lockbox envelope read `lockbox.scut.result`: while it is `null` they show the sealed slot and the written expectation (mean of `expected_tier2_boosting`); once filled they show the mean of the six r and open. Open `index.html#opened-preview` to see the opened envelope layout with no data.

The two `result_PLACEHOLDER` strings are reminders and are not read by the page.

## Computed keys

- `meta`: `built`, `n`, `mean_r {boosting, ridge}`, `density_axes {x: [min, max], y: [min, max], cells}`.
- `sets[set]`: `n`, and for `boosting` and `ridge`: `r` (mean over 10 seeds), `ci`, `rho` (Spearman), `grid` (16 x 16 face counts, rows = judges' z from low to high, columns = prediction), `bins` (eight equal-size groups by prediction: `[mean prediction, mean judges' z, half-width of the 95 % interval of that mean, 10th percentile, 90th percentile, n]`), `pred_quantiles` (21 quantiles of the out-of-fold predictions, used for the "above x %" line under the face).
- `remove[key]`: key = removed levers joined by commas (`none`, `source`, `age`, `capture`, `source,age`, ...). `pooled`, `ci`, and `sets` when source is removed. Source kept = raw 1-5 ratings, six sets mixed, boosting told the set. Source removed = inside the set. Age / capture removed = partial correlation given perceived age / the six capture statistics (linear, inside the set when source is removed). Every r is the mean over the 10 seeds of the out-of-fold r (the pooled raw rung over 3 seeds), so `source` alone is the headline within-set r (0.694, results/2026-11-09_S_dev_reference.txt: 0.695 on 3 seeds). `python page/build_data.py --remove` recomputes only this key. **These partial correlations were computed for this page (2026-11-10), exploratory, with no expectation written beforehand.**
- `clusters[]`: `id`, `drop` (held-out R2 lost when the cluster's columns are shuffled together; boosting, 5 folds x 2 repeats x 4 shuffles), `share` (drop / sum of drops), `features[] {id, drop, within (share inside the cluster), rho (mean within-set Spearman with the judges)}`. **Computed for this page; the grouping in `cluster_defs` is a proposal, the seven image groups are the repo's.**
- `commonality[set]`: `G`, `Q`, `A`, `G+Q`, `G+A`, `Q+A`, `G+Q+A` as `[mean, p2.5, p97.5]`, copied from `data/cache_expl/vp.json` (ridge, 30 subsamples; results/2026-10-29_expl_vp.txt).
- `ceiling_scut[set]`: `one_judge` (mean r of one judge with the mean of the other 59), `one_judge_range` (10th-90th percentile over judges), `ceiling` (square root of the reliability of the 60-judge mean). Computed for this page from All_Ratings.xlsx, development faces.
- `face`:
  - `points.index`: 94 entries: 92 MediaPipe indices + `trichion` + `glabella`. `points.model`: the 42 named points `js/metrics.js` measures. `points.paths`: which indices each drawn curve runs through.
  - `props`: the 42 proportions in column order. `grid`: z values of the partial-dependence curves. `range`: slider limit in SD.
  - `ridge {intercept, coef{prop}}`: pooled tier-2 ridge fitted on all development faces, slope per 1 z. `boosting {base, pd{prop}[9]}`: `base` = prediction at the all-average row; `pd` = partial dependence on `grid`, minus its value at 0. Browser value = base + sum of the curves.
  - `sets[set]`: `n` (faces with a found hairline), `mean` (94 x [x, y]; hairline-to-chin axis vertical, length 1), `stats{prop}: [mean, sd]` in raw units (after 1/99 % winsorising), `morph{prop}` (94 x [dx, dy] per +1 z asked), `response[k][j]` (z of proportion k that the drawn face measures per +1 z asked of proportion j).

## Hand-written keys

- `set_labels[set] {race: C|A, sex: M|F, source, en, ru}`.
- `cluster_defs[] {id, kind: geometry|image, en, ru, features[]}`; `feature_labels[feature] {en, ru}`.
- `quotes[] {date, lever: source|age|capture|all, kind, src, en, ru, paraphrased?}`: shown next to the lever that is switched on. A paraphrase is displayed without quotation marks.
- `remove_notes[] {lever, value, src, en, ru}`: numbers taken from result files as they stand.
- `mebeauty {groups[] {id, n, ceiling, indomain [r, lo, hi], transfer [r, lo, hi], given_age, capture_only, age_label_r}, mean_of_four}`: folds 1-4, boosting.
- `lockbox {status, frozen, scut {faces, primary_faces, band, expected_tier2_boosting{set}, expected_geometry_boosting{set}, h1_band_geometry, downgrade_below_mean, result}, mebeauty_B {faces, expected_mean_of_four [lo, hi], useful_at, downgrade_below_mean, result}, registered[] {en, ru}}`.
- `not_claimed[] {en, ru}`; `lookmaxxing {status, en, ru}`.

## Round 4: `heldout` (hand-written, confirmatory)

Copied by hand (one-off parse) from `results/S_lockbox_run.txt` and `results/mebeauty_B_run.txt`, PRIMARY block; `build_data.py` never reads lockbox files. Per-set aggregates only: no per-face rows.

- `heldout {status, note, src[], faces_six, primary, twins, mean_r, mean_r2, written_mean, downgrade_below, sets, meb, claims, checks}`.
- `sets[set] {n, r [r, lo, hi], r2, ridge, geometry, h2 {model, judge, diff [d, lo, hi]}, h3 {pop, sex: [gap, lo, hi]}}`: tier 2 boosting r; ridge and geometry are point estimates.
- `meb {a_groups {group: {n, r [r, lo, hi], word}}, a_mean, b_mean, b_ridge_mean, pool_gain [g, lo, hi], a_pooled [r, lo, hi]}`: MEBeauty fold 0, four primary groups.
- `claims[] {id, group: free|jaw|sym, verdict: supported|contradicted|indistinct|no_gain, text, measure, sets {set: [rho, lo, hi]}, mean | mean_m + mean_f}`: C13-C15 hold both sexes (men: three sets, women: three); C11 men only, C12 women only; C10 is a gain in r, not a Spearman.
- `checks[] {id, written, result, verdict: hit|partial|miss}`: the "written, then measured" list; texts hand-written, EN and RU.
- Also changed by hand: `lockbox.status` opened, `lockbox.frozen` 2026-11-11 (v1.4), `lockbox.scut.faces` 1246 and `primary_faces` 1196, `lockbox.scut.result` filled (six [r, lo, hi]), `lockbox.mebeauty_B.result {A_mean_of_four, B_mean_of_four}` and `expected_A_mean_of_four`, `lookmaxxing.status` opened (text unused), `not_claimed` rewritten (11 items).
- The placeholders table above (`lockbox.scut.result`, `lookmaxxing`) is filled; the sealed-envelope code path remains only as the opened state.
