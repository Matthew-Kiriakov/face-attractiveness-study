# Evaluation diagnostics | SCUT and MEBeauty

2026-11-19. Post hoc analyses on test samples already used in the manuscript. The original S/B predictions, reports, and protocol are source inputs. Bootstrap uses 5,000 draws and seed 20261119. The original input hashes and library versions were verified before reconstructing the raw SCUT fit and fitting the new MEBeauty-only model.

## Table 1. SCUT | Same 1,196 test photographs

| Model and target scale | Pooled r | Macro r | Pooled minus macro r [95% interval] | Pooled R² |
|---|---:|---:|---:|---:|
| 70 features, within-group z scale | 0.672 | 0.678 | -0.006 [-0.019, 0.010] | 0.451 |
| 70 features, back-transformed to 1–5 | 0.824 | 0.678 | 0.146 [0.118, 0.177] | 0.678 |
| 70 features + group indicators, 1–5 | 0.823 | 0.677 | 0.147 [0.120, 0.178] | 0.678 |
| Development group mean only, 1–5 | 0.608 | Undefined | — | 0.369 |

| Group | n | r, 70 features (same inside group on either scale) | r, 70 features + group indicators |
|---|---:|---:|---:|
| CM 10k_US | 148 | 0.576 | 0.567 |
| AM DataTang | 328 | 0.620 | 0.626 |
| AM internet | 145 | 0.809 | 0.803 |
| CF 10k_US | 124 | 0.671 | 0.670 |
| AF internet | 246 | 0.685 | 0.686 |
| AF DataTang | 205 | 0.707 | 0.707 |

For the raw-target model, pooled r after centering both columns within each test group is 0.699. Between-group means account for 54.6% of its pooled covariance numerator (total 0.26412, within 0.11980). The variance terms in Pearson's denominator also change.

Macro r gives equal weight to each group. A constant prediction within each group makes its r undefined. The back-transformation applies the development mean and standard deviation of each group to the saved predictions and requires no refit. Pooled R² uses the mean of the 1,196 test targets on that row's scale; compare R² values on the same scale. Intervals resample photographs within groups and reuse the same draws for the pooled and macro metrics.

## Table 2. MEBeauty | Same fold 0 test photographs

| Group | n | r A [95% interval] | R² A [95% interval] | ΔMSE A [95% interval] | r B | R² B | r C | R² C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| caucasian_female | 108 | 0.218 [0.037, 0.377] | -0.018 [-0.204, 0.112] | 0.017 [-0.114, 0.168] | 0.306 | 0.084 | 0.205 | -0.012 |
| caucasian_male | 86 | 0.160 [-0.025, 0.343] | -0.158 [-0.431, 0.044] | 0.130 [-0.035, 0.343] | 0.307 | 0.053 | 0.377 | 0.134 |
| asian_female | 38 | 0.290 [0.037, 0.653] | -0.047 [-0.764, 0.241] | 0.048 [-0.283, 0.531] | 0.412 | 0.152 | 0.577 | 0.322 |
| asian_male | 29 | 0.678 [0.382, 0.869] | 0.436 [0.059, 0.667] | -0.347 [-0.610, -0.035] | 0.698 | 0.460 | 0.732 | 0.477 |
| Equal-group mean | 261 | 0.337 [0.227, 0.454] | 0.053 [-0.169, 0.157] | — | 0.431 | 0.187 | 0.473 | 0.230 |

A = SCUT only. B = SCUT + MEBeauty folds 1–4. C = MEBeauty folds 1–4 only. A and B use saved boosting predictions. C uses the same 70 features, estimator settings, and MEBeauty training pre-processing as B. The score is evaluated for all three against the saved group-standardized test target (scale fitted on all MEBeauty folds in step B).

ΔMSE = model MSE minus the MSE of a constant fitted to the *test group mean*. Positive is worse. The constant is an evaluation reference and uses the observed test outcomes. It is refitted within each paired bootstrap sample. This comparison quantifies error on the reported scale with the models fixed. A prospective comparison would fix the baseline using training data.

| Paired contrast | Δ macro r [95% interval] | Δ mean group R² [95% interval] |
|---|---:|---:|
| B − A | 0.094 [0.036, 0.145] | 0.134 [0.074, 0.218] |
| C − A | 0.136 [0.010, 0.228] | 0.177 [0.059, 0.324] |
| B − C | -0.042 [-0.107, 0.045] | -0.043 [-0.122, 0.035] |

Intervals use paired resampling within each group. Small groups retain the same weight as large groups. The intervals describe variation across test photographs for the fixed models. They are descriptive and unadjusted for multiple comparisons.

The inputs are `data/lockbox_S_predictions.csv`, `data/mebeauty_B_predictions.csv`, `scripts/scut_lockbox_S.py`, `scripts/posthoc_ladder_lockbox.py`, and `scripts/mebeauty_B.py`. Existing headline point estimates were reproduced from the saved predictions or the reconstructed fit. New intervals use the bootstrap specified above.
