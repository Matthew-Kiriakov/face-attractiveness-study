# Review follow-up on editorial v4

## SCUT | Sensitivity to two excluded groups

Fit one booster with the original 70 features and fixed hyperparameters on all eight development groups. Each group uses its own development statistics. Retain the original split and reapply the original ArcFace threshold to the expanded training sample. Both models are compared on identical photographs in the six original groups. There are 1246 original-six test photographs before filtering, 1196 under the original filter, and 1195 on the matched sample. This analysis uses previously opened test data.

| Group | Training n | Test before filtering | Test after filtering | Original six-group fit r [95%] | Eight-group fit r [95%] |
|---|---:|---:|---:|---:|---:|
| CM 10k_US | 448 | 150 | 147 | 0.570 [0.442, 0.676] | 0.585 [0.462, 0.690] |
| AM DataTang | 1006 | 336 | 328 | 0.620 [0.554, 0.679] | 0.631 [0.565, 0.690] |
| AM internet | 488 | 162 | 145 | 0.809 [0.746, 0.860] | 0.815 [0.754, 0.863] |
| CF 10k_US | 380 | 126 | 124 | 0.671 [0.568, 0.755] | 0.660 [0.558, 0.743] |
| AF internet | 783 | 262 | 246 | 0.685 [0.588, 0.761] | 0.679 [0.579, 0.755] |
| AF DataTang | 632 | 210 | 205 | 0.707 [0.640, 0.765] | 0.710 [0.640, 0.771] |
| CM internet | 114 | 38 | 33 | — | 0.227 [-0.124, 0.573] |
| CF internet | 183 | 61 | 57 | — | 0.369 [0.165, 0.558] |

- Original fit, six groups on the matched sample: **0.677**.
- Expanded fit, same six groups and same photographs: **0.680**. Change from the original fit: 0.003 [-0.005, 0.011].
- Expanded fit, eight groups: **0.585**. Change in the macro average from reporting eight instead of six with the expanded fit: -0.095 [-0.147, -0.045].

Intervals use 5,000 paired bootstrap samples of photographs within groups with fixed fits and group membership. The expanded sample is a sensitivity scenario for the original six-group result.

## MEBeauty | Error against a constant fixed before test scoring

For every group, use the mean and sample standard deviation of the MEBeauty scores in folds 1–4, restricted to photographs with features as in step B. Score the saved A predictions as z-scores on that reference scale. Equivalently, interpret a prediction in raw beauty-score units as the group mean plus the saved prediction times the group standard deviation. Both the scale and the constant (zero on this z scale) are fixed by folds 1–4. The SCUT model weights are fixed and the interpretation requires group metadata and labelled reference data from MEBeauty. This post hoc calculation reuses part 0.

| Group | n | r | Model MSE | Fixed-mean MSE | Model minus fixed-mean MSE [95%] | MSE skill vs fixed mean |
|---|---:|---:|---:|---:|---:|---:|
| caucasian_female | 108 | 0.218 | 0.955 | 0.937 | 0.018 [-0.127, 0.160] | -0.019 |
| caucasian_male | 86 | 0.160 | 0.921 | 0.788 | 0.133 [-0.044, 0.326] | -0.168 |
| asian_female | 38 | 0.290 | 1.082 | 1.032 | 0.050 [-0.319, 0.502] | -0.048 |
| asian_male | 29 | 0.678 | 0.423 | 0.774 | -0.351 [-0.651, -0.071] | 0.454 |

Positive error differences favour the constant. Intervals use 5,000 paired draws of test photographs within each group with a fixed constant. MSE skill is 1 − model MSE / fixed-mean MSE. The original manuscript R² uses test-group means and the original all-fold scale.
