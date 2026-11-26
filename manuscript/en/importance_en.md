# Appendix F. Grouped feature importance

This exploratory analysis uses the SCUT development set (3,737 photographs). The figures come from the saved clusters in the project page's data of 10 November 2026. The page data schema documents how the groups were formed and evaluated. The table includes all 14 groups.

For boosting, importance was measured by the decrease in validation-fold predictive R² when columns within a group were permuted together (five folds, two repeats, four permutations). Each share is the group's positive drop divided by the sum of positive drops across groups. Non-positive drops were assigned zero for this calculation. The share measures relative sensitivity to permutation. An earlier report on 30 October used a different grouping.

| Group | Features | R² drop | Share of positive drop |
|---|---|---:|---:|
| Brow, eye, and lip contrast | `brow_L`, `brow_contrast`, `eye_contrast`, `mouth_contrast` | 0.0951 | 18.8% |
| Chin and lip tone | `chin_dL`, `chin_sd`, `lip_dL` | 0.0585 | 11.5% |
| Brow placement | `brow-gap-l`, `brow-gap-r`, `brow-tilt-l`, `brow-tilt-r`, `brow-len-l`, `brow-len-r` | 0.0483 | 9.5% |
| Nasal geometry | `bridge-len`, `nose-canthi`, `nose-face`, `nostril-l`, `nostril-r`, `tip-shift` | 0.0473 | 9.4% |
| Mouth and chin | `lip-ratio`, `cupid`, `mouth-nose`, `mouth-face`, `chin-philtrum`, `corners` | 0.0450 | 8.9% |
| Eye size and spacing | `eye-gap`, `eye-shape-l`, `eye-shape-r`, `ipd-w`, `intercanthal-face`, `eye-face`, `canthal-l`, `canthal-r` | 0.0401 | 7.9% |
| Face shape | `face-wh`, `temple-w`, `jaw-w`, `wide-h`, `brow-face` | 0.0376 | 7.4% |
| Sclera, eyelids, and under-eye area | `sclera_show_low`, `sclera_show_up`, `dark_dL`, `dark_da`, `dark_db`, `bag_range`, `crease_dist`, `crease_depth`, `lid_dL`, `ridge_dL` | 0.0329 | 6.5% |
| Nose tip | `tip_w`, `tip_h`, `tip_shade` | 0.0270 | 5.3% |
| Contour fullness and averageness | `adip_pa`, `adip_lower`, `avg_dist` | 0.0268 | 5.3% |
| Vertical thirds | `third-upper`, `third-mid`, `third-low`, `lower-mid`, `midface`, `pupil-mouth` | 0.0237 | 4.7% |
| Jaw angle and chin width | `jaw_turn`, `chin_w` | 0.0119 | 2.3% |
| Jawline | `jaw-angle-l`, `jaw-angle-r`, `jaw-slope-l`, `jaw-slope-r`, `ala-jaw` | 0.0095 | 1.9% |
| Brow density and curvature | `brow_thick`, `brow_dens`, `brow_arch` | 0.0027 | 0.5% |

These results describe a fitted model's use of features. When features are interrelated, the drops depend on how features were grouped and which substitutes were available. The unique R² components in [Table D1](supplement_en.md#variance-partitioning-and-image-embeddings) are from a different calculation.
