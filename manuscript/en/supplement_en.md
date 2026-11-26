# Supplementary appendices

Appendix E (technical details) is at the end of the [main article](paper_en.md). On SCUT, a group is a combination of sex, population category, and image source. The 70 features comprise 42 geometric measurements and 28 detail features. The text distinguishes analyses recorded before the final run, post hoc analyses, and exploratory results using development data.

## Appendix A. Other decisions made after seeing results

**A1.** Expectations for the unique age contribution (A) in two groups were changed after a trial run on development photographs, before opening the held-out ratings. The trial used development photographs only.

**A2.** CM internet and CF internet (38 and 61 held-out photographs) were omitted from training and reporting under a minimum of 200 development photographs. This replaced the earlier instruction to report small groups without a verdict. The decision was made on the day of the run. These had been the weakest groups in preliminary analysis.

**A3.** The early sequence of comparisons on the held-out photographs (`posthoc_ladder_lockbox.txt` and `posthoc_ladder_ci.txt`) was produced after the ratings were opened. The specification was recorded before calculation. The intervals were calculated after their point estimates were known. Their specification (1,000 paired bootstrap resamples of photographs within groups) preceded the interval calculation.

**A4.** I used four sliders to rate eyes, angularity, dimorphism, and grooming on 200 development photographs. This describes my own judgments and was separate from the main model. The plan recorded on 6 November specified 444 photographs and one analysis. I stopped after rating 200 (195 analyzed) for reasons unrelated to the results, after interim reviews with visible results at sample sizes from 20 to 195. The recorded analysis was not completed. All slider results were exploratory. Twelve of the original 19 repeat pairs were redrawn randomly within groups and 20 pairs were placed in one session. The paper does not use slider results. I was the sole rater.

**A5.** On 12 November the intervals for the sequence of comparisons were calculated with the point estimates already known. Split-half reliability of the 60-rater mean and correlations among the looksmaxxing measures were exploratory calculations on development photographs.

**A6.** The first trial runs wrote a result file using development photographs in place of held-out photographs. The guard was corrected and the file removed. The trial runs did not read the held-out ratings.

## Appendix B. Features and measures for looksmaxxing claims

Facial landmarks are from MediaPipe Face Landmarker (478 points). Zygion is the lateral extreme at cheekbone level, gonion the angle of the jaw contour, trichion the hairline mark, glabella the point between the eyebrows, subnasale the base of the nose, menton the bottom of the chin, and stomion the center of the lip opening. IPD is interpupillary distance. Left and right refer to the image. Geometry is measured in `js/metrics.js`.

**42 geometric measurements.** Seven depend on the hairline mark: third-upper, third-mid, third-low, wide-h, pupil-mouth, bridge-len, and face-wh. The research log records a known offset in that mark. These measurements were retained.

| Feature | Definition |
|---|---|
| face-wh | Width between the zygia divided by hairline-to-chin height |
| temple-w | Temple width divided by face width |
| jaw-w | Distance between the gonia divided by face width |
| wide-h | Height of the widest location on the outline, as a percentage of face height |
| third-upper, third-mid, third-low | Hairline to glabella, glabella to subnasale, and subnasale to chin, each as a percentage of their sum |
| lower-mid | Lower third (subnasale to chin) divided by middle third (glabella to subnasale) |
| brow-face | Glabella-to-chin height divided by face width |
| midface | Distance from pupil midpoint to stomion divided by IPD |
| eye-gap | Distance between inner canthi divided by mean eye width |
| eye-shape-l, eye-shape-r | Eye width divided by eye opening height |
| ipd-w | IPD divided by face width |
| intercanthal-face | Distance between inner canthi divided by face width |
| eye-face | Mean eye width divided by face width |
| canthal-l, canthal-r | Angle from inner to outer canthus relative to the facial horizontal, in degrees |
| brow-gap-l, brow-gap-r | Vertical gap from brow midpoint to upper eyelid divided by eye opening height |
| brow-tilt-l, brow-tilt-r | Tilt from brow head to tail |
| brow-len-l, brow-len-r | Brow length divided by eye width |
| bridge-len | Nasion-to-tip distance as a percentage of face height |
| nose-canthi, nose-face | Nasal alar width divided by intercanthal distance or face width |
| nostril-l, nostril-r | Tilt of the line joining the top and bottom of the nostril |
| tip-shift | Horizontal offset of nose tip from the midline, as a percentage of face width |
| lip-ratio | Lower lip height divided by upper lip height |
| cupid | Cupid's bow depth as a percentage of mouth width |
| mouth-nose, mouth-face | Mouth width divided by nose width or face width |
| chin-philtrum | Stomion-to-chin distance divided by subnasale-to-stomion distance |
| pupil-mouth | Pupil-to-mouth distance as a percentage of face height |
| corners | Mouth corner elevation relative to lip center, as a percentage of mouth width |
| jaw-angle-l, jaw-angle-r | Angle at gonion between the directions to zygion and to the chin |
| jaw-slope-l, jaw-slope-r | Tilt of the gonion-to-chin line |
| ala-jaw | Mean absolute difference in tilt between the ala-to-mouth-corner line and jawline |

**28 detail features.** The seven categories combine landmarks and pixels. L, a, and b denote Lab color channels. See `block2a_colour.py`, `block2b_features.py`, `block2c_features.py`, and `block5_dimorphism.py`.

| Feature | Category | Definition |
|---|---|---|
| brow_L | Contrast | Mean L of the darkest 40% of pixels inside the eyebrows |
| brow_contrast | Contrast | Skin L minus brow_L |
| eye_contrast | Contrast | Skin L minus mean L within the eye opening |
| mouth_contrast | Contrast | Mean a inside the outer lip contour minus skin a |
| adip_pa | Facial outline | Squared perimeter divided by face outline area (mesh contour). Lower values indicate a rounder, fuller contour |
| adip_lower | Facial outline | Same measure for the lower face from cheek to cheek via the chin |
| avg_dist | Facial outline | Root mean square distance of 40 landmarks to the development-group mean shape after Procrustes alignment |
| chin_dL, lip_dL | Chin and lip tone | Cheek L minus L in the chin region or the region above the upper lip |
| chin_sd | Chin and lip tone | Standard deviation of L in the chin region |
| brow_thick, brow_dens, brow_arch | Eyebrows | Mean dark run across the brow in IPD units, dark-pixel fraction in an expanded brow outline, and peak height above the brow head-to-tail chord |
| sclera_show_low, sclera_show_up | Eyes and lids | Visible sclera below and above the iris, in iris radii (negative when obscured by the lid) |
| dark_dL, dark_da, dark_db | Eyes and lids | Cheek minus under-eye skin in L, a, and b |
| bag_range | Eyes and lids | Range of row-mean L in the strip 0.08–0.35 IPD below the lower lid |
| crease_dist, crease_depth | Eyes and lids | Distance in IPD units and depth in L of the darkest row above the upper lid margin |
| lid_dL | Eyes and lids | Eyelid skin L minus cheek L |
| ridge_dL | Eyes and lids | Forehead L minus L between the brow peak and upper eyelid (supraorbital shading) |
| tip_w, tip_h, tip_shade | Nose tip | Distance between nostril bottoms divided by alar width, subnasale-to-tip distance divided by nose length, and tip L minus alar L |
| jaw_turn | Jaw and chin | Mean jawline turning angle near gonion across both sides, in degrees |
| chin_w | Jaw and chin | Chin width divided by jaw width |

**Measures for looksmaxxing claims.** The sign is defined so that a positive signed association agrees with the claim. An asterisk marks measures outside the 70 features. IDs C01–C15 match the protocol.

| Claim | Measure | Sign |
|---|---|---|
| C01 Long chin | chin-philtrum | + |
| C02 Cheekbones wider relative to the jaw | zyg_jaw* (distance between mesh points 116 and 345 divided by distance between 58 and 288, `angularity.py`) | + |
| C03 Positive canthal tilt | Mean of canthal-l and canthal-r | + |
| C04 Closer eyes | eye-gap | − |
| C05 Less visible sclera below the iris | sclera_show_low | − |
| C06 Wider face relative to height | face-wh | + |
| C07 Equal vertical thirds | Sum of absolute deviations of each percentage from 33.3 | − |
| C08 Lower eyebrows | Mean of brow-gap-l and brow-gap-r | − |
| C09 Thicker, denser eyebrows | Mean standardized brow_thick and brow_dens | + |
| C10 Symmetry | Ridge r gain from ten symmetry features* (sym_rms, sym_eyes, sym_brows, sym_mouth, sym_nose, sym_jaw, sym_h, sym_v, sym_axis, sym_tilt). These use deviations of mirrored landmark pairs from the midline in IPD units (`block3_symmetry.py`) | + |
| C11 Masculine faces in men, C12 feminine faces in women | M_core* averages six components: jaw (jaw-w, jaw_turn, chin_w), brows (reversed brow-gap-l/r, brow_thick, brow_dens, ridge_dL), eyes (reversed eye-face), nose (nose-face), lips (reversed lip_thick*, sum of lip heights divided by mouth width), and shape (face-wh). See `block5_dimorphism.py` | C11 +, C12 − |
| C13 Sharp jaw angle | jaw_turn | + |
| C14 Tapered outline | adip_pa | + |
| C15 Wide jaw | jaw-w | + |

Appendix E reports correlations among jaw measures and the calculation of M_core without its jaw component.

## Appendix C. Data access and protocol history

| Date | Record | Timing relative to the final runs |
|---|---|---|
| 2026-10-29 | Set aside 25% of each group and source using SHA-1 of “20261029:” plus filename | Before |
| 2026-11-07 | MEBeauty descriptive audit of rating distributions and splits, including fold 0 | Before |
| 2026-11-09 | v1: 70-feature model, SCUT (S) and MEBeauty (B) checks, decision criteria | Before |
| 2026-11-09 | v1.1: boosting as the main model, near-duplicate exclusion | Before |
| 2026-11-09 | v1.2: S and B scripts and single-run markers | Before |
| 2026-11-11 | v1.3: 15 looksmaxxing claims and 18 tests | Before |
| 2026-11-11 | v1.4: failure rule, input hashes, B-script transfer criteria based on Model A, and summary criteria | Before |
| 2026-11-11 | S and B run once each | Final runs |
| 2026-11-11 | v1.5: corrected wording on raters, origin of the claims, specification of the post hoc sequence | After |
| 2026-11-11 | Post hoc sequence on held-out photographs | After |
| 2026-11-12 | v1.6: intervals for steps in the post hoc sequence | After |

**Use of MEBeauty fold 0 ratings.** On 7 November, fold 0 was included in the descriptive audit. On 11 November it was used to evaluate Models A and B. On 19 November, it was used to evaluate Model C, compare error against the evaluation-group mean, and calculate additional intervals. On 20 November it was used to compare Model A against a constant forecast using a scale and means fixed from folds 1–4. Figure 2 uses the 19 November results.

On 20 November, the intermediate values of pooled r under successive transformations were calculated from saved SCUT predictions. A separate comparison of the sheet-only and sheet-by-source baselines used SCUT development data.

**What had been seen or decided.**

1. The main model was chosen on development data after trying earlier feature sets. The headline figure in early records referred to a different model.
2. The within-group target and six groups were selected after the raw-score results showed differences between sources. The “80% of faces” criterion was withdrawn.
3. Fold 0 ratings were used in the 7 November audit. The exploratory transfer evaluation used folds 1–4. Final training excluded fold 0, and the final fold 0 evaluation occurred on 11 November.
4. Early analyses included some photographs later held out. Some descriptive summaries continued to include their ratings after the split. I saw and rated some held-out photographs. Section 2 describes the timing of feature development.
5. The claim topics took shape in discussion with an AI assistant. I proposed topics and additions. The assistant mapped them to measurable indicators and set expectations using development results. It showed the table of development results before I approved the list. The conversation shows when those numbers became available to me. It does not establish which numbers I read or considered.

### Recorded expectations and observed results

Expectations were recorded in script headers before the runs and used development results. Falling inside the resulting ranges is limited evidence. Misses are retained as recorded. See the original comparison in `2026-11-11_table3_registered_vs_observed.md`.

| Item | Recorded | Observed | Match |
|---|---|---|---|
| Macro r, 70 features | 0.66 | 0.678 | Yes |
| Macro r floor, below which the interpretation of explained variance would be weakened | ≥0.60 | 0.678 | Yes |
| ±0.08 band, 70 features | Six of six groups | Five of six (CF 10k_US higher by 0.091) | No |
| Geometry band | Six of six | Five of six (AM internet higher by 0.088) | No |
| Ridge below boosting | In every group | Lower by 0.01–0.04 where it was lower, higher in three | No |
| Effect of excluding near-duplicates | Small | +0.044 in AM internet | No |
| Population gap ≥0.10 | In Asian groups | In one of four | No |
| Variance partitioning by block | 18 items | Five missed (one by 0.001) | Partial |
| Model versus one rater | No support in CM and CF, possible support in DataTang | No support, including both DataTang groups | Partial |
| MEBeauty checks | Eight items | Seven met, all-groups criterion unmet | Partial |
| Looksmaxxing claims | 18 verdicts | 17 matched. “Masculine faces in men” was expected inconclusive and classified as contradicted | All but one |

*Table C2. For MEBeauty, A attained 0.337 (expected 0.28–0.38) and B attained 0.431 (expected 0.38–0.52). The average criterion and benefit from adding target-domain data (+0.094) were met. Ridge was below boosting, quality-flagged photographs performed worse, and adjusting for apparent age reduced Model B's mean correlation by 0.096. The recorded expectation was that small groups would likely prevent meeting the all-groups criterion, with an Asian group expected to be inconclusive. The inconclusive group was instead Caucasian male (n = 86), and Asian male (n = 29) had the highest r.*

## Appendix D. Exploratory analyses

Before opening the held-out ratings, I checked what additional feature blocks contributed beyond geometry, capture measures, and apparent age using development data (research log items 40–42, 54, 57, and 62–68).

- **Face shape, estimated subcutaneous fat, and expression:** improvements near zero in all six groups, possibly because the 42 geometric measurements already captured some of the same information.
- **Marks I listed** (sclera, under-eye region, eyelids, nose tip, skin texture, and brows): no reliable gain, with the inclusion threshold reached in one of six groups. The features are represented in the 28 detail measures. The main model was formed by removing proxies from the full feature list (capture, age, pose, expression, texture and lighting, color levels, and symmetry). It was not assembled by ranking gains. The research log identifies chin and lip features as possible proxies for capture conditions.
- **Landmark symmetry:** −0.017 to +0.003 on development data and a mean gain of +0.004 on held-out photographs.
- **Eight 2D angularity measures:** +0.003 for boosting and +0.009 [0.004, 0.015] for ridge. Their angles are redundant with existing features (jaw angle versus jaw_turn correlation −0.97).
- **Head rotation and tilt** from six landmarks: +0.0008 for boosting.
- **Reviewer-suggested features** (skin tone, local contrast, lash-line strip, creases, iris, and lip color): small gains, with +0.009 for the lash line in boosting and smaller gains for the remaining features. The nasolabial fold gave +0.0155 [0.010, 0.023] in ridge. Sex-specific typicality gave +0.004.
- **Seventy features plus twelve selected features:** +0.020 for boosting and +0.0385 for ridge. After adjustment for capture measures, the gains were −0.009 [−0.019, +0.003] for boosting and +0.012 [0.001, 0.027] for ridge. The same photographs were used to make the selection.

These results concern the listed measures and the precision attainable in these 350-pixel images. Landmark noise reduced r by about 10–20% in the development diagnostics (research log items 38 and 61).

**Additional Model B results on MEBeauty.** Mean boosting r was 0.495 after excluding quality-flagged images and 0.335 after adjustment for apparent age. Ridge averaged r = 0.423 on the full evaluation partition.

### Variance partitioning and image embeddings

**Variance partitioning (descriptive).** I fitted ridge by cross-validation separately within each group of held-out photographs. This is a new fit within those photographs. The commonality analysis partitioned R² among geometry G (42 features), capture measures Q (six), and apparent age A (one). It did not use the 28 detail features. Unique contributions were estimated over 30 random 80% subsamples. R² for each subset of blocks was estimated by cross-validation within the subsample (five folds, two repeats). Negative estimates can arise from sampling variation. The 2.5th–97.5th percentiles describe stability across subsamples.

**Image embeddings (development only).** A ridge model on full-image DINOv2-base embeddings (CLS and mean pooled patches, 1,536 columns, standardized within groups) was cross-validated on the development set. I also tested embeddings residualized on the six capture measures.

For geometry, the unique R² estimates ranged from about 0.05 (CF 10k_US, range includes zero) to 0.38 (AF DataTang). For capture measures they ranged from −0.056 to +0.051 and were above zero throughout the range only for AF internet. Apparent age reached 0.095 (CM 10k_US) and at most 0.04 elsewhere.

| Group | G, geometry | Q, capture | A, apparent age |
|---|---|---|---|
| CM 10k_US | 0.235 [0.149, 0.348] | −0.014 [−0.058, 0.051] | 0.095 [0.052, 0.152] |
| AM DataTang | 0.359 [0.323, 0.403] | 0.000 [−0.017, 0.020] | 0.008 [−0.003, 0.019] |
| AM internet | 0.357 [0.265, 0.424] | 0.020 [−0.012, 0.061] | −0.003 [−0.014, 0.011] |
| CF 10k_US | 0.053 [−0.085, 0.171] | −0.056 [−0.118, 0.008] | 0.039 [−0.010, 0.089] |
| AF internet | 0.176 [0.120, 0.240] | 0.051 [0.020, 0.085] | 0.008 [−0.004, 0.017] |
| AF DataTang | 0.378 [0.313, 0.452] | 0.036 [−0.003, 0.079] | 0.008 [−0.004, 0.018] |

*Table D1. Unique R² components. Entries show means and the 2.5th–97.5th percentile ranges across 30 subsamples. The ranges describe subsampling stability.*

Ridge on the DINOv2 embedding yielded R² = 0.635, compared with 0.483 for boosting on the 70 features. Adjusting the embedding for capture measures lowered R² to 0.473. Combining embedding and features yielded 0.671 without adjustment and 0.627 with adjustment. This shows sensitivity of embedding-based prediction to the adjustment. The capture measures may also be related to image content. These comparisons used the development set only.

## Appendix G. Additional checks

### Comparison with a single rater

For each rater j, I subtracted the correlation between that rater and the mean of the other 59 raters from the correlation between the model and the same reference. The differences were averaged across 60 raters. Intervals used a paired bootstrap over photographs. The model and raters were fixed, the model was trained on the mean from the same panel, individual ratings were integers, and the mean difference does not show how many raters the model surpassed.

The prespecified criterion for superiority was a lower bound above zero. Overall superiority required at least four of six groups. Equivalence was not prespecified. The six rater comparisons and twelve transfer-gap tests below use no multiplicity correction.

**Reliability of the mean.** In an exploratory development-set analysis, split-half reliability of the 60-rater mean after Spearman–Brown correction was 0.97–0.99. The estimated correlation ceiling for this panel was 0.986–0.995. By that estimate, noise in the mean was small relative to the gap between model correlation and one. The estimate pertains to these development photographs and this panel.

The intervals included zero and allowed an advantage in either direction in all six groups. Wang et al. (2025) also compared Face++ scores with human SCUT ratings, including individual raters.

| Group | n | Model r | Single-rater r | Difference [95%] |
|---|---|---|---|---|
| CM 10k_US | 148 | 0.576 | 0.585 | −0.009 [−0.117, +0.093] |
| AM DataTang | 328 | 0.619 | 0.607 | +0.012 [−0.060, +0.093] |
| AM internet | 145 | 0.809 | 0.782 | +0.027 [−0.034, +0.076] |
| CF 10k_US | 124 | 0.671 | 0.645 | +0.026 [−0.051, +0.096] |
| AF internet | 246 | 0.685 | 0.715 | −0.030 [−0.111, +0.032] |
| AF DataTang | 205 | 0.707 | 0.683 | +0.023 [−0.038, +0.089] |

*Table G1. Final H2 comparison, primary photographs. Both correlations use the same mean of the remaining 59 raters for each comparison and are then averaged over raters. Intervals for the difference use a paired bootstrap over photographs. The difference between the two rounded correlations may differ from the reported difference by 0.001.*

### Transfer between SCUT population and sex categories

For a target group T, three boosting models with 70 features were fitted on equal-sized development samples: the target group, another population category, and the same population category of a different sex. Predictions were averaged over 20 random subsamples. The population and sex gaps are correlations from the target-group model minus correlations from the corresponding comparison model. Intervals use paired resampling of photographs. The prespecified threshold was a gap of at least 0.10 with the interval above zero. Overall support required at least four of six groups.

The population gap met the criterion only in AF internet (+0.156 [0.087, 0.225]). It was +0.096 [0.031, 0.165] in AF DataTang and varied from −0.025 to +0.060 elsewhere. The gap was expected in four Asian groups and found in one. It had been 0.24 for Asian women in exploratory development analysis. The sex gap never reached 0.10 (range −0.043 to +0.085), although the intervals were above zero for AF internet and AF DataTang. The other-population training sample also differs in image source and sex composition, limiting interpretation.

| Group | Population gap | Sex gap |
|---|---|---|
| CM 10k_US | +0.044 [−0.019, 0.102] | +0.085 [−0.005, 0.169] |
| AM DataTang | +0.060 [−0.034, 0.151] | +0.052 [−0.031, 0.131] |
| AM internet | −0.009 [−0.059, 0.041] | +0.025 [−0.014, 0.067] |
| CF 10k_US | −0.025 [−0.111, 0.055] | −0.043 [−0.150, 0.070] |
| AF internet | +0.156 [0.087, 0.225] | +0.061 [0.008, 0.120] |
| AF DataTang | +0.096 [0.031, 0.165] | +0.077 [0.015, 0.139] |

*Table G2. Differences in held-out correlation between target-group training and alternative training. Intervals use paired resampling of photographs. The criteria were recorded in advance.*

### Looksmaxxing claims

For each claim I fixed one measure and direction before the final run (Appendix B). The test used within-group Spearman correlation with standardized ratings for the relevant sex (six groups for claims covering both sexes, three for a claim about one sex), and 1,000 photograph bootstrap resamples. For groups with n ≥ 100, support required a mean signed ρ of at least 0.15 and a lower bound above zero in two thirds of the groups. Contradiction required mean signed ρ no greater than −0.15 and an upper bound below zero in two thirds. Other results were inconclusive. Symmetry was tested as the ridge r gain from ten symmetry measures added to the 70 features. Support required mean gain of at least 0.01 and a lower bound above zero in two thirds of the groups, using paired resampling.

The rule does not control an error rate. The eighteen tests are dependent, no multiplicity correction was used, and the directions were visible in development data. Jaw measures are interrelated (absolute development Spearman correlations up to 0.82, Appendix E), so their results are considered together. The analysis did not adjust for subcutaneous fat, age, expression, or head rotation.

On the held-out images, positive canthal tilt and lower brows were associated with higher ratings. Several lower-face measures, including jaw width, showed associations opposite the stated directions. The incremental result for symmetry was inconclusive.

I contributed topics and additions. The assistant proposed the final wording and measures, and recorded expectations based on development results. I approved the list once the assistant had sent the table of those results and before the final run. The held-out check shows which associations repeated using the fixed measures.

| Direction | Image measure or test | Mean signed ρ |
|---|---|---:|
| As stated | Wider cheekbones relative to jaw | +0.314 |
| As stated | Positive canthal tilt | +0.229 |
| As stated | Smaller intercanthal distance relative to eye width | +0.299 |
| As stated | Lower eyebrows | +0.367 |
| As stated | “Feminine” face in women | +0.216 |
| As stated | Less compact outline (squared perimeter divided by area), men / women | +0.430 / +0.376 |
| Opposite | Less scleral show below iris: more scleral show associated with higher ratings* | −0.258 |
| Opposite | “Masculine” face in men | −0.221 |
| Opposite | Sharp jaw angle, men / women | −0.315 / −0.291 |
| Opposite | Wide jaw, men / women | −0.278 / −0.313 |
| Inconclusive | Long chin relative to upper lip | +0.129 |
| Inconclusive | Face width divided by height to hairline (a different definition from conventional fWHR) | −0.173 |
| Inconclusive | Equal vertical thirds | +0.152 |
| Inconclusive | Thick and dense eyebrows | +0.157 |
| No support | Symmetry: r gain over the 70 features | +0.004 |

*Table G3. The sign of ρ follows each claim. The symmetry row gives a change in r. The asterisk indicates a measure that may also reflect eye opening, which was not tested. Including all held-out photographs changed the verdicts for the two borderline claims on width-to-height and equal thirds.*

Jaw and outline measures are related. These photographs cannot separate the contribution of jaw structure from soft tissue and age. Eye-region measurements may also depend on expression and eye opening. Fourteen claims used Spearman correlation, three of them tested separately by sex, for 17 tests. Symmetry was the eighteenth. The ratio of intercanthal distance to eye width changes when either quantity changes.

**Symmetry and dimorphism.** Said and Todorov (2011) compared a multivariate model of synthetic faces with models based on typicality and dimorphism. Holzleitner et al. (2019) compared theory-driven measures, including asymmetry and dimorphism, with a face-space model using photographs of women. Here symmetry was tested as predictive gain over the 70 features, and dimorphism as the association of a particular composite index with ratings. The results pertain to those measurements and feature sets.

### Adjustments, features, and further comparisons

**Partial correlations.** I regressed ratings and predictions on apparent age, six capture measures, or both within each evaluation group, then averaged residual correlations across groups. Intervals for correlations and differences came from an earlier analysis with 1,000 paired bootstrap resamples. The model and raters were fixed.

**Group-mean benchmark on a broader development sample.** In a five-fold cross-validation on 4,124 SCUT development photographs, predicting the rating-sheet mean (sex × population category) gave pooled r = 0.132. Including image source within each sheet raised it to 0.657. Group means were calculated from the training portion of each fold. For one photograph whose group did not occur in the training portion, the forecast used the sheet mean. These values describe the given development sample and splits.

**Within-group adjustments.** Unadjusted macro r was 0.678. It was 0.656 after adjustment for apparent age, 0.621 after adjustment for capture, and 0.605 after adjustment for both. The paired difference from the unadjusted value was 0.073 [0.049, 0.099]. These are residual associations for the selected measures.

Local contrast and tone may depend on lighting, exposure, and makeup. These pixel features were outside the six-variable adjustment block. The unique R² components in Table D1 refer to the same narrow block of capture measures. Establishing the contribution of capture conditions would require repeated photographs of the same people under known conditions.

**Importance to the model.** Rao et al. (2026) analyzed DisFace3DNet components with matched ablations on identical splits. In the SCUT development analysis, jointly permuting the brow, eye, and lip contrast features or the chin and lip tone features yielded the two largest drops in R². They accounted for 18.8% and 11.5% of the total positive drop across groups ([Appendix F](importance_en.md)). These are sensitivities to permutation. They depend on feature grouping and redundancy and do not estimate the effect of lighting or makeup on ratings.

**Age adjustment in both datasets.** Mean within-group r on held-out SCUT changed from 0.678 to 0.656 after adjustment for apparent age. In exploratory transfer from SCUT to MEBeauty folds 1–4 it changed from 0.331 to 0.216 (`2026-11-09_review_mebeauty_capture_age.txt`). Groups, photographs, target scores, and standardization differ. The adjustment regression was fitted separately in each dataset. Differences in the size of the decline are descriptive. Testing their causes would require comparable age distributions, capture conditions, and rating procedures.

In earlier exploratory cross-validation on folds 1–4, boosting with 70 features averaged r = 0.472 when trained within MEBeauty and 0.331 when transferred from SCUT. The MEBeauty-only Model C estimate of 0.473 (§4.4) is close to the 0.472 on other evaluation photographs from MEBeauty. The analyses share a dataset and some training photographs.
