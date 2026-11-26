# Table 3 for the paper: registered expectation against observed result (compiled 2026-11-11 [AI] from the S and B headers, results/S_lockbox_run.txt, results/mebeauty_B_run.txt; no new analysis)

Expectations are those written in the headers of scripts/scut_lockbox_S.py and scripts/mebeauty_B.py before the runs (S header governs, FREEZE v1.4 p.6). 'Hit' means inside the registered range or band; edge cases are stated as such. All observed numbers are PRIMARY (no twins) unless said.

## S, H1: r inside set, tier 2 boosting (band: expectation +-0.08)
| Set | Expected | Observed | Result |
|---|---|---|---|
| CM 10k_US | 0.60 | 0.576 | hit |
| AM DataTang | 0.61 | 0.620 | hit |
| AM internet | 0.76 | 0.809 | hit (+0.049) |
| CF 10k_US | 0.58 | 0.671 | MISS (+0.091) |
| AF internet | 0.73 | 0.685 | hit |
| AF DataTang | 0.67 | 0.707 | hit |
| Mean of six | 0.66; downgrade below 0.60 | 0.678 | rule not triggered |

## S, geometry boosting (band +-0.08; H1 band 0.40-0.70 per set)
| Set | Expected | Observed | Result |
|---|---|---|---|
| CM | 0.56 | 0.549 | hit |
| AM DataTang | 0.54 | 0.618 | hit, at the edge (+0.078) |
| AM internet | 0.71 | 0.798 | MISS (+0.088); above 0.70 |
| CF | 0.52 | 0.590 | hit (+0.070) |
| AF internet | 0.67 | 0.653 | hit; inside 0.40-0.70 (risk of 'above' did not materialise) |
| AF DataTang | 0.59 | 0.649 | hit |
H1 count (v1.4): 5 sets inside the band, 1 above (AM internet), 0 below. Expected: four inside, AM internet and AF internet at risk of 'above': partly (one of two).

## S, other registered statements
| Statement | Result |
|---|---|
| Ridge 0.01-0.04 below boosting in every set (tier 2) | MISS: ridge above boosting in CM (+0.044), AM DataTang (+0.021), AF DataTang (+0.028); below in AM internet (-0.006), CF (-0.059), AF internet (-0.035) |
| Removing the 50 twins changes r by -0.03 to +0.01 in a set | partly: AM internet +0.044 (PRIMARY 0.809 vs SECONDARY 0.765) is outside, AF DataTang +0.013 at the edge, the other four inside |
| H2 not supported in CM and CF; possibly supported in the DataTang sets | hit for CM, CF; 'possibly' did not materialise (0 of 6; differences -0.030 to +0.027) |
| H3 population gap >= 0.10 in the Asian sets, sex gap < 0.10 | population gap supported in 1 of 4 Asian sets (AF internet +0.156; AF DataTang +0.096 just below; AM DataTang +0.060; AM internet -0.009): MOSTLY MISS; sex gap < 0.10 in all six: hit |
| H5 unique G >= 0.25 in AM internet, AF internet, AF DataTang | AM internet 0.357 hit; AF DataTang 0.378 hit; AF internet 0.176 MISS |
| H5 unique G about 0.00-0.15 in CM, CF, AM DataTang | CF 0.053 hit; CM 0.235 MISS; AM DataTang 0.359 MISS |
| H5 unique Q <= 0.05 everywhere | hit in five sets; AF internet 0.051 at the edge (miss by 0.001) |
| H5 unique A <= 0.05 except CM and CF (0.08-0.11 expected) | CM 0.095 hit; CF 0.039 MISS; the other four hit |

## B (MEBeauty fold 0)
| Statement | Expected | Observed | Result |
|---|---|---|---|
| A (SCUT-only boosting), mean of four primary | 0.28-0.38 | 0.337 | hit |
| B (pooled training) boosting, mean of four | 0.38-0.52 | 0.431 | hit |
| 'transfers' (read from A_boosting, v1.4) | met | not met: Caucasian men [-0.022, 0.345] 'not decidable' | MISS (the power note expected the undecided group among the Asian groups; the Asian men were the strongest, 0.678 on 29 faces) |
| 'useful' (mean of four >= 0.30) | met | 0.337 | hit |
| 'pooling helps' (+0.03 to +0.12, lower bound > 0) | met | +0.094 [0.034, 0.146] | hit |
| Ridge below boosting by 0.00-0.05 | yes | B 0.423 vs 0.431; A 0.321 vs 0.337 | hit |
| Flagged faces fit worse than the rest | yes | B boosting without flagged faces 0.495 vs 0.431 | hit |
| Partial given perceived age 0.08-0.15 below the plain value | yes | B boosting 0.335 vs 0.431 (0.096) | hit |
| Claim 71 downgrade (any 'failed' group or mean < 0.25) | not expected | no group 'failed'; mean 0.337 | not downgraded |

## L block (18 tests, PRIMARY verdict against the header list)
| Expected | Tests | Result |
|---|---|---|
| SUPPORTED | C02, C03, C04, C08, C12, C14M, C14F | 7 of 7 hit |
| CONTRADICTED | C05, C13M, C13F, C15M, C15F | 5 of 5 hit |
| not distinguishable | C01, C06, C07, C09, C11 | 4 of 5 hit; C11 MISS (CONTRADICTED, mean -0.221) |
| not supported | C10 | hit (mean gain +0.004) |
Total 17 of 18 hit. SECONDARY (with twins) differs on two edge claims: C06 CONTRADICTED, C07 SUPPORTED.

## Not a registered expectation (for the text, not for this table)
Post hoc held-out ladder (FREEZE v1.5): 0.823 -> 0.678 -> 0.605, expectation hit; status POST HOC.

## Correction note
An earlier summary (STATE entry of 2026-11-11, claim 74, the page card, BRIEF4) called the ridge/boosting order in B a miss. That came from the fold-4 smoke; in the fold-0 run ridge is below boosting, as expected. Corrected in STATE, RESEARCH, page and BRIEF4 on 2026-11-11.
