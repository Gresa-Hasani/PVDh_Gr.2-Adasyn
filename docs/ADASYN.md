# ADASYN – handling class imbalance in the business size class

```bash
python -m pip install -r requirements.txt       # adds imbalanced-learn
python -m part1_preprocessing                   # creates data/processed/arbk_clean.csv (Part I, unchanged)
python -m adasyn_balancing                      # ~10 minutes (incl. cross-validation); writes outputs/adasyn/
python -m unittest discover -s tests -v         # 13 unit tests, no data needed
```

All numbers below come from `outputs/adasyn/` and were produced by this code. Two complete runs gave identical results.

---

## 1. Original project structure

| Item | Repository before this work |
|---|---|
| Pipeline | `part1_preprocessing/` (14 steps) and `common/` (paths, column names, helpers) |
| Data | `arbk_clean.csv`: 271,948 unique businesses × 47 columns (output of Part I) |
| Supervised target | None for modelling. `is_active` is used only for mutual information (s13), and it is almost balanced (51.7 / 48.3%). |
| Models, train/test split, metrics | None |
| Imbalance handling | None. The 10% stratified sample (s12) keeps proportions; it does not rebalance. |

ADASYN needs a classification task with real imbalance, so the target is the EU size class **`Madhësia`**,
with all 4 classes kept: 0-9 (98.45%), 10-49 (1.32%), 50-249 (0.20%), 250- (0.03%).

The size class is computed from the employee count. Every employee-based column (`Nr. punëtorëve`,
`log_employees`, `flag_employees_imputed`) is therefore **excluded** from the predictors. Part I is not modified.

## 2. Methodology

| Step | Module | Content |
|---|---|---|
| A1 | `s01_dataset` | Target, 48 features (2 continuous, 5 integer, 3 binary, 38 one-hot from legal form, status, NACE section, region), split, preprocessing |
| A2 | `s02_adasyn` | ADASYN on training data, followed by the repair of the synthetic rows |
| A6 | `s06_ratio_selection` | Choice of a milder ADASYN ratio by cross-validation **inside the training set** |
| A5 | `s05_validation` | 13 checks for every ADASYN training set. The run stops if any check fails. |
| A3 | `s03_models` | Logistic Regression, Decision Tree, Random Forest. Same hyper-parameters, no `class_weight`, same test set. |
| A4 | `s04_report` | Writes the tables and figures in `outputs/adasyn/` |

### ADASYN

ADASYN (He et al., 2008) gives each minority sample a "difficulty": the share of its k nearest neighbours that
belong to another class. Harder samples receive proportionally more synthetic points. Each synthetic point is a
linear interpolation towards a random neighbour of the same class.

| Setting | Value |
|---|---|
| Implementation | `imblearn.over_sampling.ADASYN`, called once per minority class with `sampling_strategy={class: target}`. This is the same computation as one multi-class call, because imblearn also processes each class against the original data. A class that cannot be resampled is skipped and logged. |
| `n_neighbors` | 5, reduced to `n_class − 1` for small classes. A class with fewer than 2 rows is left unchanged. |
| `target_ratio` | Each minority class is grown to `ratio × majority count` (1.0 = full balance) |
| Distance space | Numeric columns are standardised (fitted on train). Binary and one-hot columns stay 0/1 (see §8). |
| Repair of synthetic rows | One-hot → the largest value becomes 1. Binary → rounded to 0/1. Integers → rounded. Values are clipped to the training range. Registration year and month are rounded **jointly**, and the age is capped (see §8). `owners_total` and `has_female_owner` are recomputed. |
| `random_state` | 42 everywhere |

Original training rows are kept unchanged and first in the resampled set. Synthetic rows are appended after them.

### Ratio selection (no test data)

- **Cross-validation:** 3-fold stratified CV of the **training set** (seed 42).
- **ADASYN inside each fold:** it is fitted on the training folds only, and the validation fold is never resampled.
- **Grid:** ratios {0, 0.05, 0.10, 0.25, 0.50, 1.0}, where 0 means no ADASYN.
- **Criterion:** mean **macro F1**. Balanced accuracy would always pick the most aggressive ratio, because it
  ignores precision.
- **Choice:** the best ratio above 0, with ties going to the smaller ratio.
- **Compared on the test set:** **Baseline**, **ADASYN full (1.0)** (kept as the reference experiment) and
  **ADASYN CV-selected**.

## 3. Train/test split and leakage prevention

1. The **raw** rows are split 70/30, stratified by the target (seed 42), before any preprocessing.
2. Imputation medians, category lists and the ADASYN scaler are fitted on training rows only. A category unseen
   in training becomes an all-zero one-hot row in the test set.
3. ADASYN receives only `X_train`. In cross-validation it receives only the training folds.
4. The test set (81,585 rows) is never resampled and is never used to choose anything.

These are verified automatically for every ADASYN training set (`06_validation_checks.csv`, 13 checks, all
passed):
- train and test rows are disjoint
- every class is present in both sets
- the test class counts are unchanged
- the original training rows are unchanged
- synthetic rows belong only to minority classes
- no employee-based feature is used
- the columns are identical in the train, test and ADASYN matrices
- there are no missing values
- one-hot groups are valid
- binary and integer columns contain whole numbers
- `owners_total = M + F`
- **synthetic age ≤ time since registration**
- informational: overlap of synthetic vectors with test vectors (the features are mostly discrete, so equal
  vectors also occur among original training rows)

## 4. Class distributions (training set; the test set is shown for comparison)

| Class | Original train | ADASYN 0.05 | ADASYN 0.10 | ADASYN 1.0 | Test (never resampled) |
|---|---|---|---|---|---|
| 0-9 | 187,408 (98.45%) | 187,408 (87.1%) | 187,408 (77.0%) | 187,408 (25.0%) | 80,318 (98.45%) |
| 10-49 | 2,505 (1.32%) | 8,989 (4.2%) | 18,661 (7.7%) | 187,631 (25.0%) | 1,074 (1.32%) |
| 50-249 | 384 (0.20%) | 9,442 (4.4%) | 18,710 (7.7%) | 187,483 (25.0%) | 165 (0.20%) |
| 250- | 66 (0.04%) | 9,361 (4.4%) | 18,725 (7.7%) | 187,393 (25.0%) | 28 (0.03%) |

**Every count above the "Original train" column is synthetic.** These are interpolated points, not observed
businesses. At ratio 1.0, class 250- consists of 66 real businesses and 187,327 synthetic points (a factor of
about 2,840). Even at ratio 0.05, the 66 businesses produce 9,295 synthetic points (a factor of about 140).

## 5. Results on the test set (81,585 rows, real distribution)

### Overall

| Model | Training set | Ratio | Accuracy | Balanced acc. | Precision (macro) | F1 (macro) | F1 (weighted) | ROC-AUC macro / weighted |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Baseline | – | 0.984 | 0.269 | 0.463 | 0.283 | 0.977 | 0.880 / 0.834 |
| | ADASYN full | 1.0 | 0.700 | **0.496** | 0.260 | 0.228 | 0.813 | 0.823 / 0.829 |
| | ADASYN CV | 0.05 | 0.975 | 0.397 | 0.351 | 0.280 | 0.973 | 0.875 / 0.830 |
| Decision Tree | Baseline | – | 0.985 | 0.301 | 0.547 | **0.332** | 0.978 | 0.717 / 0.750 |
| | ADASYN full | 1.0 | 0.704 | **0.428** | 0.259 | 0.227 | 0.815 | 0.705 / 0.773 |
| | ADASYN CV | 0.05 | 0.974 | 0.337 | 0.301 | 0.286 | 0.974 | 0.714 / 0.787 |
| Random Forest | Baseline | – | 0.985 | 0.253 | 0.472 | 0.255 | 0.977 | 0.868 / 0.825 |
| | ADASYN full | 1.0 | 0.891 | **0.378** | 0.269 | 0.273 | 0.930 | 0.845 / 0.818 |
| | ADASYN CV | 0.10 | 0.975 | 0.368 | 0.317 | **0.317** | 0.974 | 0.868 / 0.829 |

### Minority classes: recall and precision (Baseline → ADASYN full → ADASYN CV)

| Model | Class | Recall | Precision |
|---|---|---|---|
| Logistic Regression | 10-49 | 0.006 → 0.409 → 0.001 | 0.368 → 0.024 → 0.333 |
| | 50-249 | 0.000 → 0.442 → 0.170 | 0.000 → 0.014 → 0.063 |
| | 250- | 0.071 → 0.429 → 0.429 | 0.500 → 0.008 → 0.020 |
| Decision Tree | 10-49 | 0.022 → 0.438 → 0.034 | 0.387 → 0.025 → 0.151 |
| | 50-249 | 0.109 → 0.279 → 0.146 | 0.529 → 0.011 → 0.052 |
| | 250- | 0.071 → 0.286 → 0.179 | 0.286 → 0.005 → 0.012 |
| Random Forest | 10-49 | 0.007 → 0.303 → 0.080 | 0.571 → 0.047 → 0.170 |
| | 50-249 | 0.006 → 0.236 → 0.224 | 0.333 → 0.029 → 0.077 |
| | 250- | 0.000 → 0.071 → 0.179 | 0.000 → 0.009 → 0.033 |

"0-9" businesses wrongly predicted as a larger class (false alarms):

| Model | Baseline | ADASYN full | ADASYN CV |
|---|---|---|---|
| Logistic Regression | 8 | 23,761 | 799 |
| Decision Tree | 41 | 23,434 | 889 |
| Random Forest | 6 | 7,976 | 902 |

The full confusion matrices are in `05_confusion_matrices.csv` and `fig04`.

### Interpretation: four separate questions

1. **Minority-class recall: improved.** Every ADASYN setting raises balanced accuracy (= macro recall) above the
   baseline, by +0.04 to +0.23. The baseline predicts "0-9" for almost every business. Full balancing finds
   28–44% of the 10-49 and 50-249 businesses with Logistic Regression and the Decision Tree.
2. **Precision: a strong trade-off.** The extra recall is paid for with false alarms. Full balancing produces
   7,976–23,761 of them, and minority precision falls to 0.5–5%. With the CV-selected ratio there are about
   800–900 false alarms. Precision then stays low (1–33%) but is 2–14 times higher than with full balancing.
3. **Overall performance: does not improve in general.**
   - Accuracy and weighted F1 decrease in every ADASYN setting, slightly with the CV ratio and strongly with
     full balancing.
   - Macro F1 improves only for **Random Forest with the CV ratio** (0.255 → 0.317). It is unchanged for
     Logistic Regression (0.283 → 0.280) and lower for the Decision Tree (0.332 → 0.286).
   - ROC-AUC, which measures how well the model ranks businesses and does not depend on a threshold, never
     improves by more than 0.04 and usually decreases.
   - ADASYN mainly **moves the decision boundary** towards the small classes; it adds no new information.
4. **Limits caused by the few real minority businesses.** The test set has only 28 businesses in "250-", so
   one business equals 3.6 percentage points of recall. For example, 0.429 means 12 of 28 businesses, and
   0.179 means 5 of 28. Training has 66 real businesses in this class. Every synthetic "250-" point is a
   combination of two of these 66, so the model learns their neighbourhood, not the population of large
   businesses.

**Conclusion.** ADASYN is useful only if finding larger businesses matters more than false alarms, for example
when pre-selecting businesses for a manual check. In that case a mild ratio (0.05–0.10) is clearly preferable
to full balancing: it keeps 97.5% accuracy and still raises balanced accuracy. Full balancing (1.0) is too
aggressive for this data.

## 6. Sensitivity analysis (3-fold CV inside the training set, `07_cv_ratio_sensitivity.csv`, `fig05`)

| Ratio | LR macro F1 | DT macro F1 | RF macro F1 | LR balanced acc. | RF accuracy |
|---|---|---|---|---|---|
| 0 (no ADASYN) | 0.259 ± 0.009 | 0.292 ± 0.013 | 0.252 ± 0.003 | 0.256 | 0.984 |
| 0.05 | **0.277** ± 0.005 | **0.292** ± 0.005 | 0.307 ± 0.009 | 0.393 | 0.980 |
| 0.10 | 0.273 ± 0.004 | 0.291 ± 0.007 | **0.310** ± 0.011 | 0.419 | 0.973 |
| 0.25 | 0.273 ± 0.003 | 0.279 ± 0.003 | 0.298 ± 0.003 | 0.458 | 0.951 |
| 0.50 | 0.259 ± 0.001 | 0.253 ± 0.003 | 0.288 ± 0.005 | 0.483 | 0.924 |
| 1.0 | 0.228 ± 0.000 | 0.228 ± 0.003 | 0.279 ± 0.007 | 0.500 | 0.890 |

(± = standard deviation over the 3 folds; bold = selected ratio)

- **The pattern is the same for every model.** Balanced accuracy rises steadily with the ratio, while macro F1
  peaks at a small ratio (0.05–0.10) and then falls. Among the ADASYN ratios, full balancing has the lowest macro F1 for every model.
- **The Decision Tree gains nothing.** In CV it shows no gain from ADASYN (0.292 with and without), and on the
  test set its macro F1 is lower. Its "selected" ratio is only the best ratio above 0.
- **Limitations of this analysis:**
  - Only 3 folds and one grid are used, and each fold holds only about 22 businesses of "250-".
  - The differences between 0.05 and 0.10 are within one standard deviation.
  - The selection depends on the criterion: balanced accuracy would choose 1.0.
  - The ratio is chosen per model, without nested CV, so the CV score of the selected ratio is slightly
    optimistic. The test results are not affected, because the test set played no part in the selection.

## 7. Independent audit findings

| Area | Finding | Status after corrections |
|---|---|---|
| ADASYN only on training data; per-class calls equivalent to imblearn's multi-class call; adaptive weights preserved | PASS | PASS |
| Scaling / inverse scaling consistent | PASS (error 6e-14) | PASS (tested) |
| Rare one-hot columns dominated distances (value 1 scaled up to 308) | WARNING | **Corrected** (§8.1) |
| Synthetic ages larger than the time since registration (16,724 rows, 3.0%) | WARNING | **Corrected**: 0 violations (§8.2) |
| Full balancing excessively aggressive (66 → about 187,000) | WARNING | **Addressed**: milder ratios evaluated and chosen by CV on train; 1.0 kept only as a reference. The limitation remains (§9). |
| Synthetic data shrinks spread and heavy tails | WARNING | Remains (§9) |
| `businesses_per_1000_municipality` computed in Part I from all rows | WARNING (minor) | Remains, documented (§9) |
| Train/test separation, train-only preprocessing, no employee features, no test data in ADASYN | PASS | PASS |
| Identical evaluation conditions; ROC-AUC matches a manual one-vs-rest calculation; Logistic Regression converges | PASS | PASS |

**Quality of the synthetic "250-" data after the corrections** (ratio 1.0, compared with the 66 real businesses):

- **Distributions:** every single-feature KS test has p ≥ 0.83. Category shares stay close (for example,
  Prishtinë 0.58 real vs 0.60 synthetic).
- **Constraints:** 0 impossible ages, all one-hot groups valid, and `owners_total = M + F` holds.
- **Spread shrinks:** the standard deviation of owner counts falls from 50.1 to 28.5. Interpolation pulls
  values inwards.
- **Correlations:** the largest difference is 0.27 (age vs registration month). With 66 real rows, the real
  correlation itself is uncertain (standard error about 0.12).
- **Duplicates:** 5,387 synthetic rows (2.9%) are exact copies of a real "250-" business, and only 149,542 of
  the 187,327 synthetic vectors are distinct.
- **Distance to the real data:** the median distance from a synthetic point to the nearest real "250-" business
  is 0.91, compared with 4.38 between real businesses (both measured on the same fully standardised scale).
  Synthetic points stay close to the 66 originals.
- **Adaptivity:** 54 of the 66 businesses have difficulty 1.0, so ADASYN behaves almost like SMOTE for this
  class. Each real business produces 1,201–3,002 synthetic points.

## 8. Corrections made to synthetic data generation

### 8.1 Distance scaling (`s02_adasyn.NumericScaler`)
- **Before:** every column was standardised. A one-hot level with 2 training rows became 308 after scaling and
  dominated every k-NN distance. This affected 5 of the 66 "250-" businesses.
- **After:** only the 7 numeric columns are standardised. The 41 binary and one-hot columns stay 0/1, so a
  category mismatch adds the same distance whatever the category's frequency.
- **Consistency:** the scaling is affine and works column by column. The inverse is exact, and interpolating in
  the scaled space and inverting gives the same point as interpolating in the original space. Both properties
  are covered by unit tests.

### 8.2 Age and registration date (`s02_adasyn._repair_registration_date`)
- **The constraint:** Part I computes the age as (end − registration date) / 365.25, with the end at or before
  the reference date 05.10.2026. So every real business satisfies **age ≤ time since registration**. This holds
  within Part I's 2-decimal rounding: the largest excess is 0.0046.
- **What actually broke it:** the constraint is linear, so a linear interpolation between two real businesses
  satisfies it as well. Interpolation was **not** the cause. The violations came from the repair, which rounded
  the year and the month **separately**. For example, year 2005.5 and month 6.5 became 2006/June, almost a
  year after the interpolated date of about 2005-12.
- **Fix:**
  1. Year and month are combined into one month index, rounded once and split again, so the date moves by at
     most half a month.
  2. The age is capped at the time between that registration date and the reference date.
- **Result:** the number of violations went from 16,724 to 0. A validation check and two unit tests cover this.
  One of the tests fails when run against the old behaviour.

### Effect on the full-balancing results (before → after the corrections)
| Model | Balanced accuracy | Macro F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.478 → 0.496 | 0.229 → 0.228 | 0.707 → 0.700 |
| Decision Tree | 0.448 → 0.428 | 0.234 → 0.227 | 0.727 → 0.704 |
| Random Forest | 0.370 → 0.378 | 0.282 → 0.273 | 0.915 → 0.891 |

The changes are small, and the earlier conclusions about full balancing still hold.

## 9. Remaining limitations and warnings

- **Synthetic businesses are not real observations.** At ratio 1.0, class 250- is 99.96% synthetic. Every
  synthetic point lies on a line between two of 66 real businesses, so the model learns about 66
  neighbourhoods, not the population of large businesses. Some synthetic points are exact copies of real
  businesses. Results for "250-" therefore rest on 66 training and 28 test businesses, whatever the number of
  synthetic rows.
- **Spread and correlations are distorted.** Interpolation shrinks the spread (heavy-tailed owner counts most of
  all) and can change correlations between features.
- **Weak features.** Without employee counts, the features separate business size only weakly (baseline
  ROC-AUC 0.72–0.88). No oversampling method can add the missing signal.
- **Statistical uncertainty.** "250-" recall rests on 28 test businesses. The CV differences between nearby
  ratios are within one standard deviation.
- **No tuning.** Hyper-parameters are fixed and there is no nested CV. This is sufficient for a fair comparison
  of the settings, but not for a final production model.
- **A Part I feature uses all rows.** `businesses_per_1000_municipality` is a municipality-level count over all
  rows, test rows included, that does not involve the target. Its effect is negligible, and Part I is
  intentionally left unchanged.
- **Synthetic vectors equal to test rows.** Some synthetic vectors equal a test row. This is informational: the
  features are mostly discrete, ADASYN never sees the test set, and such equalities are also frequent among
  original training rows.
