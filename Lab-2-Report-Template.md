# Lab 2 Report Template

Name: Jordan Jesayen

NetID: jxj190024

Use this template to create `Lab-2-Report.md`. Keep final code in `Lab-2-Assignment.py` and save referenced plots, tables, or other outputs under `artifacts/`.

## Problem 1: Data Exploration and KNN Risk Notes

**Dataset shape and target cleanup.** `adult.data` loads as 32,561 rows x 15
columns and `adult.test` as 16,281 rows x 15 columns. Two file quirks are
handled in `load_adult_frame`: the test file's first metadata line is skipped
with `skiprows=1`, and the trailing period on test labels (`>50K.`) is removed so
both splits use the standardized labels `<=50K` and `>50K`. Leading spaces are
removed with `skipinitialspace=True` and every string cell is stripped, so the
missing marker becomes a clean `?`.

**Missing values.** Missing markers appear only in three categorical columns:
`occupation` (~1,932), `workclass` (~1,931), and `native-country` (~621). The six
numeric columns have no missing markers. These counts are saved in
`artifacts/tables/data_quality_summary.csv`.

**Feature types.** Six continuous predictors (`age`, `fnlwgt`, `education-num`,
`capital-gain`, `capital-loss`, `hours-per-week`) and eight categorical
predictors. Cardinality ranges from 2 (`sex`) up to ~42 for `native-country`,
which is the high-cardinality column targeted for target encoding.

**Target distribution and class imbalance.** The development data is roughly
~71.8% `<=50K` and ~28.2% `>50K`; the test data is similar (~71.4% / ~28.6%).
This is meaningful class imbalance: a trivial "always predict `<=50K`" model
already scores ~0.72 accuracy, so accuracy alone is misleading and F1/recall on
the minority `>50K` class matter more. The target-distribution artifact is
`artifacts/figures/target_distribution.png`.

**Problem 1.3 - Why KNN is difficult here (risk notes).**
1. *Class imbalance.* With only ~28% positives, neighborhoods around a query are
   dominated by `<=50K` rows, so the majority vote is biased toward the majority
   class and minority recall suffers, especially at large `k`.
2. *High-cardinality categoricals.* One-hot encoding `native-country` (~42 levels)
   adds many sparse binary columns; in that high-dimensional, mostly-zero space,
   Euclidean distances become less discriminative (curse of dimensionality).
3. *Large numeric ranges.* `fnlwgt` (tens of thousands to ~1.5M) and
   `capital-gain` (0 to 99,999) dwarf features like `age` or `hours-per-week`, so
   without scaling they dominate the distance and effectively erase the other
   features.
4. *Runtime cost.* KNN does no real training; every prediction scans the full
   ~24K-row training matrix. Tuning `k` and the distance metric over many
   settings repeats that full scan many times, so runtime grows with both data
   size and the size of the search grid.

---

## Problem 2: Leakage-Safe Preprocessing and Feature Engineering

**Split sizes and class balance (Problem 2.1).** Development data is split 75/25
with stratification and `random_state=42`: ~24,420 training rows and ~8,141
validation rows, each keeping the ~28% `>50K` share. `adult.test` (16,281 rows)
is held out untouched for the single final evaluation.

**`fit_preprocessing` / `transform_preprocessing` design (Problem 2.2).**
`fit_preprocessing(train_df, y_train, country_encoding)` learns *every*
transformation from the training split only and stores it in a dictionary;
`transform_preprocessing(df, state, scale)` only applies that stored state. This
separation is what makes the workflow leakage-safe: validation and test rows are
never inspected during fitting.

**State stored in `preprocessing_state`:** the numeric/one-hot/target column
lists, per-column imputation values (training mode for categoricals, training
median for numerics), the fitted `OneHotEncoder` and its output column names, the
manual target-encoding map plus the fallback value, the assembled
`feature_columns` order, and the fitted `StandardScaler`.

**Missing values, encoding, unknowns, scaling.** Categorical `?` markers are
mapped to `NaN` and filled with the stored training mode; numerics are filled with
the stored training median. Low/moderate-cardinality categoricals are one-hot
encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` so
unseen validation/test levels are silently zero-filled instead of crashing.
`native-country` is target encoded with training-only group means of the label.
Unknown categories at transform time fall back to the global training positive
rate (~0.28). Finally a single `StandardScaler` (fit on the assembled training
matrix) standardizes all columns so no feature dominates the distance.

**One-hot vs target encoding for `native-country` (Problem 2.3).** Using a fixed
`KNN(n_neighbors=5)` on the validation split:

| strategy | feature count | val accuracy | val F1 | runtime |
|----------|--------------|-------------|--------|---------|
| one-hot  | ~105         | ~0.816      | ~0.626 | ~1.8 s  |
| target   | ~65          | ~0.837      | ~0.688 | ~1.5 s  |

Target encoding produces ~40 fewer columns, runs faster, and gives equal-or-better
validation accuracy and F1. The training-only target-encoded values for several
countries and the unknown-category fallback are saved in
`artifacts/tables/target_encoding_sample_values.csv`. The fallback is the global
training positive rate, which is the safest neutral estimate for a country never
seen during fitting.

**Leakage audit (Problem 2.4).** See `artifacts/tables/leakage_audit.csv`. The
**highest-risk** point is **target encoding**, because it uses the label `y` to
build the feature. If the means were computed over the full dataset, the
validation/test labels would leak directly into the features and inflate scores.
This is controlled by computing the group means only on `y_train`, storing the
map, and applying it (with the fixed fallback) to the other splits.

**Final feature-matrix summary (Problem 2.5).** See
`artifacts/tables/feature_matrix_summary.csv`. Train and validation matrices have
the same ~65 columns in the same order, contain no missing values, and are fully
numeric.

---

## Problem 3: KNN Mechanics From Scratch

**Euclidean distance implementation (Problem 3.1).**
`euclidean_distances_from_one(row, training_matrix)` broadcasts the single row
against the whole training matrix, squares the element-wise differences, sums
across the feature axis, and takes the square root — fully vectorized, no Python
loop over features. The nearest-5 example for one validation row is saved in
`artifacts/tables/nearest_neighbor_example.csv`.

**Manual KNN (Problem 3.2).** On a subset of 1,000 training rows and 100
validation rows with scaled features and `k=5`, the manual majority-vote
classifier scored roughly accuracy ~0.81 and F1 ~0.65. At least 12 manual
predictions with their true labels are in
`artifacts/tables/manual_knn_predictions.csv`.

**Manual vs scikit-learn (Problem 3.3).** With the same subset, `k`, and Euclidean
distance, the manual predictions matched `KNeighborsClassifier` on **100% (100/100)**
of validation rows, confirming the from-scratch logic is correct. (Tiny
differences are possible only when distance ties are broken differently.)

**Problem 3.4 - Manual KNN reflection.** Manual KNN becomes expensive because it
is "lazy": there is no fitted model, so every query recomputes distances to all
training rows, making cost scale with (queries x training rows x features). On the
full 24K-row split this becomes slow, which is exactly why the subset is capped at
1,000/100. The easiest pieces were the vectorized distance function and the
majority vote (`round(mean)` for binary labels). The parts that needed careful
inspection were keeping array shapes aligned (one row vs the full matrix),
confirming `argsort` returned the nearest and not the farthest neighbors, and
checking that label arrays were integer `{0,1}` so the vote averaged correctly.

---

## Problem 4: KNN Experiments and Runtime-Aware Model Selection

**Baseline scikit-learn KNN (Problem 4.1).** `KNN(k=5)` on the full scaled
training features scored roughly: accuracy ~0.837, precision ~0.748, recall
~0.638, F1 ~0.688 on validation. Accuracy sits above the ~0.72 majority baseline,
and recall ~0.64 shows the model recovers a solid majority of the minority `>50K`
class — better than the trivial classifier, but with clear room to improve on the
minority class.

**Distance diagnostics before/after scaling (Problem 4.2).** See
`artifacts/tables/distance_diagnostics.csv`. Before scaling, distances are
enormous and almost entirely driven by `fnlwgt`/`capital-gain`, with a
nearest/farthest ratio near ~0.0006 — neighbors are essentially indistinguishable.
After standardization the distances collapse to a small comparable range and the
nearest/farthest ratio rises sharply (to ~0.25), meaning neighbors become
meaningfully closer or farther. This is direct evidence that scaling is essential
for a distance-based model on this dataset.

**Encoding strategy choice (Problem 4.3).** Based on the Problem 2.3 validation
evidence (fewer features, lower runtime, equal-or-better F1), **target encoding**
for `native-country` is chosen for the rest of the lab. It keeps the feature space
compact, which is exactly what helps KNN avoid distance dilution in high
dimensions. The choice is made on validation only, never on test.

**Runtime-aware k / distance loop (Problem 4.4).** The loop sweeps 10 `k` values
(1, 3, 5, 7, 9, 11, 15, 21, 31, 51) across two distance settings (Euclidean
`p=2`, Manhattan `p=1`), recording validation accuracy, validation F1, fit time,
and predict time for all 20 experiments
(`artifacts/tables/knn_k_distance_results.csv`,
`artifacts/figures/knn_k_distance_validation.png`). Performance is poor and noisy
at `k=1`, rises quickly, and plateaus around `k≈11`; very large `k` slightly
smooths away minority-class signal. The selected configuration (highest
validation F1, tie-broken on accuracy then speed) was **k=11, Euclidean (p=2)**,
with validation F1 ~0.695. KNN fit time is negligible (it just stores the data);
the real cost is predict time, which is the tradeoff to weigh against marginal
accuracy gains.

**Final test evaluation (Problem 4.5).** With choices frozen
(target encoding, StandardScaler, k=11, Euclidean) and retrained on the full
training split, the single test run scored: accuracy ~0.843, precision ~0.780,
recall ~0.628, F1 ~0.696. These are very close to the validation numbers, which
indicates the model generalized and was not overfit to the validation split during
selection.

---

## Problem 5: Evaluation, Artifacts, and Reflection

**Confusion matrices (Problem 5.1).** Saved as
`artifacts/figures/confusion_matrix_validation.png` and
`artifacts/figures/confusion_matrix_test.png`. On the test set the counts were
about TN ~10,799, FP ~826, FN ~1,733, TP ~2,923. The large FN count relative to TP
shows where KNN struggles: it misses a meaningful slice of true high earners,
which is the recall gap created by class imbalance.

**Metrics under imbalance.** Accuracy (~0.84) looks strong but is inflated by the
easy majority class — the trivial baseline already reaches ~0.72. Precision
(~0.78) says most `>50K` predictions are correct; recall (~0.63) says many true
`>50K` cases are still missed; F1 (~0.70) is the balanced summary that exposes
this gap. This is exactly why accuracy alone is insufficient under imbalance: a
model can score high accuracy while badly under-serving the minority class.

**Required summary table (Problem 5.2).** See `artifacts/tables/final_summary.csv`,
which collects the baseline validation metrics, the encoding comparison, the
distance-diagnostic summary, the manual-vs-scikit-learn agreement, the
runtime-aware loop winner, and the final test metrics, each with its related
figure/table filename.

**Problem 5.3 - Final recommendation and reflection.**
1. *Recommended configuration:* target-encoded `native-country`, full
   `StandardScaler`, **k=11** with **Euclidean** distance — the best
   validation-F1 setting that also kept prediction runtime reasonable.
2. *Is KNN strong here?* It is a reasonable, interpretable baseline (test accuracy
   above the majority baseline, F1 ~0.70), but the recall gap on `>50K` and the
   per-prediction scan cost make it only moderate for this task; tree- or
   boosting-based models would likely do better and are explored in later labs.
3. *Preprocessing effects:* scaling was decisive (the distance diagnostics show
   features were meaningless without it), and target encoding kept the feature
   space compact, which helped KNN more than the sparse one-hot version.
4. *Leakage controlled:* all imputation values, encoder state, target-encoding
   means, and scaler statistics were learned on training data only and reused via
   `preprocessing_state`; the test set was transformed with frozen state and
   scored exactly once.
5. *Manually verified detail:* I confirmed the from-scratch KNN matched
   scikit-learn (100% agreement) on the shared subset before trusting the manual
   logic, and I inspected the before/after-scaling distance ratios to confirm
   scaling actually changed neighbor structure.
6. *Questions for later modules:* (a) Would class weighting, threshold tuning, or
   resampling raise minority-class recall without hurting precision? (b) How much
   would a tree-based model or approximate-nearest-neighbor index improve both
   accuracy and runtime over plain KNN?

---

## Workflow Reflection

I had to manually verify that the from-scratch KNN was correct before relying on
it: I compared its predictions to `KNeighborsClassifier` on the same subset, the
same `k`, and the same Euclidean metric, and confirmed they agreed on 100% of the
sampled validation rows. I checked for leakage by making `fit_preprocessing` the
only place any statistic is learned — imputation modes/medians, the one-hot
encoder, the target-encoding means, and the scaler are all fit on the training
split alone and stored in `preprocessing_state`, while `transform_preprocessing`
only applies that frozen state to validation and test. I also kept `adult.test`
untouched until every choice was frozen, so model selection could not see it.
About KNN itself, I learned that it is extremely sensitive to feature scale and
dimensionality: the distance diagnostics showed that without standardization the
distances were dominated by `fnlwgt` and `capital-gain` and were nearly useless,
and that one-hot encoding a 42-level column hurt both runtime and F1 relative to
target encoding. Its main strengths are simplicity and a training step that is
essentially free; its main weaknesses are slow per-query prediction and a tendency
to under-predict the minority class under imbalance. Two questions I would
investigate next: first, whether distance weighting (`weights='distance'`) or an
adjusted decision threshold improves minority-class recall; and second, how KNN
compares to decision trees and ensemble methods on both accuracy and runtime once
those tools are available in later labs.
