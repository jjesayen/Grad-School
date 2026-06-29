# Lab 3 Report

Name: Jordan Jesayen

NetID: jxj190024 

Final code lives in `Lab-3-Assignment.py`. All referenced plots and tables are saved under `artifacts/figures/` and `artifacts/tables/`. All numbers below come from a single run with `random_state=42` on `datasets/Car_Prices_Poland.csv`.

Problem 1: Data Exploration and Regression Framing

The raw CSV is `(117927, 11)`. The first column reads in as the index-like `Unnamed: 0` and is dropped. After also removing 6,473 exact duplicate listings that surface once that column is gone, the cleaned frame is `(111454, 10)`: nine predictors plus the `price` target. The only column with missing values is `generation_name` (28,658 missing after de-duplication); all other columns are complete. Numeric predictors are `year`, `mileage`, and `vol_engine`; categorical predictors are `mark`, `model`, `generation_name`, `fuel`, `city`, and `province`.

`price` is strongly right-skewed (skew ≈ 3.81). The mean is about 68,873 PLN while the median is only 41,500 PLN, and the maximum reaches 2,399,900 PLN. The raw histogram piles almost everything against the low end with a long thin tail of expensive cars, and the log-scale panel becomes roughly bell-shaped (see `artifacts/figures/price_distribution.png`). Those extreme high-value listings are real outliers that dominate squared-error metrics. Cardinality (`artifacts/tables/categorical_cardinality.csv`) is very uneven: `city` has 4,427 distinct values, `generation_name` 364, and `model` 328, while `mark`, `province`, and `fuel` have only 23, 23, and 6. The three high-cardinality columns are the central modeling challenge here.

Several properties make plain linear regression difficult. First, the heavy target skew means a squared-error model is pulled hard toward a small number of luxury listings, inflating RMSE relative to MAE. Second, the high-cardinality categoricals (`city`, `model`, `generation_name`) cannot be naively one-hot encoded without exploding the feature space and overfitting sparse levels. Third, the outliers in both `price` and `mileage` (mileage runs up to ~2.8 million km) give high-leverage rows that distort coefficients. Fourth, the true price surface is non-linear: depreciation with `year` and `mileage` is curved, not straight, so a strictly additive linear model leaves structure on the table. The mean-target baseline I use after splitting is the training mean (≈ 69,023 PLN), which scores validation MAE ≈ 52,046, RMSE ≈ 80,494, and R² ≈ 0. A baseline matters because it sets the "do nothing" reference; any real model has to beat predicting the average price, and R² is literally defined against this baseline.

Problem 2: Leakage-Safe Preprocessing and Feature Engineering

Splits.** I separate `X` and `y`, carve off a 20% final test split first, then split the remaining development data 80/20 into train and validation. The result is 71,330 train rows (64%), 17,833 validation rows (16%), and 22,291 test rows (20%), all with `random_state=42`. No preprocessing object is fit before this split.

`fit_preprocessing` / `transform_preprocessing` design.** `fit_preprocessing(train_df, y_train)` learns every transformation from training rows only and returns a `preprocessing_state` dictionary. `transform_preprocessing(df, state)` applies that state to any split and returns a numeric DataFrame with identical columns in identical order. The stored state is: `numeric_medians` (training medians for safe imputation), `ref_year` (max training year, used for `car_age`), `global_mean_price` and `mark_price_mean` (training target aggregations), `freq_maps` (training frequency proportions for the high-cardinality columns), a fitted `OneHotEncoder` plus its `ohe_feature_names`, a fitted `StandardScaler`, the `numeric_columns` list, and the final `feature_order`.

Missing values are handled with training medians for numerics and a `__MISSING__` sentinel category for categoricals (this is how the 28,658 missing `generation_name` rows are absorbed). Low-cardinality columns (`mark`, `fuel`, `province`) use `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. High-cardinality columns (`model`, `generation_name`, `city`) use training frequency encoding, with unknown validation/test categories falling back to 0. The `mark` aggregation falls back to the global training mean for unseen makes. Engineered features (below) are computed inside the shared `_build_numeric_block` helper so fit and transform stay identical. Only the continuous/engineered block is standardized; the 0/1 one-hot indicators are left unscaled. The final matrix has 62 features.

High-cardinality strategy comparison** (`artifacts/tables/high_cardinality_comparison.csv`), evaluated on `model` with a small linear model on top of the three raw numerics:

| strategy | n_features | val MAE | val RMSE | runtime (s) | unknown fallback |
|---|---|---|---|---|---|
| frequency encoding | 4 | 32,052 | 53,028 | 0.01 | 0.0 |
| rare-group + one-hot | 239 | 26,640 | 44,859 | 1.25 | `__RARE__` bucket / all-zero |
| smoothed target encoding | 4 | 26,438 | 42,369 | 0.04 | global mean ≈ 69,023 |

Smoothed target encoding wins on both MAE and RMSE while keeping a single feature and tiny runtime; rare-group one-hot is close but needs 239 columns and ~30× the time; frequency encoding is cheapest but weakest because frequency is only loosely tied to price. The unknown-category fallback is the key safety mechanism: frequency maps return 0 for unseen levels, target encoding returns the global training mean, and the one-hot encoder emits an all-zero row, so a never-before-seen `model`, `city`, or `mark` cannot crash transformation. For target encoding I controlled small-group overfitting with smoothing `(count·group_mean + 10·global_mean) / (count + 10)`, which pulls thinly populated groups back toward the global mean.

Feature engineering** (`artifacts/tables/feature_engineering_impact.csv`). I added six training-only features: `car_age = ref_year − year` (transformation), `log_mileage = log1p(mileage)` (transformation), `engine_liters = vol_engine/1000`, `age_x_liters = car_age × engine_liters` (interaction), `mileage_per_year` (derived ratio), and `mark_price_mean` (training mean price by make with a global fallback — the statistical aggregation). Adding these moved validation MAE from 29,072 to 24,419 and validation RMSE from 47,345 to 41,689, a clear and consistent improvement.

Leakage audit** (`artifacts/tables/leakage_audit.csv`) covers all eight required steps with `step`, `fit_on`, `applied_to`, `stored_state`, `leakage_risk`, and `how_risk_was_controlled` columns. The highest-risk point is the engineered statistical features: `mark_price_mean` is derived directly from the target, so computing it on anything other than `y_train` would leak future prices into the feature matrix. It is computed from training rows only, with a global-mean fallback for unseen makes. The final feature-matrix summary (`artifacts/tables/feature_matrix_shapes.csv`) confirms train/validation/test are `(71330, 62)`, `(17833, 62)`, `(22291, 62)`, with identical column order across all three and zero missing values after transformation.

Problem 3: Linear Regression Mechanics From Scratch

I fit a small `LinearRegression` on the first 500 training rows using three standardized predictors (`car_age`, `log_mileage`, `engine_liters`), giving the equation

```
price_hat = 70327.60 − 42875.95·car_age_z − 34720.21·log_mileage_z + 44555.51·engine_liters_z
```

The intercept (≈ 70,328) is the predicted price when all three standardized inputs are at their training mean (z = 0); each coefficient is the price change per one-standard-deviation move in that feature. I then computed predictions for 12 validation rows by hand as `intercept + Σ coefficient·x`, and compared them with `.predict()`, actual prices, and residuals (`artifacts/tables/manual_prediction_comparison.csv`).

Manual and scikit-learn metrics agree exactly: manual MAE 23,691.79 vs sklearn 23,691.79, and manual RMSE 29,449.60 vs sklearn 29,449.60. The maximum absolute difference between manual and scikit-learn predictions is ≈ 1.5 × 10⁻¹¹ (`artifacts/tables/manual_vs_sklearn_predictions.csv`), comfortably inside a 1 × 10⁻⁶ tolerance. Any difference is pure floating-point rounding from the order of summation, not a difference in method. Intercept, coefficients, feature scale, and residuals are tightly linked: because the features are standardized, the coefficients are directly comparable in PLN per standard deviation; the intercept absorbs the mean price; and the residual is just the part of each actual price the additive equation cannot reach.

What the exercise revealed that `.fit()`/`.predict()` hide: prediction is nothing more than a dot product plus the intercept, the intercept is meaningful only relative to how features were centered, coefficient size depends entirely on feature scaling (which is why standardization matters before comparing coefficients), and residuals are computed per row rather than handed down by the library. The negative coefficient on `engine_liters` would have been impossible to read off without standardizing first.

Problem 4: Linear-Model Experiments and Runtime-Aware Model Selection

Baseline `LinearRegression`** (full 62-feature matrix): train MAE 24,717, RMSE 45,327, R² 0.71; validation MAE 24,419, RMSE 41,689, R² 0.73. Train and validation are close, so the model is not overfitting. Validation diagnostics are saved at `artifacts/figures/linear_regression_diagnostics.png`.

Ridge/Lasso validation loops** over alphas `[0.001, 0.01, 0.1, 1, 10, 100]` (`artifacts/tables/ridge_lasso_validation.csv`). Both families are essentially flat across alpha because the model is already well-conditioned; the best validation RMSE for each occurs at alpha = 10. Lasso sparsity grows monotonically with alpha: 1 zero coefficient at alpha = 0.01, 12 zeros at alpha = 10, and 26 zeros at alpha = 100 — Lasso is pruning weak one-hot make levels first. I selected **Lasso(alpha=10)** as the final scikit-learn model (lowest validation RMSE among LinearRegression, best Ridge, and best Lasso): validation MAE 24,377, RMSE 41,680, R² 0.732, with a simpler 12-features-pruned model than plain LinearRegression.

Residual diagnostics and coefficients** (`artifacts/figures/selected_sklearn_diagnostics.png`, `artifacts/tables/selected_model_coefficients.csv`). The largest positive validation residual is ≈ +716,940 (a very expensive car the linear model badly under-predicts) and the largest negative is ≈ −219,556. The residual plot shows a clear funnel (heteroscedasticity) and the linear model even predicts negative prices for the cheapest cars. Strongest positive coefficients: `fuel_Electric` (≈ +149,656), `vol_engine` (≈ +51,839), `mark_price_mean` (≈ +19,440), `mark_opel`, and `fuel_Hybrid`. Strongest negative: `age_x_liters` (≈ −55,831), `log_mileage` (≈ −25,932), `mileage` (≈ −22,632), `mark_chevrolet`, and `mark_mazda`. One limitation of coefficient interpretation here: features are correlated and partly redundant (`vol_engine`, `engine_liters`, and `age_x_liters` all encode engine size; `mileage` and `log_mileage` overlap), so a single coefficient is a partial effect conditional on the rest of the encoded space, not a clean standalone price driver.

Bounded H2O-3 GLM.** I started a local cluster with `h2o.init(name="ML_Project_Cluster", max_mem_size="4G", nthreads=4, verbose=False)`, converted the cleaned train/validation/test pandas splits to `H2OFrame`s, and cast all six categorical predictors to factors with `.asfactor()`. The baseline GLM (`family="gaussian"`, `lambda_=0`) scored validation MAE 16,386, RMSE 30,094, R² 0.86. The regularized GLM (ridge, `alpha=0`, `lambda_=0.001`) scored validation MAE 19,341, RMSE 34,267, R² 0.82 — slightly worse, because mild shrinkage costs a little fit when the make/model signal is strong and the training set is large. I selected the **baseline GLM** on validation RMSE. The cluster was shut down with `h2o.cluster().shutdown(prompt=False)` at the end of the run (console prints "H2O cluster shut down").

Final test evaluation** (run once, no changes afterward): selected scikit-learn Lasso — MAE 24,290, RMSE 43,221, R² 0.724. Selected H2O baseline GLM — MAE 16,280, RMSE 31,174, R² 0.856.

Problem 5: Comparison, Artifacts, and Reflection

Cross-framework comparison** (`artifacts/tables/framework_comparison.csv`):

| framework | model | split | MAE | RMSE | R² |
|---|---|---|---|---|---|
| scikit-learn | Lasso(alpha=10) | validation | 24,377 | 41,680 | 0.732 |
| scikit-learn | Lasso(alpha=10) | test | 24,290 | 43,221 | 0.724 |
| H2O-3 | Baseline GLM (λ=0) | validation | 16,386 | 30,094 | 0.860 |
| H2O-3 | Baseline GLM (λ=0) | test | 16,280 | 31,174 | 0.856 |

H2O wins decisively on every metric and every split — about 8,000 PLN lower MAE and ~0.13 higher R² on test. The reason is encoding: H2O internally one-hot expands every high-cardinality factor (`model`, `city`, `generation_name`) into thousands of indicator levels, capturing make/model/location signal that my deliberately compact frequency encoding compresses into a single number per column. Both models generalize cleanly (validation and test metrics are close), so neither is overfitting; the gap is a representational-capacity difference, not a tuning artifact.

Interpretation.** The influential features make business sense: electric/large-engine cars and high-priced makes push price up; age, mileage, and the age×displacement interaction push it down. The residuals tell the real story of model limits — they fan out with predicted price (heteroscedasticity), the additive linear form predicts negative prices for the cheapest cars, and a handful of luxury listings produce six-figure residuals the model cannot reach. **Pricing takeaway:** make/model identity and powertrain dominate value far more than raw mileage, so a pricing tool should condition on those categoricals first rather than treating mileage as the headline lever. **Next modeling step:** move to a tree-based model (gradient-boosted trees or random forest), which handles high-cardinality splits, non-linear depreciation, and interactions natively, and/or model `log(price)` to tame the skew and stop negative predictions.

Artifacts.** Every referenced figure and table is indexed in `artifacts/tables/artifact_summary.csv`.

Workflow Reflection

I checked for leakage by enforcing one rule everywhere: every imputation value, encoder, frequency map, target aggregation, and scaler is fit inside `fit_preprocessing` on training rows only, then merely applied by `transform_preprocessing`; the eight-row leakage audit table makes each fit-on/applied-to relationship explicit. The detail I verified by hand was the linear-regression prediction itself — recomputing 12 predictions as `intercept + Σ coefficient·x` and confirming a max difference of ~1.5e-11 against `.predict()` gave me confidence the equation, not a black box, drives the numbers. Residuals and coefficients revealed the model's ceiling: clear heteroscedasticity, negative predicted prices at the low end, and unreachable luxury outliers show an additive linear model on standardized features is structurally too simple for skewed price data, even though it comfortably beats the mean baseline (R² 0.73 vs ~0). On regularization and GLMs I learned that Ridge and Lasso barely moved the metrics here because the feature matrix was already well-conditioned — Lasso's main contribution was sparsity (pruning 12 weak coefficients at alpha=10) rather than accuracy — and that H2O's native high-cardinality factor handling, not any regularization trick, was what produced the large quality gap. Two questions I would investigate next: (1) how much of H2O's advantage survives if I give scikit-learn the same rare-grouped one-hot expansion of `model`/`city`, isolating encoding from framework? (2) does modeling `log(price)` plus a tree-based learner remove the heteroscedasticity and negative-price artifacts while improving tail accuracy on the most expensive cars?
