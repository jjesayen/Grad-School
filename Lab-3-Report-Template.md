# Lab 3 Report Template

Name:

NetID:

Use this template to create `Lab-3-Report.md`. Keep final code in `Lab-3-Assignment.py` and save referenced plots, tables, or other outputs under `artifacts/`.

## Problem 1: Data Exploration and Regression Framing

- Summarize cleaned dataset shape, dropped index-like column, missing values, duplicated rows, and numeric/categorical feature types.
- Describe the `price` distribution, outliers, and high-cardinality categorical features.
- Explain which data properties may make linear regression difficult: target skew, high-cardinality categories, outliers, and possible non-linear relationships.

## Problem 2: Leakage-Safe Preprocessing and Feature Engineering

- State your train, validation, and final test split sizes.
- Explain your `fit_preprocessing` and `transform_preprocessing` design.
- List what state is stored in `preprocessing_state`.
- Explain missing-value handling, one-hot encoding, high-cardinality encoding, unknown-category fallback, engineered features, and scaling.
- Compare high-cardinality strategies: feature count, approximate runtime, validation MAE, and validation RMSE.
- Reference your leakage audit table and explain the highest-risk leakage point.
- Reference your final feature-matrix summary.

## Problem 3: Linear Regression Mechanics From Scratch

- Explain the manual prediction equation, including intercept and coefficient use.
- Reference your table comparing manual predictions, scikit-learn predictions, actual prices, and residuals.
- Report manual MAE/RMSE and compare them with scikit-learn metric functions.
- State the maximum absolute difference between manual and scikit-learn predictions.
- Explain what the mechanics exercise revealed that `.fit()` and `.predict()` hide.

## Problem 4: Linear-Model Experiments and Runtime-Aware Model Selection

- Report train and validation metrics for `LinearRegression`, Ridge, and Lasso.
- Explain the Ridge/Lasso alpha validation loop, selected scikit-learn model, and Lasso sparsity behavior.
- Interpret residual diagnostics and coefficient patterns.
- Report H2O initialization settings, categorical factor handling, baseline GLM validation metrics, and regularized GLM validation metrics.
- State final scikit-learn and H2O test metrics.
- Confirm H2O was shut down after the run.

## Problem 5: Comparison, Artifacts, and Reflection

- Compare final scikit-learn and H2O model metrics on validation and final test data.
- Interpret influential coefficients and residual patterns.
- State one pricing-strategy takeaway and one next modeling step.
- Reference the required artifact summary table.

## Workflow Reflection

Write 8-12 lines covering:

- how you checked for leakage
- one implementation or interpretation detail you had to verify manually
- what residuals or coefficients revealed about model limitations
- what you learned about regularization and GLM behavior
- two questions you would investigate in later modules
