# Lab 2 Report Template

Name:

NetID:

Use this template to create `Lab-2-Report.md`. Keep final code in `Lab-2-Assignment.py` and save referenced plots, tables, or other outputs under `artifacts/`.

## Problem 1: Data Exploration and KNN Risk Notes

- Summarize dataset shape, target cleanup, missing values, feature types, target distribution, and class imbalance.
- Reference any target-distribution or data-quality artifacts.
- Explain which data properties may make KNN difficult: class imbalance, high-cardinality categories, large numeric ranges, and runtime cost.

## Problem 2: Leakage-Safe Preprocessing and Feature Engineering

- State your train/validation/test split sizes and class balance.
- Explain your `fit_preprocessing` and `transform_preprocessing` design.
- List what state is stored in `preprocessing_state`.
- Explain missing-value handling, one-hot encoding, target encoding, unknown-category fallback, and scaling.
- Compare one-hot vs target encoding for `native-country`: feature count, runtime, validation accuracy, and validation F1.
- Reference your leakage audit table and explain the highest-risk leakage point.
- Reference your final feature-matrix summary.

## Problem 3: KNN Mechanics From Scratch

- Explain your Euclidean distance implementation.
- Report the manual KNN subset size, selected `k`, accuracy, and F1.
- Reference the table with at least 10 manual predictions.
- Report the agreement rate between manual KNN and scikit-learn KNN.
- Explain why manual KNN becomes expensive on larger data.

## Problem 4: KNN Experiments and Runtime-Aware Model Selection

- Report the baseline scikit-learn KNN validation result.
- Summarize distance diagnostics before and after scaling.
- Explain which encoding strategy you chose for the final workflow and why.
- Summarize the runtime-aware validation loop across at least 10 `k` values and at least 2 distance settings.
- State the selected final `k`, distance metric, preprocessing strategy, and evidence for that choice.
- Report final test metrics after freezing model choices.

## Problem 5: Evaluation, Artifacts, and Reflection

- Interpret validation and test confusion matrices.
- Explain accuracy, precision, recall, and F1-score under class imbalance.
- Reference the required final artifact summary table.
- Recommend one final KNN configuration for this practice task.

## Workflow Reflection

Write 8-12 lines covering:

- one implementation or interpretation detail you had to verify manually
- how you checked for leakage
- what you learned about KNN strengths and weaknesses
- two questions you would investigate in later modules
