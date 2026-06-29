"""
Lab 3: Leakage-Safe Linear Models for Regression

Final graded submission template.

Submit this file, Lab-3-Report.md, and any referenced files under artifacts/.
The notebook may be used only as an optional exploration workspace.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "price"
LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "datasets"
ARTIFACT_DIR = LAB_DIR / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"

CAR_PRICES_PATH = DATA_DIR / "Car_Prices_Poland.csv"


def ensure_artifact_dirs():
    """Create local artifact folders used by the final report."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def rmse_score(y_true, y_pred):
    """Return RMSE without relying on removed sklearn keyword arguments."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred):
    """Metric bundle for regression model comparisons."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse_score(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def fit_preprocessing(train_df, y_train):
    """Learn all training-only preprocessing state.

    Replace this template body with your implementation. Store imputation
    values, encoder state, high-cardinality mappings, engineered-feature state,
    scaler state, and final feature order in the returned object.
    """
    preprocessing_state = {}
    return preprocessing_state


def transform_preprocessing(df, preprocessing_state):
    """Apply learned preprocessing state to any compatible split.

    Replace this template body with your implementation. Return a numeric
    feature matrix with the same columns in the same order for train,
    validation, and final test data.
    """
    transformed_features = None
    return transformed_features


def main():
    ensure_artifact_dirs()
    print("Lab 3 Python template started.")
    print(f"Data path: {CAR_PRICES_PATH}")
    print(f"Artifacts folder: {ARTIFACT_DIR}")

    # ==========================================================
    # PART 1: Data Exploration and Regression Framing (10 points)
    # ==========================================================

    # Problem 1.1: Load, Clean, and Audit Dataset (4 points)
    car_df = None

    # Problem 1.2: Target Distribution and Feature Review (4 points)
    target_feature_summary = None

    # Problem 1.3: Regression Baseline Risk Notes (2 points)
    baseline_metrics = None

    # ==========================================================
    # PART 2: Leakage-Safe Preprocessing and Feature Engineering (30 points)
    # ==========================================================

    # Problem 2.1: Create Train, Validation, and Final Test Splits (5 points)
    train_df = None
    validation_df = None
    test_df = None

    # Problem 2.2: Implement fit_preprocessing and transform_preprocessing (10 points)
    preprocessing_state = None
    X_train = None
    X_validation = None
    X_test = None

    # Problem 2.3: Encoding Strategy for High-Cardinality Categories (8 points)
    high_cardinality_results = None

    # Problem 2.4: Feature Engineering with Training-Only State (5 points)
    feature_engineering_results = None

    # Problem 2.5: Leakage Audit and Feature Matrix Checks (2 points)
    leakage_audit = None
    feature_matrix_checks = None

    # ==========================================================
    # PART 3: Linear Regression Mechanics From Scratch (20 points)
    # ==========================================================

    # Problem 3.1: Manual Prediction Equation on a Small Subset (5 points)
    manual_prediction_results = None

    # Problem 3.2: Manual Regression Metrics (6 points)
    manual_metric_results = None

    # Problem 3.3: Compare Manual Calculations to scikit-learn (5 points)
    manual_comparison_results = None

    # Problem 3.4: Mechanics Reflection (4 points)
    mechanics_reflection = None

    # ==========================================================
    # PART 4: Linear-Model Experiments and Runtime-Aware Model Selection (25 points)
    # ==========================================================

    # Problem 4.1: Baseline scikit-learn Linear Regression (5 points)
    linear_regression_results = None

    # Problem 4.2: Ridge and Lasso Validation Loops (8 points)
    ridge_results = None
    lasso_results = None

    # Problem 4.3: Residual Diagnostics and Coefficient Review (4 points)
    diagnostic_outputs = None

    # Problem 4.4: Bounded H2O-3 GLM Comparison (5 points)
    # Import h2o inside this section. Use:
    # h2o.init(name="ML_Project_Cluster", max_mem_size="4G", nthreads=4, verbose=False)
    # Cast categorical columns with .asfactor(), and shut the cluster down after Part 4.
    h2o_frames = None
    h2o_baseline_results = None
    h2o_regularized_results = None

    # Problem 4.5: Final Test Evaluation (3 points)
    final_sklearn_test_results = None
    final_h2o_test_results = None

    # ==========================================================
    # PART 5: Comparison, Artifacts, and Reflection (15 points)
    # ==========================================================

    # Problem 5.1: Cross-Framework Model Comparison (5 points)
    framework_comparison = None

    # Problem 5.2: Coefficient and Residual Interpretation (5 points)
    interpretation_outputs = None

    # Problem 5.3: Required Artifact Summary and Reflection (5 points)
    artifact_summary = None

    print("Template execution completed. Replace placeholder blocks with your implementation.")


if __name__ == "__main__":
    main()
