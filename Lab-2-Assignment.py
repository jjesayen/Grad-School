"""
Lab 2: K-Nearest Neighbors and Classification

Final graded submission template.

Submit this file, Lab-2-Report.md, and any referenced files under artifacts/.
The notebook may be used only as an optional exploration workspace.
"""

from pathlib import Path
import time

from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "datasets"
ARTIFACT_DIR = LAB_DIR / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"

ADULT_DATA_PATH = DATA_DIR / "adult.data"
ADULT_TEST_PATH = DATA_DIR / "adult.test"


def ensure_artifact_dirs():
    """Create local artifact folders used by the final report."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def fit_preprocessing(train_df, y_train):
    """
    Learn all preprocessing state from the training split only.

    Store imputation values, encoder state, target-encoding mappings,
    fallback values, feature-column order, and scaler state in an explicit
    object such as a dictionary. Do not inspect validation or test rows here.
    """
    preprocessing_state = {
        "numeric_columns": None,
        "one_hot_columns": None,
        "target_encoded_columns": None,
        "imputation_values": None,
        "one_hot_encoder": None,
        "target_encoder": None,
        "target_encoding_fallback": None,
        "scaler": None,
        "feature_columns": None,
    }
    return preprocessing_state


def transform_preprocessing(df, preprocessing_state):
    """
    Apply already learned preprocessing state to one compatible split.

    This function should not fit, refit, or recompute training-only values.
    It should return a numeric feature matrix with the same column order for
    train, validation, and test.
    """
    transformed_features = None
    return transformed_features


def euclidean_distances_from_one(row, training_matrix):
    """
    Compute Euclidean distances from one row to every row in training_matrix.

    Implement this with NumPy vectorized operations for Problem 3.1.
    """
    distances = None
    return distances


def manual_knn_predict(X_train, y_train, X_query, k):
    """
    Predict binary labels for X_query using a manual KNN majority vote.

    Do not call scikit-learn inside this function.
    """
    predictions = None
    return predictions


def main():
    ensure_artifact_dirs()
    print("Lab 2 Python template started.")
    print(f"Data folder: {DATA_DIR}")
    print(f"Artifacts folder: {ARTIFACT_DIR}")

    # ==========================================================
    # PART 1: Data Exploration & Understanding (10 points)
    # ==========================================================

    # Problem 1.1: Load, Clean Labels, and Inspect Files (4 points)
    adult_train = None
    adult_test = None

    # Problem 1.2: Target Distribution and Data Quality Review (4 points)
    target_distribution = None

    # Problem 1.3: Baseline Risk Notes for KNN (2 points)
    knn_risk_notes = None

    # ==========================================================
    # PART 2: Leakage-Safe Preprocessing and Feature Engineering (30 points)
    # ==========================================================

    # Problem 2.1: Split Before Fitting Transformations (5 points)
    split_data = None

    # Problem 2.2: Implement fit_preprocessing and transform_preprocessing (10 points)
    preprocessing_state = None
    X_train = None
    X_validation = None
    X_test = None

    # Problem 2.3: Target Encoding Comparison for High-Cardinality Categories (8 points)
    target_encoding_comparison = None

    # Problem 2.4: Leakage Audit Table (4 points)
    leakage_audit_table = None

    # Problem 2.5: Final Feature Matrix Checks (3 points)
    feature_matrix_checks = None

    # ==========================================================
    # PART 3: KNN Mechanics From Scratch (20 points)
    # ==========================================================

    # Problem 3.1: Manual Euclidean Distance Function (5 points)
    manual_distance_example = None

    # Problem 3.2: Manual KNN Prediction on a Small Subset (8 points)
    manual_knn_results = None

    # Problem 3.3: Compare Manual KNN to scikit-learn KNN (4 points)
    manual_vs_sklearn_agreement = None

    # Problem 3.4: Manual KNN Reflection (3 points)
    manual_knn_reflection_notes = None

    # ==========================================================
    # PART 4: KNN Experiments and Runtime-Aware Model Selection (25 points)
    # ==========================================================

    # Problem 4.1: Baseline scikit-learn KNN (5 points)
    baseline_knn = None

    # Problem 4.2: Distance Diagnostics Before and After Scaling (5 points)
    distance_diagnostic_results = None

    # Problem 4.3: Encoding Strategy Comparison (5 points)
    encoding_strategy_results = None

    # Problem 4.4: Runtime-Aware k and Distance-Metric Validation Loop (7 points)
    runtime_validation_loop = None

    # Problem 4.5: Final Test Evaluation (3 points)
    final_test_results = None

    # ==========================================================
    # PART 5: Evaluation, Artifacts, and Reflection (15 points)
    # ==========================================================

    # Problem 5.1: Confusion Matrices and Metric Interpretation (5 points)
    confusion_matrix_artifacts = None

    # Problem 5.2: Required Artifact Summary Table (5 points)
    artifact_summary = None

    # Problem 5.3: Final Recommendation and Reflection (5 points)
    final_reflection_notes = None

    print("Template execution completed. Replace TODO blocks with your implementation.")


if __name__ == "__main__":
    main()
