"""
Lab 3: Leakage-Safe Linear Models for Regression

Final graded submission.

Submit this file, Lab-3-Report.md, and any referenced files under artifacts/.
The notebook may be used only as an optional exploration workspace.

Workflow summary
----------------
PART 1  Load/clean/audit, target + feature review, baseline framing.
PART 2  Leakage-safe train/val/test splits, fit/transform preprocessing,
        high-cardinality encoding comparison, training-only feature
        engineering, and a leakage audit.
PART 3  Linear-regression mechanics rebuilt by hand on a small subset.
PART 4  scikit-learn LinearRegression/Ridge/Lasso experiments with explicit
        validation loops, a bounded H2O-3 GLM comparison, and a single final
        test evaluation.
PART 5  Cross-framework comparison, interpretation, and artifact summary.

Every preprocessing object is fit on the training split only. Validation data
drives all model-selection decisions. The final test split is touched once,
after every modeling choice is frozen.
"""

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, file-only plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "price"
LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "datasets"
ARTIFACT_DIR = LAB_DIR / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"

CAR_PRICES_PATH = DATA_DIR / "Car_Prices_Poland.csv"

# Column roles used throughout the preprocessing functions.
NUMERIC_RAW = ["year", "mileage", "vol_engine"]
LOW_CARD_CATEGORICAL = ["mark", "fuel", "province"]
HIGH_CARD_CATEGORICAL = ["model", "generation_name", "city"]
CATEGORICAL_RAW = LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL
MISSING_SENTINEL = "__MISSING__"


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


# ----------------------------------------------------------------------
# Required design pattern: fit_preprocessing / transform_preprocessing
# ----------------------------------------------------------------------
def fit_preprocessing(train_df, y_train):
    """Learn all training-only preprocessing state.

    Everything below is computed from the training split only. Nothing from
    validation or test ever influences the stored state, which is the core
    leakage control of this lab.
    """
    state = {}
    train_df = train_df.copy()

    # 1. Missing-value handling learned from training rows only.
    #    Numeric: training medians. Categorical: a sentinel category.
    state["numeric_medians"] = {c: float(train_df[c].median()) for c in NUMERIC_RAW}
    train_df[NUMERIC_RAW] = train_df[NUMERIC_RAW].fillna(state["numeric_medians"])
    for c in CATEGORICAL_RAW:
        train_df[c] = train_df[c].fillna(MISSING_SENTINEL).astype(str)

    # 2. Engineered-feature state (training only).
    #    Reference year for car_age; mark-level mean price for aggregation.
    state["ref_year"] = int(train_df["year"].max())
    global_mean_price = float(np.mean(y_train))
    state["global_mean_price"] = global_mean_price
    mark_price = pd.DataFrame({"mark": train_df["mark"].values, "price": np.asarray(y_train)})
    state["mark_price_mean"] = mark_price.groupby("mark")["price"].mean().to_dict()

    # 3. High-cardinality frequency maps (training proportions; unknown -> 0).
    freq_maps = {}
    n_train = len(train_df)
    for c in HIGH_CARD_CATEGORICAL:
        freq_maps[c] = (train_df[c].value_counts() / n_train).to_dict()
    state["freq_maps"] = freq_maps

    # Build the engineered/encoded numeric block so the scaler and column
    # order can be learned on exactly what transform will later produce.
    numeric_block = _build_numeric_block(train_df, state)

    # 4. One-hot encoder for low-cardinality categoricals (training only).
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(train_df[LOW_CARD_CATEGORICAL])
    state["ohe"] = ohe
    state["ohe_feature_names"] = list(ohe.get_feature_names_out(LOW_CARD_CATEGORICAL))

    # 5. Scaler learned from the training numeric block only. One-hot 0/1
    #    indicator columns are intentionally left unscaled.
    scaler = StandardScaler()
    scaler.fit(numeric_block.values)
    state["scaler"] = scaler
    state["numeric_columns"] = list(numeric_block.columns)

    # 6. Final feature order (scaled numeric block first, then one-hot block).
    state["feature_order"] = state["numeric_columns"] + state["ohe_feature_names"]

    return state


def _build_numeric_block(df, state):
    """Assemble the continuous/engineered numeric columns (pre-scaling).

    Shared by fit and transform so the two paths stay identical.
    """
    out = pd.DataFrame(index=df.index)

    # Raw numeric predictors.
    for c in NUMERIC_RAW:
        out[c] = df[c].astype(float)

    # Engineered transformations.
    car_age = (state["ref_year"] - df["year"]).clip(lower=0).astype(float)
    out["car_age"] = car_age
    out["log_mileage"] = np.log1p(df["mileage"].astype(float))
    out["engine_liters"] = df["vol_engine"].astype(float) / 1000.0
    # Interaction feature.
    out["age_x_liters"] = out["car_age"] * out["engine_liters"]
    # Derived ratio.
    out["mileage_per_year"] = df["mileage"].astype(float) / (out["car_age"] + 1.0)

    # Aggregation/statistical feature: training mean price by mark, global fallback.
    out["mark_price_mean"] = (
        df["mark"].map(state["mark_price_mean"]).fillna(state["global_mean_price"]).astype(float)
    )

    # High-cardinality frequency encodings (unknown categories -> 0.0).
    for c in HIGH_CARD_CATEGORICAL:
        out[f"{c}_freq"] = df[c].map(state["freq_maps"][c]).fillna(0.0).astype(float)

    return out


def transform_preprocessing(df, preprocessing_state):
    """Apply learned preprocessing state to any compatible split.

    Returns a numeric DataFrame with identical columns in identical order for
    train, validation, and final test data. Unknown categories never crash:
    frequency maps fall back to 0, the mark aggregation falls back to the
    global training mean, and the one-hot encoder ignores unseen levels.
    """
    state = preprocessing_state
    df = df.copy()

    # Apply learned missing-value handling.
    df[NUMERIC_RAW] = df[NUMERIC_RAW].fillna(state["numeric_medians"])
    for c in CATEGORICAL_RAW:
        df[c] = df[c].fillna(MISSING_SENTINEL).astype(str)

    # Numeric/engineered block, then scale with the training scaler.
    numeric_block = _build_numeric_block(df, state)
    numeric_block = numeric_block[state["numeric_columns"]]
    scaled = state["scaler"].transform(numeric_block.values)
    scaled_df = pd.DataFrame(scaled, columns=state["numeric_columns"], index=df.index)

    # One-hot block from the fitted encoder.
    ohe_arr = state["ohe"].transform(df[LOW_CARD_CATEGORICAL])
    ohe_df = pd.DataFrame(ohe_arr, columns=state["ohe_feature_names"], index=df.index)

    out = pd.concat([scaled_df, ohe_df], axis=1)
    out = out[state["feature_order"]]  # enforce identical column order
    return out


# ----------------------------------------------------------------------
# Small helpers for tables/plots
# ----------------------------------------------------------------------
def save_table(df, name):
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def manual_mae(y_true, y_pred):
    """MAE rebuilt from first principles."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def manual_rmse(y_true, y_pred):
    """RMSE rebuilt from first principles."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    ensure_artifact_dirs()
    print("Lab 3 Python solution started.")
    print(f"Data path: {CAR_PRICES_PATH}")
    print(f"Artifacts folder: {ARTIFACT_DIR}")

    # ==========================================================
    # PART 1: Data Exploration and Regression Framing (10 points)
    # ==========================================================

    # Problem 1.1: Load, Clean, and Audit Dataset (4 points)
    raw_df = pd.read_csv(CAR_PRICES_PATH)
    print("\n[1.1] Raw shape:", raw_df.shape)
    print("[1.1] Raw columns:", list(raw_df.columns))
    print("[1.1] Dtypes:\n", raw_df.dtypes)
    print("[1.1] Head:\n", raw_df.head())

    # Drop the index-like first column (read in as 'Unnamed: 0').
    index_like = raw_df.columns[0]
    car_df = raw_df.drop(columns=[index_like])
    # Drop exact duplicate listings revealed once the index column is gone.
    n_dupes = int(car_df.duplicated().sum())
    car_df = car_df.drop_duplicates().reset_index(drop=True)

    expected_cols = set(NUMERIC_RAW + CATEGORICAL_RAW + [TARGET])
    assert expected_cols.issubset(set(car_df.columns)), "Missing expected columns."
    print(f"[1.1] Dropped index-like column: {index_like!r}")
    print(f"[1.1] Dropped duplicate rows: {n_dupes}")
    print("[1.1] Missing values per column:\n", car_df.isna().sum())
    print("[1.1] Cleaned shape:", car_df.shape)

    # Problem 1.2: Target Distribution and Feature Review (4 points)
    assert pd.api.types.is_numeric_dtype(car_df[TARGET]), "price must be numeric"
    price = car_df[TARGET]
    print("\n[1.2] price describe:\n", price.describe())
    print("[1.2] price skew:", round(float(price.skew()), 3))

    numeric_features = NUMERIC_RAW
    categorical_features = CATEGORICAL_RAW
    cardinality = (
        pd.DataFrame(
            {
                "feature": categorical_features,
                "n_unique": [int(car_df[c].nunique()) for c in categorical_features],
                "n_missing": [int(car_df[c].isna().sum()) for c in categorical_features],
            }
        )
        .sort_values("n_unique", ascending=False)
        .reset_index(drop=True)
    )
    print("[1.2] Categorical cardinality:\n", cardinality)
    save_table(cardinality, "categorical_cardinality.csv")

    target_feature_summary = {
        "price_describe": price.describe().to_dict(),
        "price_skew": float(price.skew()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cardinality": cardinality,
    }

    # Target-distribution artifact (raw and log scale side by side).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(price, bins=60, ax=axes[0], color="#3b6ea5")
    axes[0].set_title("Price distribution (raw)")
    axes[0].set_xlabel("price (PLN)")
    sns.histplot(np.log1p(price), bins=60, ax=axes[1], color="#a5533b")
    axes[1].set_title("Price distribution (log1p)")
    axes[1].set_xlabel("log1p(price)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "price_distribution.png", dpi=120)
    plt.close(fig)

    # Problem 1.3: Regression Baseline Risk Notes (2 points)
    # Mean-target baseline is computed AFTER splitting (below), using only the
    # training mean, so it never sees validation or test rows.

    # ==========================================================
    # PART 2: Leakage-Safe Preprocessing and Feature Engineering (30 points)
    # ==========================================================

    # Problem 2.1: Create Train, Validation, and Final Test Splits (5 points)
    X = car_df.drop(columns=[TARGET])
    y = car_df[TARGET]

    # Final test split first (20%), frozen until the very end.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    # Remaining 80% -> 64% train / 16% validation (0.20 of dev).
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=RANDOM_STATE
    )
    print("\n[2.1] Split sizes -> train:", X_train_df.shape[0],
          "val:", X_val_df.shape[0], "test:", X_test.shape[0])

    train_df = X_train_df.copy()
    validation_df = X_val_df.copy()
    test_df = X_test.copy()

    # Mean-target baseline from training rows only (answers 1.3).
    baseline_pred_value = float(np.mean(y_train))
    baseline_metrics = {
        "baseline_value": baseline_pred_value,
        "val": regression_metrics(y_val, np.full(len(y_val), baseline_pred_value)),
        "test": regression_metrics(y_test, np.full(len(y_test), baseline_pred_value)),
    }
    print("[1.3] Mean-target baseline value:", round(baseline_pred_value, 2))
    print("[1.3] Baseline validation metrics:", {k: round(v, 2) for k, v in baseline_metrics["val"].items()})

    # Problem 2.2: Implement fit_preprocessing and transform_preprocessing (10 points)
    preprocessing_state = fit_preprocessing(train_df, y_train.values)
    X_train = transform_preprocessing(train_df, preprocessing_state)
    X_validation = transform_preprocessing(validation_df, preprocessing_state)
    X_test_mat = transform_preprocessing(test_df, preprocessing_state)
    print("\n[2.2] Stored preprocessing_state keys:", list(preprocessing_state.keys()))
    print("[2.2] X_train shape:", X_train.shape,
          "| n features:", X_train.shape[1])

    # Problem 2.3: Encoding Strategy for High-Cardinality Categories (8 points)
    high_cardinality_results = compare_high_cardinality(
        train_df, validation_df, y_train.values, y_val.values
    )
    print("\n[2.3] High-cardinality strategy comparison (column = 'model'):")
    print(high_cardinality_results.to_string(index=False))
    save_table(high_cardinality_results, "high_cardinality_comparison.csv")

    # Problem 2.4: Feature Engineering with Training-Only State (5 points)
    feature_engineering_results = evaluate_feature_engineering(
        train_df, validation_df, y_train.values, y_val.values, preprocessing_state
    )
    print("\n[2.4] Feature-engineering impact on validation:")
    print(feature_engineering_results.to_string(index=False))
    save_table(feature_engineering_results, "feature_engineering_impact.csv")

    # Problem 2.5: Leakage Audit and Feature Matrix Checks (2 points)
    leakage_audit = build_leakage_audit()
    save_table(leakage_audit, "leakage_audit.csv")

    same_cols = (
        list(X_train.columns) == list(X_validation.columns) == list(X_test_mat.columns)
    )
    no_missing = not (
        X_train.isna().any().any()
        or X_validation.isna().any().any()
        or X_test_mat.isna().any().any()
    )
    feature_matrix_checks = pd.DataFrame(
        [
            {"matrix": "train", "rows": X_train.shape[0], "cols": X_train.shape[1]},
            {"matrix": "validation", "rows": X_validation.shape[0], "cols": X_validation.shape[1]},
            {"matrix": "test", "rows": X_test_mat.shape[0], "cols": X_test_mat.shape[1]},
        ]
    )
    print("\n[2.5] Feature-matrix shapes:\n", feature_matrix_checks.to_string(index=False))
    print("[2.5] Identical columns/order across splits:", same_cols)
    print("[2.5] No missing values after transform:", no_missing)
    save_table(feature_matrix_checks, "feature_matrix_shapes.csv")
    assert same_cols and no_missing, "Feature-matrix integrity check failed."

    # ==========================================================
    # PART 3: Linear Regression Mechanics From Scratch (20 points)
    # ==========================================================

    # Problem 3.1: Manual Prediction Equation on a Small Subset (5 points)
    mech_features = ["car_age", "log_mileage", "engine_liters"]
    n_small = 500
    X_small = X_train[mech_features].iloc[:n_small].values
    y_small = y_train.values[:n_small]

    small_model = LinearRegression()
    small_model.fit(X_small, y_small)
    intercept = float(small_model.intercept_)
    coefs = small_model.coef_.astype(float)

    n_rows = 12
    X_val_small = X_validation[mech_features].iloc[:n_rows].values
    y_val_small = y_val.values[:n_rows]

    # Manual prediction: intercept + sum(coef_j * x_j).
    manual_preds = intercept + X_val_small.dot(coefs)
    sklearn_preds = small_model.predict(X_val_small)

    manual_prediction_results = pd.DataFrame(
        {
            "row": np.arange(n_rows),
            "car_age_z": X_val_small[:, 0],
            "log_mileage_z": X_val_small[:, 1],
            "engine_liters_z": X_val_small[:, 2],
            "manual_pred": manual_preds,
            "sklearn_pred": sklearn_preds,
            "actual_price": y_val_small,
            "residual": y_val_small - manual_preds,
        }
    )
    print("\n[3.1] Manual prediction equation:")
    print(f"      price_hat = {intercept:.2f} "
          f"+ ({coefs[0]:.2f})*car_age_z "
          f"+ ({coefs[1]:.2f})*log_mileage_z "
          f"+ ({coefs[2]:.2f})*engine_liters_z")
    print(manual_prediction_results.round(2).to_string(index=False))
    save_table(manual_prediction_results, "manual_prediction_comparison.csv")

    # Problem 3.2: Manual Regression Metrics (6 points)
    m_mae = manual_mae(y_val_small, manual_preds)
    m_rmse = manual_rmse(y_val_small, manual_preds)
    sk_mae = mean_absolute_error(y_val_small, manual_preds)
    sk_rmse = rmse_score(y_val_small, manual_preds)
    manual_metric_results = pd.DataFrame(
        [
            {"metric": "MAE", "manual": m_mae, "sklearn": sk_mae, "abs_diff": abs(m_mae - sk_mae)},
            {"metric": "RMSE", "manual": m_rmse, "sklearn": sk_rmse, "abs_diff": abs(m_rmse - sk_rmse)},
        ]
    )
    print("\n[3.2] Manual vs sklearn metrics:")
    print(manual_metric_results.to_string(index=False))

    # Problem 3.3: Compare Manual Calculations to scikit-learn (5 points)
    max_abs_diff = float(np.max(np.abs(manual_preds - sklearn_preds)))
    within_tol = max_abs_diff < 1e-6
    print(f"\n[3.3] Max abs diff (manual vs sklearn predictions): {max_abs_diff:.3e}")
    print(f"[3.3] Within 1e-6 tolerance: {within_tol}")
    manual_comparison_results = manual_prediction_results.copy()
    manual_comparison_results["pred_abs_diff"] = np.abs(
        manual_comparison_results["manual_pred"] - manual_comparison_results["sklearn_pred"]
    )
    save_table(manual_comparison_results, "manual_vs_sklearn_predictions.csv")

    mechanics_reflection = {
        "intercept": intercept,
        "coefficients": dict(zip(mech_features, coefs.tolist())),
        "max_abs_diff": max_abs_diff,
        "within_tol": within_tol,
    }

    # ==========================================================
    # PART 4: Linear-Model Experiments and Runtime-Aware Model Selection (25 pts)
    # ==========================================================
    Xtr = X_train.values
    Xva = X_validation.values
    Xte = X_test_mat.values
    ytr = y_train.values
    yva = y_val.values
    yte = y_test.values

    # Problem 4.1: Baseline scikit-learn Linear Regression (5 points)
    linreg = LinearRegression()
    linreg.fit(Xtr, ytr)
    lr_train_pred = linreg.predict(Xtr)
    lr_val_pred = linreg.predict(Xva)
    linear_regression_results = {
        "train": regression_metrics(ytr, lr_train_pred),
        "val": regression_metrics(yva, lr_val_pred),
    }
    print("\n[4.1] LinearRegression train:",
          {k: round(v, 2) for k, v in linear_regression_results["train"].items()})
    print("[4.1] LinearRegression val:  ",
          {k: round(v, 2) for k, v in linear_regression_results["val"].items()})

    # Validation residual + actual-vs-predicted plots.
    _save_diagnostic_plots(yva, lr_val_pred, "linear_regression",
                           "LinearRegression (validation)")

    # Problem 4.2: Ridge and Lasso Validation Loops (8 points)
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    rows = []
    ridge_results = {}
    lasso_results = {}
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        ridge.fit(Xtr, ytr)
        r_tr = regression_metrics(ytr, ridge.predict(Xtr))
        r_va = regression_metrics(yva, ridge.predict(Xva))
        ridge_results[alpha] = {"model": ridge, "val": r_va}
        rows.append({
            "model": "Ridge", "alpha": alpha,
            "train_mae": r_tr["mae"], "train_rmse": r_tr["rmse"], "train_r2": r_tr["r2"],
            "val_mae": r_va["mae"], "val_rmse": r_va["rmse"], "val_r2": r_va["r2"],
            "n_zero_coef": int(np.sum(np.abs(ridge.coef_) < 1e-8)),
        })

        lasso = Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=20000)
        lasso.fit(Xtr, ytr)
        l_tr = regression_metrics(ytr, lasso.predict(Xtr))
        l_va = regression_metrics(yva, lasso.predict(Xva))
        lasso_results[alpha] = {"model": lasso, "val": l_va}
        rows.append({
            "model": "Lasso", "alpha": alpha,
            "train_mae": l_tr["mae"], "train_rmse": l_tr["rmse"], "train_r2": l_tr["r2"],
            "val_mae": l_va["mae"], "val_rmse": l_va["rmse"], "val_r2": l_va["r2"],
            "n_zero_coef": int(np.sum(np.abs(lasso.coef_) < 1e-8)),
        })

    validation_loop_table = pd.DataFrame(rows)
    print("\n[4.2] Ridge/Lasso validation loop:")
    print(validation_loop_table.round(3).to_string(index=False))
    save_table(validation_loop_table, "ridge_lasso_validation.csv")

    # Candidate models: plain LinearRegression + best Ridge + best Lasso by val RMSE.
    best_ridge_alpha = min(ridge_results, key=lambda a: ridge_results[a]["val"]["rmse"])
    best_lasso_alpha = min(lasso_results, key=lambda a: lasso_results[a]["val"]["rmse"])
    candidates = {
        "LinearRegression": (linreg, linear_regression_results["val"]),
        f"Ridge(alpha={best_ridge_alpha})": (
            ridge_results[best_ridge_alpha]["model"], ridge_results[best_ridge_alpha]["val"]),
        f"Lasso(alpha={best_lasso_alpha})": (
            lasso_results[best_lasso_alpha]["model"], lasso_results[best_lasso_alpha]["val"]),
    }
    selected_name = min(candidates, key=lambda k: candidates[k][1]["rmse"])
    selected_model = candidates[selected_name][0]
    print(f"[4.2] Best Ridge alpha: {best_ridge_alpha} | Best Lasso alpha: {best_lasso_alpha}")
    print(f"[4.2] Selected final scikit-learn model: {selected_name}")

    # Problem 4.3: Residual Diagnostics and Coefficient Review (4 points)
    sel_val_pred = selected_model.predict(Xva)
    _save_diagnostic_plots(yva, sel_val_pred, "selected_sklearn",
                           f"Selected {selected_name} (validation)")
    residuals = yva - sel_val_pred
    order = np.argsort(residuals)
    feat_names = list(X_train.columns)
    coef_series = pd.Series(selected_model.coef_, index=feat_names).sort_values()
    diagnostic_outputs = {
        "largest_negative_residual": float(residuals[order[0]]),
        "largest_positive_residual": float(residuals[order[-1]]),
        "strongest_negative_coefs": coef_series.head(5).to_dict(),
        "strongest_positive_coefs": coef_series.tail(5).to_dict(),
    }
    coef_table = (
        pd.Series(selected_model.coef_, index=feat_names)
        .rename("coefficient")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("coefficient")
        .reset_index(drop=True)
    )
    save_table(coef_table, "selected_model_coefficients.csv")
    print("\n[4.3] Largest +/- residuals (val):",
          round(diagnostic_outputs["largest_positive_residual"], 1),
          "/", round(diagnostic_outputs["largest_negative_residual"], 1))
    print("[4.3] Strongest positive coefs:",
          {k: round(v, 1) for k, v in diagnostic_outputs["strongest_positive_coefs"].items()})
    print("[4.3] Strongest negative coefs:",
          {k: round(v, 1) for k, v in diagnostic_outputs["strongest_negative_coefs"].items()})

    # Problem 4.4: Bounded H2O-3 GLM Comparison (5 points)
    h2o_frames, h2o_baseline_results, h2o_regularized_results, h2o_models, h2o_selected_name = (
        run_h2o_glm(train_df, validation_df, test_df, y_train, y_val, y_test)
    )

    # Problem 4.5: Final Test Evaluation (3 points)
    sel_test_pred = selected_model.predict(Xte)
    final_sklearn_test_results = regression_metrics(yte, sel_test_pred)
    final_h2o_test_results = h2o_models["test_metrics"]
    print("\n[4.5] Final scikit-learn test:",
          {k: round(v, 2) for k, v in final_sklearn_test_results.items()})
    print("[4.5] Final H2O test:        ",
          {k: round(v, 2) for k, v in final_h2o_test_results.items()})

    # ==========================================================
    # PART 5: Comparison, Artifacts, and Reflection (15 points)
    # ==========================================================

    # Problem 5.1: Cross-Framework Model Comparison (5 points)
    sel_val_metrics = regression_metrics(yva, sel_val_pred)
    framework_comparison = pd.DataFrame(
        [
            {"framework": "scikit-learn", "model": selected_name, "split": "validation",
             **sel_val_metrics},
            {"framework": "scikit-learn", "model": selected_name, "split": "test",
             **final_sklearn_test_results},
            {"framework": "H2O-3", "model": h2o_selected_name, "split": "validation",
             **h2o_models["val_metrics"]},
            {"framework": "H2O-3", "model": h2o_selected_name, "split": "test",
             **final_h2o_test_results},
        ]
    )
    print("\n[5.1] Cross-framework comparison:")
    print(framework_comparison.round(3).to_string(index=False))
    save_table(framework_comparison, "framework_comparison.csv")

    # Problem 5.2: Coefficient and Residual Interpretation (5 points)
    interpretation_outputs = {
        "top_positive": diagnostic_outputs["strongest_positive_coefs"],
        "top_negative": diagnostic_outputs["strongest_negative_coefs"],
        "test_residual_std": float(np.std(yte - sel_test_pred)),
    }

    # Problem 5.3: Required Artifact Summary and Reflection (5 points)
    artifact_summary = pd.DataFrame(
        [
            ["figures/price_distribution.png", "figure", "Problem 1.2",
             "Raw and log-scale price distribution"],
            ["tables/categorical_cardinality.csv", "table", "Problem 1.2",
             "Cardinality of categorical predictors"],
            ["tables/high_cardinality_comparison.csv", "table", "Problem 2.3",
             "Encoding strategy comparison on 'model'"],
            ["tables/feature_engineering_impact.csv", "table", "Problem 2.4",
             "Validation metrics with vs without engineered features"],
            ["tables/leakage_audit.csv", "table", "Problem 2.5",
             "Leakage audit across the pipeline"],
            ["tables/feature_matrix_shapes.csv", "table", "Problem 2.5",
             "Train/val/test feature-matrix shapes"],
            ["tables/manual_prediction_comparison.csv", "table", "Problem 3.1",
             "Manual vs sklearn predictions on 12 rows"],
            ["tables/manual_vs_sklearn_predictions.csv", "table", "Problem 3.3",
             "Prediction difference detail"],
            ["figures/linear_regression_diagnostics.png", "figure", "Problem 4.1",
             "Baseline LinearRegression validation diagnostics"],
            ["tables/ridge_lasso_validation.csv", "table", "Problem 4.2",
             "Ridge/Lasso alpha validation loop"],
            ["figures/selected_sklearn_diagnostics.png", "figure", "Problem 4.3",
             "Selected model residual diagnostics"],
            ["tables/selected_model_coefficients.csv", "table", "Problem 4.3",
             "Coefficients of the selected scikit-learn model"],
            ["tables/framework_comparison.csv", "table", "Problem 5.1",
             "scikit-learn vs H2O on validation and test"],
            ["tables/artifact_summary.csv", "table", "Problem 5.3",
             "This artifact index"],
        ],
        columns=["path", "type", "referenced_in", "description"],
    )
    save_table(artifact_summary, "artifact_summary.csv")
    print("\n[5.3] Artifact summary rows:", len(artifact_summary))

    # Keep linter-style references so each graded variable is clearly produced.
    _ = (target_feature_summary, baseline_metrics, manual_metric_results,
         mechanics_reflection, feature_engineering_results, h2o_frames,
         h2o_baseline_results, h2o_regularized_results, interpretation_outputs)

    print("\nLab 3 solution finished. Artifacts written under:", ARTIFACT_DIR)


# ----------------------------------------------------------------------
# Problem 2.3 helper: compare high-cardinality encodings on 'model'
# ----------------------------------------------------------------------
def compare_high_cardinality(train_df, val_df, y_train, y_val, column="model"):
    """Compare frequency, rare+one-hot, and smoothed target encodings.

    Each strategy is evaluated with a tiny LinearRegression on the three raw
    numeric predictors plus the chosen encoding of `column`, so differences
    reflect the encoding only. All mappings are learned on training rows; the
    unknown-category fallback is documented in the returned table.
    """
    base_cols = NUMERIC_RAW
    tr_base = train_df[base_cols].astype(float).values
    va_base = val_df[base_cols].astype(float).values
    n_train = len(train_df)
    global_mean = float(np.mean(y_train))
    results = []

    # --- Strategy A: frequency encoding (unknown -> 0).
    t0 = time.perf_counter()
    freq = (train_df[column].value_counts() / n_train).to_dict()
    tr_enc = train_df[column].map(freq).fillna(0.0).values.reshape(-1, 1)
    va_enc = val_df[column].map(freq).fillna(0.0).values.reshape(-1, 1)
    Xtr = np.hstack([tr_base, tr_enc])
    Xva = np.hstack([va_base, va_enc])
    m = LinearRegression().fit(Xtr, y_train)
    pred = m.predict(Xva)
    rt = time.perf_counter() - t0
    results.append({
        "strategy": "frequency_encoding", "n_features": Xtr.shape[1],
        "val_mae": manual_mae(y_val, pred), "val_rmse": manual_rmse(y_val, pred),
        "runtime_s": rt, "unknown_fallback": "0.0 (unseen freq)",
    })

    # --- Strategy B: rare-category grouping + one-hot (unknown -> RARE).
    t0 = time.perf_counter()
    counts = train_df[column].value_counts()
    frequent = set(counts[counts >= 50].index)  # keep categories seen >= 50x

    def group(v):
        return v if v in frequent else "__RARE__"

    tr_grouped = train_df[column].map(group)
    va_grouped = val_df[column].map(lambda v: v if v in frequent else "__RARE__")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    tr_oh = ohe.fit_transform(tr_grouped.values.reshape(-1, 1))
    va_oh = ohe.transform(va_grouped.values.reshape(-1, 1))
    Xtr = np.hstack([tr_base, tr_oh])
    Xva = np.hstack([va_base, va_oh])
    m = LinearRegression().fit(Xtr, y_train)
    pred = m.predict(Xva)
    rt = time.perf_counter() - t0
    results.append({
        "strategy": "rare_group_one_hot", "n_features": Xtr.shape[1],
        "val_mae": manual_mae(y_val, pred), "val_rmse": manual_rmse(y_val, pred),
        "runtime_s": rt, "unknown_fallback": "__RARE__ bucket / all-zero",
    })

    # --- Strategy C: smoothed target encoding (unknown -> global mean).
    t0 = time.perf_counter()
    tmp = pd.DataFrame({"cat": train_df[column].values, "y": y_train})
    grp = tmp.groupby("cat")["y"].agg(["mean", "count"])
    smoothing = 10.0
    smoothed = (grp["count"] * grp["mean"] + smoothing * global_mean) / (grp["count"] + smoothing)
    enc_map = smoothed.to_dict()
    tr_enc = train_df[column].map(enc_map).fillna(global_mean).values.reshape(-1, 1)
    va_enc = val_df[column].map(enc_map).fillna(global_mean).values.reshape(-1, 1)
    Xtr = np.hstack([tr_base, tr_enc])
    Xva = np.hstack([va_base, va_enc])
    m = LinearRegression().fit(Xtr, y_train)
    pred = m.predict(Xva)
    rt = time.perf_counter() - t0
    results.append({
        "strategy": "smoothed_target_encoding", "n_features": Xtr.shape[1],
        "val_mae": manual_mae(y_val, pred), "val_rmse": manual_rmse(y_val, pred),
        "runtime_s": rt, "unknown_fallback": f"global mean {global_mean:.0f}",
    })

    return pd.DataFrame(results)


# ----------------------------------------------------------------------
# Problem 2.4 helper: engineered features improvement check
# ----------------------------------------------------------------------
def evaluate_feature_engineering(train_df, val_df, y_train, y_val, state):
    """Compare validation metrics with and without engineered features.

    'Without' uses only raw-numeric + one-hot + frequency encodings.
    'With' uses the full feature matrix the pipeline actually produces.
    """
    engineered_cols = ["car_age", "log_mileage", "engine_liters",
                       "age_x_liters", "mileage_per_year", "mark_price_mean"]

    full_tr = transform_preprocessing(train_df, state)
    full_va = transform_preprocessing(val_df, state)
    base_tr = full_tr.drop(columns=engineered_cols)
    base_va = full_va.drop(columns=engineered_cols)

    def fit_eval(Xt, Xv):
        m = LinearRegression().fit(Xt.values, y_train)
        p = m.predict(Xv.values)
        return manual_mae(y_val, p), manual_rmse(y_val, p)

    base_mae, base_rmse = fit_eval(base_tr, base_va)
    full_mae, full_rmse = fit_eval(full_tr, full_va)
    return pd.DataFrame(
        [
            {"feature_set": "without_engineered", "n_features": base_tr.shape[1],
             "val_mae": base_mae, "val_rmse": base_rmse},
            {"feature_set": "with_engineered", "n_features": full_tr.shape[1],
             "val_mae": full_mae, "val_rmse": full_rmse},
        ]
    )


# ----------------------------------------------------------------------
# Problem 2.5 helper: leakage audit table
# ----------------------------------------------------------------------
def build_leakage_audit():
    rows = [
        {
            "step": "train/validation/test splitting",
            "fit_on": "none (raw CSV)",
            "applied_to": "whole dataset",
            "stored_state": "row indices per split",
            "leakage_risk": "rows shared across splits or preprocessing fit before split",
            "how_risk_was_controlled": "split first with random_state=42; nothing fit before splitting",
        },
        {
            "step": "missing-value handling",
            "fit_on": "train",
            "applied_to": "train/val/test",
            "stored_state": "numeric medians, categorical sentinel",
            "leakage_risk": "imputation values computed from val/test",
            "how_risk_was_controlled": "medians stored from train only in fit_preprocessing",
        },
        {
            "step": "one-hot encoding",
            "fit_on": "train",
            "applied_to": "train/val/test",
            "stored_state": "fitted OneHotEncoder categories",
            "leakage_risk": "encoder learns categories present only in val/test",
            "how_risk_was_controlled": "fit on train; handle_unknown='ignore' for unseen levels",
        },
        {
            "step": "high-cardinality encoding",
            "fit_on": "train",
            "applied_to": "train/val/test",
            "stored_state": "frequency proportion maps",
            "leakage_risk": "frequencies counted using val/test rows",
            "how_risk_was_controlled": "counts from train only; unknown categories map to 0",
        },
        {
            "step": "engineered statistical features",
            "fit_on": "train",
            "applied_to": "train/val/test",
            "stored_state": "ref_year, global mean price, mark mean-price map",
            "leakage_risk": "target-derived means leak future prices into features",
            "how_risk_was_controlled": "means computed from y_train only; global-mean fallback",
        },
        {
            "step": "scaling",
            "fit_on": "train",
            "applied_to": "train/val/test",
            "stored_state": "StandardScaler mean/std",
            "leakage_risk": "scaler statistics include val/test distribution",
            "how_risk_was_controlled": "scaler fit on train numeric block only",
        },
        {
            "step": "model-selection loops",
            "fit_on": "train",
            "applied_to": "validation",
            "stored_state": "alpha choices, selected model",
            "leakage_risk": "tuning on test data",
            "how_risk_was_controlled": "alpha/model chosen on validation; test untouched",
        },
        {
            "step": "final test evaluation",
            "fit_on": "n/a",
            "applied_to": "test (once)",
            "stored_state": "frozen final models",
            "leakage_risk": "repeated peeking at test inflates optimism",
            "how_risk_was_controlled": "test scored exactly once after choices frozen",
        },
    ]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Diagnostic plot helper
# ----------------------------------------------------------------------
def _save_diagnostic_plots(y_true, y_pred, slug, title):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred, y_true, s=6, alpha=0.25, color="#3b6ea5")
    lim = [min(np.min(y_pred), np.min(y_true)), max(np.max(y_pred), np.max(y_true))]
    axes[0].plot(lim, lim, "r--", linewidth=1)
    axes[0].set_xlabel("predicted price")
    axes[0].set_ylabel("actual price")
    axes[0].set_title(f"Actual vs predicted\n{title}")
    axes[1].scatter(y_pred, residuals, s=6, alpha=0.25, color="#a5533b")
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("predicted price")
    axes[1].set_ylabel("residual (actual - predicted)")
    axes[1].set_title(f"Residuals\n{title}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{slug}_diagnostics.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Problem 4.4/4.5 helper: bounded H2O-3 GLM comparison
# ----------------------------------------------------------------------
def run_h2o_glm(train_df, val_df, test_df, y_train, y_val, y_test):
    """Bounded local H2O GLM: one baseline, one regularized, single test score.

    H2O receives the cleaned raw predictors and does its own categorical
    handling; categorical columns are cast with .asfactor(). The cluster is
    shut down before the function returns.
    """
    import h2o
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator

    h2o.init(name="ML_Project_Cluster", max_mem_size="4G", nthreads=4, verbose=False)

    predictors = NUMERIC_RAW + CATEGORICAL_RAW

    def to_frame(df_x, y):
        d = df_x.copy()
        for c in CATEGORICAL_RAW:
            d[c] = d[c].fillna(MISSING_SENTINEL).astype(str)
        d[TARGET] = np.asarray(y)
        hf = h2o.H2OFrame(d)
        for c in CATEGORICAL_RAW:
            hf[c] = hf[c].asfactor()
        return hf

    train_hf = to_frame(train_df, y_train.values)
    val_hf = to_frame(val_df, y_val.values)
    test_hf = to_frame(test_df, y_test.values)

    def metrics_from(model, hf, y_true):
        pred = model.predict(hf).as_data_frame()["predict"].values
        return regression_metrics(np.asarray(y_true), pred)

    # Baseline GLM: no regularization.
    baseline = H2OGeneralizedLinearEstimator(family="gaussian", lambda_=0.0)
    baseline.train(x=predictors, y=TARGET, training_frame=train_hf)
    baseline_val = metrics_from(baseline, val_hf, y_val.values)

    # Regularized GLM: ridge-style elastic net (alpha=0) with a small fixed
    # lambda. With thousands of high-cardinality factor levels, H2O's automatic
    # lambda_search over-shrinks toward an intercept-only fit, so a modest fixed
    # penalty gives a stable, interpretable regularized comparison.
    regularized = H2OGeneralizedLinearEstimator(
        family="gaussian", alpha=0.0, lambda_=0.001
    )
    regularized.train(x=predictors, y=TARGET, training_frame=train_hf)
    regularized_val = metrics_from(regularized, val_hf, y_val.values)

    print("\n[4.4] H2O baseline GLM val:   ",
          {k: round(v, 2) for k, v in baseline_val.items()})
    print("[4.4] H2O regularized GLM val:",
          {k: round(v, 2) for k, v in regularized_val.items()})

    # Select the better H2O model on validation RMSE.
    if regularized_val["rmse"] <= baseline_val["rmse"]:
        selected, selected_name, selected_val = regularized, "Regularized GLM (alpha=0.5)", regularized_val
    else:
        selected, selected_name, selected_val = baseline, "Baseline GLM (lambda=0)", baseline_val

    selected_test = metrics_from(selected, test_hf, y_test.values)
    print(f"[4.4] Selected H2O model: {selected_name}")

    h2o_frames = {"train": train_hf.shape, "val": val_hf.shape, "test": test_hf.shape}
    h2o_models = {
        "selected_name": selected_name,
        "val_metrics": selected_val,
        "test_metrics": selected_test,
    }

    h2o.cluster().shutdown(prompt=False)
    print("[4.4] H2O cluster shut down.")

    return h2o_frames, baseline_val, regularized_val, h2o_models, selected_name


if __name__ == "__main__":
    main()
