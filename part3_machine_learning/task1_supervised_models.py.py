# ============================================================
# PART 1 OF 4: IMPORTS, LOGGER, PATHS & PREPROCESSOR
# ============================================================

import os
import sys
import json
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_curve,
)

# Module-level logger only (handlers configured strictly in main())
logger = logging.getLogger(__name__)

# Directory & File Configuration
OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"
ARTIFACT_DIR = "artifacts"

INPUT_FILE = os.path.join(OUTPUT_DIR, "ml_traffic_accident_proxy.csv")

CLASSIFICATION_RESULTS_FILE = os.path.join(OUTPUT_DIR, "classification_results.csv")
REGRESSION_RESULTS_FILE = os.path.join(OUTPUT_DIR, "regression_results.csv")

LOGISTIC_MODEL_FILE = os.path.join(OUTPUT_DIR, "logistic_regression_model.pkl")
RF_CLASSIFIER_FILE = os.path.join(OUTPUT_DIR, "random_forest_classifier.pkl")
LINEAR_MODEL_FILE = os.path.join(OUTPUT_DIR, "linear_regression_model.pkl")
RF_REGRESSOR_FILE = os.path.join(OUTPUT_DIR, "random_forest_regressor.pkl")


def save_pickle_model(model, filepath):
    """Save trained model to disk."""
    with open(filepath, "wb") as file:
        pickle.dump(model, file)
    logger.info("Saved model: %s", filepath)


def save_json(obj, filepath):
    """Save JSON artifact to disk."""
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4)
    logger.info("Saved artifact: %s", filepath)


def build_preprocessor(numeric_features, categorical_features, scale_numeric=True):
    """Build ColumnTransformer preprocessing pipeline."""
    transformers = []

    if numeric_features:
        if scale_numeric:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
        else:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)

# ============================================================
# PART 2 OF 4: DATA LOADING, VALIDATION, FEATURES & SPLITS
# ============================================================

def load_dataset():
    """Load Task 0 proxy-labeled dataset."""
    logger.info("=" * 60)
    logger.info("LOADING DATASET")
    logger.info("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    logger.info("Dataset loaded successfully. Shape: %s", df.shape)
    return df


def validate_dataset(df):
    """Ensure required target columns exist."""
    required_columns = ["high_risk", "traffic_volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")


def engineer_features(df):
    """Prepare shared, leakage-free modelling features."""
    logger.info("Starting feature preparation and leakage removal.")

    if "Weather_Suall" in df.columns:
        df.rename(columns={"Weather_Suall": "Weather_Squall"}, inplace=True)
        logger.info("Corrected column typo 'Weather_Suall' -> 'Weather_Squall'.")

    # Datetime parsing (only create temporal columns if neither case exists)
    if "date_time" in df.columns:
        dt = pd.to_datetime(df["date_time"], errors="coerce")
        missing_dt = int(dt.isna().sum())

        if missing_dt > 0:
            logger.warning("%d invalid datetime values encountered.", missing_dt)

        if "Hour" not in df.columns and "hour" not in df.columns:
            df["Hour"] = dt.dt.hour
        if "Day_of_Week" not in df.columns and "day_of_week" not in df.columns:
            df["Day_of_Week"] = dt.dt.dayofweek
        if "Month" not in df.columns and "month" not in df.columns:
            df["Month"] = dt.dt.month

    categorical_features = []
    if "holiday" in df.columns and not pd.api.types.is_numeric_dtype(df["holiday"]):
        categorical_features.append("holiday")
    if "weather_description" in df.columns:
        categorical_features.append("weather_description")

    logger.debug("Identified categorical features: %s", categorical_features)

    # Remove targets, target-derived leakage columns, and raw timestamp
    leakage_columns = [
        "high_risk",
        "traffic_volume",
        "weather_main",
        "Congestion_Category",
        "congestion_category",
        "is_low_visibility",
        "date_time",
    ]

    feature_columns = [col for col in df.columns if col not in leakage_columns]

    X = df[feature_columns].copy()
    y_classification = df["high_risk"].astype(int)
    y_regression = df["traffic_volume"].astype(float)

    numeric_features = [col for col in X.columns if col not in categorical_features]

    logger.info(
        "Feature summary -> Total: %d | Numeric: %d | Categorical: %d",
        len(feature_columns),
        len(numeric_features),
        len(categorical_features),
    )

    save_json(feature_columns, os.path.join(ARTIFACT_DIR, "selected_features.json"))

    return (
        X,
        y_classification,
        y_regression,
        numeric_features,
        categorical_features,
        feature_columns,
    )


def create_classification_split(X, y):
    """Create stratified 80/20 split for classification."""
    logger.info("=" * 60)
    logger.info("CLASSIFICATION DATA SPLIT")
    logger.info("=" * 60)

    logger.debug("Class Distribution: %s", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info("Classification Split -> Train: %s | Test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def create_regression_split(X, y):
    """Create 80/20 split for regression."""
    logger.info("=" * 60)
    logger.info("REGRESSION DATA SPLIT")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    logger.info("Regression Split -> Train: %s | Test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

# ============================================================
# PART 3 OF 4: CLASSIFICATION MODELS
# ============================================================

def run_classification_models(
    X_train,
    X_test,
    y_train,
    y_test,
    numeric_features,
    categorical_features,
):
    """Train and evaluate Logistic Regression vs. Random Forest Classifier."""
    logger.info("=" * 60)
    logger.info("TRAINING CLASSIFICATION MODELS")
    logger.info("=" * 60)

    # 1. Logistic Regression (Baseline)
    logger.info("Training Logistic Regression...")
    logistic_model = Pipeline(
        steps=[
            (
                "prep",
                build_preprocessor(
                    numeric_features, categorical_features, scale_numeric=True
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    logistic_model.fit(X_train, y_train)
    lr_predictions = logistic_model.predict(X_test)
    lr_probabilities = logistic_model.predict_proba(X_test)[:, 1]
    logger.info("Logistic Regression complete.")

    # 2. Random Forest Classifier
    logger.info("Training Random Forest Classifier...")
    rf_classifier = Pipeline(
        steps=[
            (
                "prep",
                build_preprocessor(
                    numeric_features, categorical_features, scale_numeric=False
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)
    rf_probabilities = rf_classifier.predict_proba(X_test)[:, 1]
    logger.info("Random Forest Classifier complete.")

    # Evaluation Metrics
    classification_results = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest Classifier"],
            "Accuracy": [
                accuracy_score(y_test, lr_predictions),
                accuracy_score(y_test, rf_predictions),
            ],
            "Precision": [
                precision_score(y_test, lr_predictions, zero_division=0),
                precision_score(y_test, rf_predictions, zero_division=0),
            ],
            "Recall": [
                recall_score(y_test, lr_predictions, zero_division=0),
                recall_score(y_test, rf_predictions, zero_division=0),
            ],
            "F1-Score": [
                f1_score(y_test, lr_predictions, zero_division=0),
                f1_score(y_test, rf_predictions, zero_division=0),
            ],
            "ROC AUC": [
                roc_auc_score(y_test, lr_probabilities),
                roc_auc_score(y_test, rf_probabilities),
            ],
        }
    )

    classification_results.to_csv(CLASSIFICATION_RESULTS_FILE, index=False)
    logger.info("Saved: %s", CLASSIFICATION_RESULTS_FILE)

    save_pickle_model(logistic_model, LOGISTIC_MODEL_FILE)
    save_pickle_model(rf_classifier, RF_CLASSIFIER_FILE)

    # Log reports cleanly (Rule 5: no print() statements)
    logger.info(
        "\nLOGISTIC REGRESSION REPORT:\n%s",
        classification_report(y_test, lr_predictions, zero_division=0),
    )
    logger.info(
        "\nRANDOM FOREST REPORT:\n%s",
        classification_report(y_test, rf_predictions, zero_division=0),
    )
    logger.info(
        "\nCLASSIFICATION MODEL COMPARISON:\n%s",
        classification_results.round(4).to_string(index=False),
    )

    # ROC Curve
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probabilities)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities)
    diag = np.linspace(0, 1, 100)

    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, label="Logistic Regression")
    plt.plot(rf_fpr, rf_tpr, label="Random Forest")
    plt.plot(diag, diag, linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Proxy Accident Risk Classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "roc_curve.png"), dpi=300)
    plt.close()
    logger.info("Saved: roc_curve.png")

    # Classification Comparison Chart
    figure_data = classification_results.set_index("Model")
    figure_data[["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]].plot(
        kind="bar", figsize=(10, 6), rot=0
    )
    plt.title("Classification Model Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "classification_model_comparison.png"), dpi=300
    )
    plt.close()
    logger.info("Saved: classification_model_comparison.png")

    return classification_results, logistic_model, rf_classifier

# ============================================================
# PART 4 OF 4: REGRESSION, FEATURE IMPORTANCE, METADATA & MAIN
# ============================================================

def run_regression_models(
    X_train,
    X_test,
    y_train,
    y_test,
    numeric_features,
    categorical_features,
):
    """Train and evaluate Linear Regression vs. Random Forest Regressor."""
    logger.info("=" * 60)
    logger.info("REGRESSION MODELLING")
    logger.info("=" * 60)

    # 1. Linear Regression (Baseline)
    logger.info("Training Linear Regression...")
    linear_model = Pipeline(
        steps=[
            (
                "prep",
                build_preprocessor(
                    numeric_features, categorical_features, scale_numeric=True
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)
    logger.info("Linear Regression training complete.")

    # 2. Random Forest Regressor
    logger.info("Training Random Forest Regressor...")
    rf_regressor = Pipeline(
        steps=[
            (
                "prep",
                build_preprocessor(
                    numeric_features, categorical_features, scale_numeric=False
                ),
            ),
            (
                "model",
                RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            ),
        ]
    )
    rf_regressor.fit(X_train, y_train)
    rf_predictions = rf_regressor.predict(X_test)
    logger.info("Random Forest Regressor training complete.")

    regression_results = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Random Forest Regressor"],
            "MAE": [
                mean_absolute_error(y_test, linear_predictions),
                mean_absolute_error(y_test, rf_predictions),
            ],
            "R-squared": [
                r2_score(y_test, linear_predictions),
                r2_score(y_test, rf_predictions),
            ],
        }
    )

    regression_results.to_csv(REGRESSION_RESULTS_FILE, index=False)
    logger.info("Saved: %s", REGRESSION_RESULTS_FILE)

    logger.info(
        "\nREGRESSION MODEL COMPARISON:\n%s",
        regression_results.round(4).to_string(index=False),
    )

    save_pickle_model(linear_model, LINEAR_MODEL_FILE)
    save_pickle_model(rf_regressor, RF_REGRESSOR_FILE)

    # Actual vs Predicted Chart
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, rf_predictions, alpha=0.4)
    plt.xlabel("Actual Traffic Volume")
    plt.ylabel("Predicted Traffic Volume")
    plt.title("Random Forest Regressor\nActual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    logger.info("Saved: actual_vs_predicted.png")

    # Regression Comparison Chart (using two separate figures or unpack without brackets)
    fig, (ax_mae, ax_r2) = plt.subplots(1, 2, figsize=(10, 5))

    regression_results.plot(
        x="Model", y="MAE", kind="bar", ax=ax_mae, legend=False, rot=0
    )
    ax_mae.set_title("Mean Absolute Error")

    regression_results.plot(
        x="Model", y="R-squared", kind="bar", ax=ax_r2, legend=False, rot=0
    )
    ax_r2.set_title("R-Squared")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "regression_model_comparison.png"), dpi=300
    )
    plt.close()
    logger.info("Saved: regression_model_comparison.png")

    return regression_results, linear_model, rf_regressor


def create_feature_importance_plot(rf_regressor, numeric_features):
    """Create Random Forest feature importance chart."""
    try:
        rf_model = rf_regressor.named_steps["model"]
        importances = rf_model.feature_importances_

        preprocessor = rf_regressor.named_steps["prep"]
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = [
                name.split("__", 1)[-1]
                for name in preprocessor.get_feature_names_out()
            ]
        else:
            feature_names = numeric_features[: len(importances)]

        feature_importance = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance["Feature"], feature_importance["Importance"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title("Random Forest Feature Importance (Top 15)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "rf_feature_importance.png"), dpi=300)
        plt.close()
        logger.info("Saved: rf_feature_importance.png")

    except Exception as ex:
        logger.warning(
            "Feature importance generation failed: %s", ex, exc_info=True
        )


def save_model_metadata(df, feature_columns):
    """Save metadata artifact."""
    metadata = {
        "classification_target": "high_risk",
        "regression_target": "traffic_volume",
        "records": int(len(df)),
        "feature_count": len(feature_columns),
        "classification_models": [
            "Logistic Regression",
            "Random Forest Classifier",
        ],
        "regression_models": [
            "Linear Regression",
            "Random Forest Regressor",
        ],
    }
    save_json(metadata, os.path.join(ARTIFACT_DIR, "model_metadata.json"))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("pipeline.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        logger.info("=" * 60)
        logger.info("PART 3 - TASK 1 SUPERVISED LEARNING: STARTING")
        logger.info("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FIGURE_DIR, exist_ok=True)
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        df = load_dataset()
        validate_dataset(df)

        (
            X,
            y_clf,
            y_reg,
            numeric_features,
            categorical_features,
            feature_columns,
        ) = engineer_features(df)

        X_train_c, X_test_c, y_train_c, y_test_c = create_classification_split(
            X, y_clf
        )
        run_classification_models(
            X_train_c,
            X_test_c,
            y_train_c,
            y_test_c,
            numeric_features,
            categorical_features,
        )

        X_train_r, X_test_r, y_train_r, y_test_r = create_regression_split(
            X, y_reg
        )
        _, _, rf_regressor = run_regression_models(
            X_train_r,
            X_test_r,
            y_train_r,
            y_test_r,
            numeric_features,
            categorical_features,
        )

        create_feature_importance_plot(rf_regressor, numeric_features)
        save_model_metadata(df, feature_columns)

        logger.info("=" * 60)
        logger.info("PART 3 - TASK 1 SUPERVISED LEARNING: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Task 1 aborted: required dataset not found.", exc_info=True)
        sys.exit(1)
    except KeyError:
        logger.error("Task 1 aborted: required column missing.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Task 1 aborted due to an unexpected error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
