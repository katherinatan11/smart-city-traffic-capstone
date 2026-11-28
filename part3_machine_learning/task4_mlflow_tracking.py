# ============================================================
# PART 1 OF 5: IMPORTS, LOGGER, CONSTANTS & FEATURE PREP
# ============================================================

import os
import sys
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import json
import pickle
import inspect
import logging
from pathlib import Path

import numpy as np
import pandas as pd

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
    mean_absolute_error,
    r2_score,
)

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Module-level logger only (handlers configured strictly in main())
logger = logging.getLogger(__name__)

# Experiment & File Configuration
EXPERIMENT_NAME = "Traffic_Mobility_MLOps_Experiment"
TRACKING_URI = "sqlite:///mlflow.db"

OUTPUT_DIR = Path("outputs")
ARTIFACT_DIR = Path("artifacts")
DATA_SOURCE = OUTPUT_DIR / "ml_traffic_accident_proxy.csv"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common feature engineering and cap extreme weather outliers."""
    df = df.copy()

    if "date_time" in df.columns:
        dt = pd.to_datetime(df["date_time"], errors="coerce")
        invalid_dt = int(dt.isna().sum())
        if invalid_dt > 0:
            logger.warning("Coerced %d invalid 'date_time' value(s) to NaT.", invalid_dt)

        if "Hour" not in df.columns and "hour" not in df.columns:
            df["hour"] = dt.dt.hour
        if "Day_of_Week" not in df.columns and "day_of_week" not in df.columns:
            df["day_of_week"] = dt.dt.dayofweek
        if "Month" not in df.columns and "month" not in df.columns:
            df["month"] = dt.dt.month
        if "Is_Weekend" not in df.columns and "is_weekend" not in df.columns:
            dow_tmp = df["Day_of_Week"] if "Day_of_Week" in df.columns else df["day_of_week"]
            df["is_weekend"] = (dow_tmp >= 5).astype(int)

    hr_col = "Hour" if "Hour" in df.columns else ("hour" if "hour" in df.columns else None)
    if hr_col is not None:
        if "Hour_Sin" not in df.columns and "hour_sin" not in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df[hr_col] / 24.0)
        if "Hour_Cos" not in df.columns and "hour_cos" not in df.columns:
            df["hour_cos"] = np.cos(2 * np.pi * df[hr_col] / 24.0)

    dow_col = "Day_of_Week" if "Day_of_Week" in df.columns else ("day_of_week" if "day_of_week" in df.columns else None)
    if dow_col is not None:
        if "Day_Sin" not in df.columns and "dow_sin" not in df.columns:
            df["dow_sin"] = np.sin(2 * np.pi * df[dow_col] / 7.0)
        if "Day_Cos" not in df.columns and "dow_cos" not in df.columns:
            df["dow_cos"] = np.cos(2 * np.pi * df[dow_col] / 7.0)

    if "holiday" in df.columns and not pd.api.types.is_numeric_dtype(df["holiday"]):
        df["is_holiday"] = (df["holiday"].notna() & (df["holiday"].astype(str) != "None")).astype(int)
        df = df.drop(columns=["holiday"])

    for col in ["rain_1h", "snow_1h"]:
        if col in df.columns:
            cap = float(df[col].quantile(0.999))
            if cap > 0:
                mask = df[col] > cap
                n_out = int(mask.sum())
                if n_out > 0:
                    df.loc[mask, col] = cap
                    logger.warning("Capped %d extreme outlier(s) in '%s' at %.2f.", n_out, col, cap)

    return df

# ============================================================
# PART 2 OF 5: PREPROCESSOR & MLFLOW TRACKING WRAPPERS
# ============================================================

def build_preprocessor(numeric_cols: list, categorical_cols: list, scale_numeric: bool) -> ColumnTransformer:
    """Create preprocessing transformer for numeric and categorical features."""
    transformers = []
    if numeric_cols:
        num_step = (
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            if scale_numeric
            else SimpleImputer(strategy="median")
        )
        transformers.append(("num", num_step, numeric_cols))

    if categorical_cols:
        cat_step = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_step, categorical_cols))

    return ColumnTransformer(transformers=transformers)


def log_sklearn_model_compat(pipeline: Pipeline) -> None:
    """Log scikit-learn model safely using skops across MLflow 2.x and 3.x+."""
    sig_params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {}
    if "serialization_format" in sig_params:
        kwargs["serialization_format"] = getattr(
            mlflow.sklearn, "SERIALIZATION_FORMAT_SKOPS", "skops"
        )
    if "skops_trusted_types" in sig_params:
        kwargs["skops_trusted_types"] = ["numpy.dtype", "sklearn.tree._tree.Tree"]
    if "name" in sig_params:
        mlflow.sklearn.log_model(pipeline, name="model", **kwargs)
    else:
        mlflow.sklearn.log_model(pipeline, artifact_path="model", **kwargs)


def run_and_track_experiment(
    run_name: str,
    task_type: str,
    pipeline: Pipeline,
    params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_artifact_path: str,
    fallback_dir: Path
) -> dict:
    """Fit model, compute metrics, and log parameters, metrics, artifacts, and model to MLflow."""
    logger.info("Starting tracked run '%s' (%s) with params=%s", run_name, task_type, params)
    pipeline.fit(X_train, y_train)

    if task_type == "classification":
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
    else:
        y_pred = pipeline.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2_score": float(r2_score(y_test, y_pred)),
        }

    if HAS_MLFLOW:
        if mlflow.active_run() is not None:
            logger.warning("Closing leftover active MLflow run before starting '%s'.", run_name)
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("task_type", task_type)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(feature_artifact_path, artifact_path="metadata")
            log_sklearn_model_compat(pipeline)
            run_id = run.info.run_id
            logger.info("Logged run '%s' (run_id=%s) to MLflow. Metrics: %s", run_name, run_id, metrics)
    else:
        run_id = f"local_{run_name}"
        model_file = fallback_dir / f"{run_name}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved local fallback model '%s' and metrics: %s", model_file, metrics)

    return {"run_name": run_name, "run_id": run_id, "task_type": task_type, **params, **metrics}

# ============================================================
# PART 3 OF 5: MAIN SETUP, VALIDATION & MLFLOW INITIALISATION
# ============================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("pipeline.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if HAS_MLFLOW:
        logging.getLogger("mlflow.utils.git_utils").setLevel(logging.ERROR)

    try:
        logger.info("=" * 60)
        logger.info("PART 3 - TASK 4 MLFLOW EXPERIMENT TRACKING: STARTING")
        logger.info("=" * 60)

        OUTPUT_DIR.mkdir(exist_ok=True)
        ARTIFACT_DIR.mkdir(exist_ok=True)

        if not DATA_SOURCE.exists():
            raise FileNotFoundError(
                f"Inbound data error: '{DATA_SOURCE}' missing. Execute Proxy Label script first."
            )

        df = pd.read_csv(DATA_SOURCE)
        logger.info("Loaded dataset '%s' with shape %s.", DATA_SOURCE, df.shape)

        required_targets = ["high_risk", "traffic_volume"]
        missing = [c for c in required_targets if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required target column(s): {missing}")

        initial_rows = len(df)
        df = df.dropna(subset=required_targets).copy()
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.warning("Dropped %d row(s) with missing target values.", dropped)

        df = prepare_features(df)

        # Exclude targets, target-derived categories, and raw timestamp
        exclude_cols = {
            "traffic_volume",
            "high_risk",
            "congestion_category",
            "Congestion_Category",
            "is_low_visibility",
            "date_time",
        }
        FEATURES = [c for c in df.columns if c not in exclude_cols]

        X = df[FEATURES]
        y_clf = df["high_risk"].astype(int)
        y_reg = df["traffic_volume"].astype(float)

        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        # Save shared feature schema artifact
        feature_schema_path = ARTIFACT_DIR / "task4_features_used.json"
        with open(feature_schema_path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_features": len(FEATURES), "numeric": numeric_cols, "categorical": categorical_cols},
                f,
                indent=2,
            )
        logger.info("Saved shared feature schema artifact to '%s'.", feature_schema_path)

        # Setup MLflow SQLite Backend or Local Fallback
        fallback_dir = Path("mlruns_fallback")
        if HAS_MLFLOW:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.set_tracking_uri(TRACKING_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            logger.info("MLflow tracking initialized. URI='%s', Experiment='%s'", TRACKING_URI, EXPERIMENT_NAME)
        else:
            fallback_dir.mkdir(exist_ok=True)
            logger.warning(
                "Package 'mlflow' is not installed; falling back to local tracking in '%s'.",
                fallback_dir,
            )

# ============================================================
# PART 4 OF 5: TRACKED CLASSIFICATION & REGRESSION RUNS
# ============================================================

        # Train/Test splits
        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
            X, y_clf, test_size=0.20, random_state=42, stratify=y_clf
        )
        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
            X, y_reg, test_size=0.20, random_state=42
        )

        tracked_records = []

        # ----------------------------------------------------
        # Track Classification Experiments (3 runs)
        # ----------------------------------------------------
        clf_experiments = [
            (
                "clf_logistic_regression_C1",
                Pipeline([
                    ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
                    ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
                ]),
                {
                    "algorithm": "LogisticRegression",
                    "C": 1.0,
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "n_features": len(FEATURES),
                    "test_size": 0.20,
                },
            ),
            (
                "clf_random_forest_50_d10",
                Pipeline([
                    ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                    ("model", RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)),
                ]),
                {
                    "algorithm": "RandomForestClassifier",
                    "n_estimators": 50,
                    "max_depth": 10,
                    "class_weight": "balanced",
                    "n_features": len(FEATURES),
                    "test_size": 0.20,
                },
            ),
            (
                "clf_random_forest_100_d20",
                Pipeline([
                    ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                    ("model", RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1)),
                ]),
                {
                    "algorithm": "RandomForestClassifier",
                    "n_estimators": 100,
                    "max_depth": 20,
                    "class_weight": "balanced",
                    "n_features": len(FEATURES),
                    "test_size": 0.20,
                },
            ),
        ]

        for run_name, pipe, params in clf_experiments:
            rec = run_and_track_experiment(
                run_name,
                "classification",
                pipe,
                params,
                X_tr_c,
                X_te_c,
                y_tr_c,
                y_te_c,
                str(feature_schema_path),
                fallback_dir,
            )
            tracked_records.append(rec)

        # ----------------------------------------------------
        # Track Regression Experiments (2 runs)
        # ----------------------------------------------------
        reg_experiments = [
            (
                "reg_linear_regression",
                Pipeline([
                    ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
                    ("model", LinearRegression()),
                ]),
                {
                    "algorithm": "LinearRegression",
                    "n_features": len(FEATURES),
                    "test_size": 0.20,
                },
            ),
            (
                "reg_random_forest_50_d12",
                Pipeline([
                    ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                    ("model", RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)),
                ]),
                {
                    "algorithm": "RandomForestRegressor",
                    "n_estimators": 50,
                    "max_depth": 12,
                    "n_features": len(FEATURES),
                    "test_size": 0.20,
                },
            ),
        ]

        for run_name, pipe, params in reg_experiments:
            rec = run_and_track_experiment(
                run_name,
                "regression",
                pipe,
                params,
                X_tr_r,
                X_te_r,
                y_tr_r,
                y_te_r,
                str(feature_schema_path),
                fallback_dir,
            )
            tracked_records.append(rec)

# ============================================================
# PART 5 OF 5: BEST MODEL SELECTION, REPORT & CLEAN LOGGING
# ============================================================

        results_df = pd.DataFrame(tracked_records)
        results_csv = OUTPUT_DIR / "task4_mlflow_experiment_summary.csv"
        results_df.to_csv(results_csv, index=False)
        logger.info("Saved experiment summary to '%s'.", results_csv)

        if not HAS_MLFLOW:
            summary_csv = fallback_dir / "experiment_runs.csv"
            results_df.to_csv(summary_csv, index=False)
            logger.info("Saved fallback experiment summary table to '%s'.", summary_csv)

        # Identify best runs programmatically
        clf_df = results_df[results_df["task_type"] == "classification"].sort_values("roc_auc", ascending=False)
        reg_df = results_df[results_df["task_type"] == "regression"].sort_values("r2_score", ascending=False)

        best_clf = clf_df.head(1).squeeze()
        best_reg = reg_df.head(1).squeeze()

        best_models = {
            "best_classification": {
                "run_name": best_clf["run_name"],
                "run_id": best_clf["run_id"],
                "roc_auc": float(best_clf["roc_auc"]),
                "f1_score": float(best_clf["f1_score"]),
            },
            "best_regression": {
                "run_name": best_reg["run_name"],
                "run_id": best_reg["run_id"],
                "r2_score": float(best_reg["r2_score"]),
                "mae": float(best_reg["mae"]),
            },
        }

        metadata_file = ARTIFACT_DIR / "best_models.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(best_models, f, indent=2)
        logger.info("Saved best model metadata to '%s'.", metadata_file)

        findings_file = ARTIFACT_DIR / "task4_findings.txt"
        report_content = (
            "TASK 4: ADVANCED AI TECHNIQUE — MLFLOW EXPERIMENT TRACKING\n"
            + "=" * 65 + "\n\n"
            f"Best Classification Run : {best_clf['run_name']} (ROC AUC = {best_clf['roc_auc']:.4f}, F1 = {best_clf['f1_score']:.4f})\n"
            f"Best Regression Run     : {best_reg['run_name']} (R2 = {best_reg['r2_score']:.4f}, MAE = {best_reg['mae']:.2f})\n\n"
            "1. WHY MLFLOW WAS SELECTED\n"
            "   - Connects directly to the MLOps lifecycle (versioning, reproducibility, and model registry).\n"
            "   - Enables structured side-by-side comparison across both classification (high_risk) and\n"
            "     regression (traffic_volume) experiments without manual spreadsheet tracking.\n\n"
            "2. HOW IT WAS IMPLEMENTED\n"
            "   - Configured a persistent SQLite tracking backend ('sqlite:///mlflow.db') under experiment\n"
            "     'Traffic_Mobility_MLOps_Experiment'.\n"
            "   - Executed 5 systematic runs (3 classification, 2 regression), logging hyperparameters,\n"
            "     evaluation metrics, feature schema JSON artifacts, and serialized scikit-learn pipelines.\n"
            "   - Programmatically selected top runs and exported metadata to 'artifacts/best_models.json'.\n\n"
            "3. VALUE ADDED\n"
            "   - Guarantees full auditability: every metric is tied to exact hyperparameters, feature schemas,\n"
            "     and serialized model binaries for downstream deployment simulation.\n\n"
            "4. LIMITATIONS\n"
            "   - Local SQLite tracking does not provide multi-user authentication or remote artifact storage\n"
            "     out of the box (which would require a remote MLflow tracking server + S3/Azure Blob store).\n"
            "   - Logging full model binaries per run increases local disk usage as hyperparameter grids grow.\n"
        )
        with open(findings_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Saved Task 4 findings and critical evaluation to '%s'.", findings_file)

        # Log Ranked Comparison Tables (Rule 5: zero print() statements)
        logger.info(
            "\nCLASSIFICATION RUNS (RANKED BY ROC AUC):\n%s",
            clf_df[["run_name", "algorithm", "accuracy", "precision", "recall", "f1_score", "roc_auc"]]
            .round(4)
            .to_string(index=False),
        )
        logger.info(
            "\nREGRESSION RUNS (RANKED BY R-SQUARED):\n%s",
            reg_df[["run_name", "algorithm", "mae", "r2_score"]]
            .round(4)
            .to_string(index=False),
        )

        if HAS_MLFLOW:
            logger.info("Inspect runs in MLflow UI via: mlflow ui --backend-store-uri %s", TRACKING_URI)

        logger.info("=" * 60)
        logger.info("PART 3 - TASK 4 MLFLOW EXPERIMENT TRACKING: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Task 4 aborted: required dataset not found.", exc_info=True)
        sys.exit(1)
    except KeyError:
        logger.error("Task 4 aborted: required column missing.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Task 4 aborted due to an unexpected error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()