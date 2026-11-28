# ============================================================
# PART 1 OF 2: IMPORTS, LOGGER, MLFLOW & FASTAPI ENDPOINTS
# ============================================================

import os
import sys
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Module-level logger only (handlers configured strictly in main())
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION & PATHS
# ============================================================

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Traffic_Mobility_MLOps_Experiment"

OUTPUT_DIR = Path("outputs")
ARTIFACT_DIR = Path("artifacts")

CLASSIFICATION_MODEL = OUTPUT_DIR / "random_forest_classifier.pkl"
REGRESSION_MODEL = OUTPUT_DIR / "random_forest_regressor.pkl"
SELECTED_FEATURES_FILE = ARTIFACT_DIR / "selected_features.json"
BEST_MODELS_FILE = ARTIFACT_DIR / "best_models.json"


# ============================================================
# 6.1 MODEL VERSION REGISTRY
# ============================================================

def build_model_registry() -> dict:
    """Construct model version registry combining historical versions and Task 4 metrics."""
    registry = {
        "classification": {
            "v1.0": {"model": "LogisticRegression", "roc_auc": 0.812, "stage": "Archived"},
            "v2.0": {"model": "RandomForestClassifier", "roc_auc": 0.894, "stage": "Production"},
        },
        "regression": {
            "v1.0": {"model": "LinearRegression", "r2_score": 0.731, "stage": "Archived"},
            "v2.0": {"model": "RandomForestRegressor", "r2_score": 0.882, "stage": "Production"},
        },
    }

    if BEST_MODELS_FILE.exists():
        try:
            with open(BEST_MODELS_FILE, "r", encoding="utf-8") as f:
                best_meta = json.load(f)
            registry["classification"]["v2.0"]["roc_auc"] = round(
                best_meta["best_classification"]["roc_auc"], 4
            )
            registry["regression"]["v2.0"]["r2_score"] = round(
                best_meta["best_regression"]["r2_score"], 4
            )
            registry["regression"]["v2.0"]["mae"] = round(
                best_meta["best_regression"]["mae"], 2
            )
        except Exception as exc:
            logger.warning("Could not merge Task 4 best_models.json into registry: %s", exc)

    return registry
    

# ============================================================
# 6.2 MLFLOW EXPERIMENT SUMMARY
# ============================================================

def get_experiment_summary() -> dict:
    """Query MLflow tracking store for experiment metadata and run counts."""
    if not HAS_MLFLOW:
        return {
            "tracking_enabled": False,
            "message": "MLflow not installed",
        }

    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            return {
                "tracking_enabled": True,
                "experiment_found": False,
                "experiment_name": EXPERIMENT_NAME,
            }

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return {
            "tracking_enabled": True,
            "experiment_found": True,
            "experiment_name": experiment.name,
            "experiment_id": experiment.experiment_id,
            "total_runs": int(len(runs)),
        }

    except Exception as exc:
        logger.warning("MLflow query encountered an issue: %s", exc)
        return {
            "tracking_enabled": False,
            "error": str(exc),
        }


# ============================================================
# LOAD MODELS & ALIGN INCOMING PAYLOAD SCHEMA
# ============================================================

def load_artifacts():
    """Load trained classifier, regressor, and expected feature list."""
    clf_model = None
    reg_model = None
    expected_features = None

    if CLASSIFICATION_MODEL.exists():
        with open(CLASSIFICATION_MODEL, "rb") as f:
            clf_model = pickle.load(f)

    if REGRESSION_MODEL.exists():
        with open(REGRESSION_MODEL, "rb") as f:
            reg_model = pickle.load(f)

    if SELECTED_FEATURES_FILE.exists():
        with open(SELECTED_FEATURES_FILE, "r", encoding="utf-8") as f:
            expected_features = json.load(f)

    return clf_model, reg_model, expected_features


classifier, regressor, EXPECTED_FEATURES = load_artifacts()


def prepare_api_dataframe(payload_dict: dict) -> pd.DataFrame:
    """
    Convert incoming API JSON payload into a DataFrame matching the exact
    column schema expected by the fitted ColumnTransformer pipeline.
    """
    row = dict(payload_dict)

    # Derive cyclical and capitalized columns if expected by the trained pipeline
    hr = row.get("hour", 12)
    dow = row.get("day_of_week", 0)

    row["Hour"] = hr
    row["Day_of_Week"] = dow
    row["Is_Weekend"] = row.get("is_weekend", int(dow >= 5))
    row["Hour_Sin"] = np.sin(2 * np.pi * hr / 24.0)
    row["Hour_Cos"] = np.cos(2 * np.pi * hr / 24.0)
    row["Day_Sin"] = np.sin(2 * np.pi * dow / 7.0)
    row["Day_Cos"] = np.cos(2 * np.pi * dow / 7.0)
    row["hour_sin"] = row["Hour_Sin"]
    row["hour_cos"] = row["Hour_Cos"]
    row["dow_sin"] = row["Day_Sin"]
    row["dow_cos"] = row["Day_Cos"]
    row["Temp_Celsius"] = row.get("temp", 285.0) - 273.15

    df_in = pd.DataFrame([row])

    if EXPECTED_FEATURES:
        for col in EXPECTED_FEATURES:
            if col not in df_in.columns:
                df_in[col] = np.nan
        df_in = df_in[EXPECTED_FEATURES]

    return df_in


# ============================================================
# 6.3 FASTAPI DEPLOYMENT MOCK-UP
# ============================================================

app = FastAPI(
    title="Traffic Mobility Prediction API",
    description="Capstone Deployment Simulation for Traffic Demand & Proxy Risk",
    version="2.0.0",
)


class TrafficFeatures(BaseModel):
    temp: float
    rain_1h: float
    snow_1h: float
    clouds_all: float
    hour: int
    day_of_week: int
    is_weekend: int


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Traffic Mobility API",
        "classifier_loaded": classifier is not None,
        "regressor_loaded": regressor is not None,
    }


@app.post("/predict_traffic")
def predict_traffic(payload: TrafficFeatures):
    if regressor is None:
        return {"error": "Regression model unavailable"}

    X = prepare_api_dataframe(payload.model_dump())
    prediction = float(regressor.predict(X)[0])
    return {"predicted_traffic_volume": round(prediction, 2)}


@app.post("/predict_risk")
def predict_risk(payload: TrafficFeatures):
    if classifier is None:
        return {"error": "Classification model unavailable"}

    X = prepare_api_dataframe(payload.model_dump())
    prediction = int(classifier.predict(X)[0])
    probability = float(classifier.predict_proba(X)[0][1])
    return {
        "high_risk_prediction": prediction,
        "risk_probability": round(probability, 4),
    }


@app.get("/experiment_tracking")
def experiment_tracking():
    return get_experiment_summary()

# ============================================================
# PART 2 OF 2: MONITORING, ALERTING, ARTIFACTS & MAIN
# ============================================================

# ============================================================
# 6.4 MONITORING (PREDICTION & FEATURE DRIFT)
# ============================================================

def monitor_prediction_drift() -> dict:
    """Compare production prediction MAE against baseline threshold (15% tolerance)."""
    baseline_mae = 420.0
    current_mae = 515.0

    if BEST_MODELS_FILE.exists():
        try:
            with open(BEST_MODELS_FILE, "r", encoding="utf-8") as f:
                best_meta = json.load(f)
            baseline_mae = round(float(best_meta["best_regression"]["mae"]), 2)
            # Simulate a 22% seasonal/weather error increase in production traffic
            current_mae = round(baseline_mae * 1.22, 2)
        except Exception as exc:
            logger.warning("Using default baseline MAE due to metadata read issue: %s", exc)

    drift_ratio = (current_mae - baseline_mae) / max(baseline_mae, 1e-6)

    return {
        "baseline_mae": baseline_mae,
        "current_mae": current_mae,
        "drift_ratio": round(drift_ratio, 4),
        "threshold": 0.15,
        "drift_detected": bool(drift_ratio > 0.15),
    }


def monitor_feature_drift() -> dict:
    """Compare live feature distribution means against training baselines (20% tolerance)."""
    monitored_features = {
        "traffic_volume": {"training_mean": 3260.0, "production_mean": 3910.0},
        "hour": {"training_mean": 11.4, "production_mean": 13.2},
        "rain_1h": {"training_mean": 0.13, "production_mean": 0.28},
    }

    results = {}
    for feature, values in monitored_features.items():
        drift = abs(values["production_mean"] - values["training_mean"]) / max(
            abs(values["training_mean"]), 1e-6
        )
        results[feature] = {
            "training_mean": values["training_mean"],
            "production_mean": values["production_mean"],
            "drift_ratio": round(drift, 4),
            "threshold": 0.20,
            "drift_detected": bool(drift > 0.20),
        }

    return results


# ============================================================
# 6.5 ALERTING DASHBOARD & ENDPOINT
# ============================================================

def build_dashboard() -> dict:
    """Aggregate drift checks into an actionable PASS / ALERT system status."""
    prediction_report = monitor_prediction_drift()
    feature_report = monitor_feature_drift()

    prediction_alert = prediction_report["drift_detected"]
    feature_alert = any(item["drift_detected"] for item in feature_report.values())

    status = "ALERT" if (prediction_alert or feature_alert) else "PASS"
    action_required = (
        "Requires investigation: trigger model retraining and feature pipeline audit."
        if status == "ALERT"
        else "Normal operation: no retraining required."
    )

    return {
        "system_status": status,
        "action_required": action_required,
        "prediction_drift_alert": prediction_alert,
        "feature_drift_alert": feature_alert,
        "prediction_details": prediction_report,
        "feature_details": feature_report,
    }


@app.get("/monitor")
def monitor():
    return build_dashboard()


# ============================================================
# ARTIFACT GENERATION & API SIMULATION
# ============================================================

def write_all_mlops_artifacts() -> dict:
    """Persist model registry, MLflow summary, monitoring reports, and alert dashboard."""
    registry = build_model_registry()
    with open(ARTIFACT_DIR / "model_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info("Saved model version registry: %s", ARTIFACT_DIR / "model_registry.json")

    exp_summary = get_experiment_summary()
    with open(ARTIFACT_DIR / "experiment_tracking_summary.json", "w", encoding="utf-8") as f:
        json.dump(exp_summary, f, indent=2)
    logger.info("Saved experiment tracking summary: %s", ARTIFACT_DIR / "experiment_tracking_summary.json")

    dashboard = build_dashboard()

    alert_dashboard = {
        "overall_status": dashboard["system_status"],
        "action_required": dashboard["action_required"],
        "prediction_alert": dashboard["prediction_drift_alert"],
        "feature_alert": dashboard["feature_drift_alert"],
    }
    with open(ARTIFACT_DIR / "alert_dashboard.json", "w", encoding="utf-8") as f:
        json.dump(alert_dashboard, f, indent=2)

    with open(ARTIFACT_DIR / "monitoring_dashboard.json", "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2)

    monitoring_df = pd.DataFrame(
        [
            {
                "Prediction_Drift_Alert": dashboard["prediction_drift_alert"],
                "Feature_Drift_Alert": dashboard["feature_drift_alert"],
                "System_Status": dashboard["system_status"],
                "Action_Required": dashboard["action_required"],
            }
        ]
    )
    monitoring_df.to_csv(ARTIFACT_DIR / "monitoring_dashboard.csv", index=False)
    logger.info("Saved monitoring & alert artifacts to '%s'.", ARTIFACT_DIR)

    report = (
        "TASK 6: MLOPS AND DEPLOYMENT SIMULATION\n"
        + "=" * 60 + "\n\n"
        "Framework: FastAPI\n"
        "Endpoints:\n"
        "  - GET  /                    (Health Check)\n"
        "  - POST /predict_traffic     (Traffic Volume Forecast)\n"
        "  - POST /predict_risk        (Proxy Accident-Risk Classification)\n"
        "  - GET  /experiment_tracking (MLflow Run Summary)\n"
        "  - GET  /monitor             (Drift Monitoring & PASS/ALERT Status)\n\n"
        f"Current Monitoring Status : {dashboard['system_status']}\n"
        f"Recommended Action        : {dashboard['action_required']}\n"
    )
    with open(ARTIFACT_DIR / "deployment_simulation.txt", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved deployment simulation report: %s", ARTIFACT_DIR / "deployment_simulation.txt")

    return dashboard


def simulate_api_requests() -> None:
    """Exercise all FastAPI endpoints via TestClient to verify live inference."""
    client = TestClient(app)

    health_resp = client.get("/")
    logger.info("API Health Check [GET /] -> %s", health_resp.json())

    sample_payload = {
        "temp": 286.15,
        "rain_1h": 2.5,
        "snow_1h": 0.0,
        "clouds_all": 85.0,
        "hour": 17,
        "day_of_week": 2,
        "is_weekend": 0,
    }

    traffic_resp = client.post("/predict_traffic", json=sample_payload)
    logger.info("API Traffic Inference [POST /predict_traffic] -> %s", traffic_resp.json())

    risk_resp = client.post("/predict_risk", json=sample_payload)
    logger.info("API Risk Inference [POST /predict_risk] -> %s", risk_resp.json())

    monitor_resp = client.get("/monitor")
    logger.info("API Monitoring Check [GET /monitor] -> Status=%s", monitor_resp.json().get("system_status"))


# ============================================================
# MAIN (ENTRY POINT - HANDLERS CONFIGURED HERE ONLY)
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

    try:
        logger.info("=" * 60)
        logger.info("PART 3 - TASK 6 MLOPS & DEPLOYMENT SIMULATION: STARTING")
        logger.info("=" * 60)

        OUTPUT_DIR.mkdir(exist_ok=True)
        ARTIFACT_DIR.mkdir(exist_ok=True)

        dashboard = write_all_mlops_artifacts()
        simulate_api_requests()

        if dashboard["system_status"] == "ALERT":
            logger.warning(
                "MLOPS ALERTING DASHBOARD: %s | %s",
                dashboard["system_status"],
                dashboard["action_required"],
            )
        else:
            logger.info(
                "MLOPS ALERTING DASHBOARD: %s | %s",
                dashboard["system_status"],
                dashboard["action_required"],
            )

        logger.info("FastAPI Launch Command: uvicorn task6_mlops_deployment:app --reload")
        logger.info("Swagger Documentation : http://127.0.0")
        logger.info("=" * 60)
        logger.info("PART 3 - TASK 6 MLOPS & DEPLOYMENT SIMULATION: COMPLETED")
        logger.info("=" * 60)

    except Exception:
        logger.error("Task 6 MLOps simulation failed due to an unexpected error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()