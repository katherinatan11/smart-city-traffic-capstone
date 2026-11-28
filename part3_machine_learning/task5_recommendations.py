# ============================================================
# PART 1 OF 2: IMPORTS, LOGGER, PATHS & SCENARIO SCORING
# ============================================================

import os
import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Module-level logger only (handlers configured strictly in main())
logger = logging.getLogger(__name__)

# Directory & File Configuration
OUTPUT_DIR = Path("outputs")
ARTIFACT_DIR = Path("artifacts")

DATA_SOURCE = OUTPUT_DIR / "ml_traffic_accident_proxy.csv"
RF_REG_MODEL_FILE = OUTPUT_DIR / "random_forest_regressor.pkl"
RF_CLF_MODEL_FILE = OUTPUT_DIR / "random_forest_classifier.pkl"
SELECTED_FEATURES_FILE = ARTIFACT_DIR / "selected_features.json"

RECOMMENDATIONS_CSV = OUTPUT_DIR / "task5_travel_recommendations.csv"
RECOMMENDATIONS_TXT = ARTIFACT_DIR / "task5_findings.txt"


def format_hour_window(hour: int) -> str:
    """Format an integer hour (0-23) into a readable 1-hour AM/PM travel window."""
    start_dt = pd.Timestamp(year=2025, month=1, day=1, hour=int(hour))
    end_dt = start_dt + pd.Timedelta(hours=1)
    return f"{start_dt.strftime('%I:%M %p').lstrip('0')} and {end_dt.strftime('%I:%M %p').lstrip('0')}"


def load_models_and_features():
    """Load trained regression & classification pipelines from Task 1 if available."""
    reg_model = None
    clf_model = None
    feature_cols = None

    if RF_REG_MODEL_FILE.exists():
        with open(RF_REG_MODEL_FILE, "rb") as f:
            reg_model = pickle.load(f)
        logger.info("Loaded trained regression model: %s", RF_REG_MODEL_FILE)
    else:
        logger.warning(
            "Regression model '%s' not found; falling back to historical traffic averages.",
            RF_REG_MODEL_FILE,
        )

    if RF_CLF_MODEL_FILE.exists():
        with open(RF_CLF_MODEL_FILE, "rb") as f:
            clf_model = pickle.load(f)
        logger.info("Loaded trained classification model: %s", RF_CLF_MODEL_FILE)
    else:
        logger.warning(
            "Classification model '%s' not found; falling back to historical risk rates.",
            RF_CLF_MODEL_FILE,
        )

    if SELECTED_FEATURES_FILE.exists():
        with open(SELECTED_FEATURES_FILE, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        logger.debug(
            "Loaded %d expected model feature names from '%s'.",
            len(feature_cols),
            SELECTED_FEATURES_FILE,
        )

    return reg_model, clf_model, feature_cols


def evaluate_hourly_profiles(df: pd.DataFrame, reg_model, clf_model, feature_cols) -> pd.DataFrame:
    """
    Combine historical traffic observations with ML model predictions across
    (Day_Type, Weather_Condition, Hour) combinations.
    """
    work = df.copy()

    # Standardise Hour, Is_Weekend, and weather_main columns
    if "Hour" not in work.columns and "hour" in work.columns:
        work["Hour"] = work["hour"]
    elif "Hour" not in work.columns and "date_time" in work.columns:
        work["Hour"] = pd.to_datetime(work["date_time"], errors="coerce").dt.hour

    if "Is_Weekend" not in work.columns and "is_weekend" in work.columns:
        work["Is_Weekend"] = work["is_weekend"]
    elif "Is_Weekend" not in work.columns and "date_time" in work.columns:
        work["Is_Weekend"] = (pd.to_datetime(work["date_time"], errors="coerce").dt.dayofweek >= 5).astype(int)

    work["Day_Type"] = np.where(work["Is_Weekend"] == 1, "Weekend", "Weekday")

    if "weather_main" not in work.columns:
        weather_ohe = [c for c in work.columns if c.startswith("Weather_")]
        if weather_ohe:
            work["weather_main"] = (
                work[weather_ohe].idxmax(axis=1).str.replace("Weather_", "", regex=False)
            )
        else:
            work["weather_main"] = "Clear"

    # Attach ML predictions when trained models & feature schema exist
    if reg_model is not None and clf_model is not None and feature_cols is not None:
        available_feats = [c for c in feature_cols if c in work.columns]
        if len(available_feats) == len(feature_cols):
            work["Predicted_Volume"] = reg_model.predict(work[feature_cols])
            work["Predicted_Risk_Prob"] = clf_model.predict_proba(work[feature_cols])[:, 1]
            logger.info(
                "Scored %d records using trained Random Forest Regression & Classification models.",
                len(work),
            )
        else:
            missing = set(feature_cols) - set(available_feats)
            logger.warning(
                "Feature mismatch (%s); using empirical traffic metrics.", missing
            )
            work["Predicted_Volume"] = work["traffic_volume"]
            work["Predicted_Risk_Prob"] = work["high_risk"].astype(float)
    else:
        work["Predicted_Volume"] = work["traffic_volume"]
        work["Predicted_Risk_Prob"] = work["high_risk"].astype(float)

    # Aggregate by Day_Type, weather_main, and Hour
    profile = (
        work.groupby(["Day_Type", "weather_main", "Hour"], observed=True)
        .agg(
            Observation_Count=("traffic_volume", "size"),
            Historical_Avg_Volume=("traffic_volume", "mean"),
            ML_Predicted_Volume=("Predicted_Volume", "mean"),
            ML_Risk_Probability=("Predicted_Risk_Prob", "mean"),
        )
        .reset_index()
    )

    # Composite Travel Suitability Penalty (lower = better: 80% predicted volume + 20% risk penalty)
    vol_norm = profile["ML_Predicted_Volume"] / max(profile["ML_Predicted_Volume"].max(), 1.0)
    profile["Suitability_Penalty"] = 0.80 * vol_norm + 0.20 * profile["ML_Risk_Probability"]

    return profile

# ============================================================
# PART 2 OF 2: RECOMMENDATION GENERATOR & MAIN PIPELINE
# ============================================================

def generate_scenario_recommendations(profile: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Day_Type, Weather_Condition) scenario, identify:
    1. Best practical daytime window (06:00 - 21:00) for standard commuters
    2. Best overall off-peak window (00:00 - 23:00) for flexible logistics/freight
    3. Peak congestion window to avoid
    and generate a plain-language travel recommendation.
    """
    recommendations = []

    scenarios = profile.groupby(["Day_Type", "weather_main"], observed=True)

    for (day_type, weather), group in scenarios:
        # Filter out ultra-rare weather-hour combinations (< 5 observations) if sufficient rows exist
        reliable = group[group["Observation_Count"] >= 5]
        if reliable.empty:
            reliable = group

        # Best overall 24-hour window (lowest suitability penalty)
        best_overall = reliable.sort_values("Suitability_Penalty", ascending=True).head(1).squeeze()

        # Best practical daytime window (06:00 to 21:00)
        daytime = reliable[(reliable["Hour"] >= 6) & (reliable["Hour"] <= 21)]
        if daytime.empty:
            daytime = reliable
        best_daytime = daytime.sort_values("Suitability_Penalty", ascending=True).head(1).squeeze()

        # Worst peak congestion window to avoid
        worst_peak = reliable.sort_values("Suitability_Penalty", ascending=False).head(1).squeeze()

        daytime_window = format_hour_window(best_daytime["Hour"])
        offpeak_window = format_hour_window(best_overall["Hour"])
        avoid_window = format_hour_window(worst_peak["Hour"])

        weather_lower = str(weather).lower()
        advisory_note = ""
        if any(k in weather_lower for k in ("rain", "snow", "thunder", "squall", "fog", "mist")):
            advisory_note = (
                f" Because {weather_lower} conditions elevate collision risk "
                f"(predicted risk score {best_daytime['ML_Risk_Probability'] * 100:.1f}%), "
                f"allow extra braking distance and buffer time."
            )

        plain_text = (
            f"For a {day_type.lower()} journey during {weather_lower} weather, consider travelling "
            f"between {daytime_window}, when predicted traffic volumes are lower "
            f"(~{best_daytime['ML_Predicted_Volume']:.0f} vehicles/hr vs. peak "
            f"~{worst_peak['ML_Predicted_Volume']:.0f} vehicles/hr between {avoid_window}). "
            f"For flexible or freight trips, the quietest overall window is between {offpeak_window} "
            f"(~{best_overall['ML_Predicted_Volume']:.0f} vehicles/hr).{advisory_note}"
        )

        recommendations.append(
            {
                "Day_Type": day_type,
                "Weather_Condition": weather,
                "Recommended_Daytime_Window": daytime_window,
                "Daytime_Pred_Volume": round(float(best_daytime["ML_Predicted_Volume"]), 0),
                "Recommended_OffPeak_Window": offpeak_window,
                "OffPeak_Pred_Volume": round(float(best_overall["ML_Predicted_Volume"]), 0),
                "Avoid_Peak_Window": avoid_window,
                "Peak_Pred_Volume": round(float(worst_peak["ML_Predicted_Volume"]), 0),
                "Daytime_Risk_Prob": round(float(best_daytime["ML_Risk_Probability"]), 4),
                "Plain_Language_Recommendation": plain_text,
            }
        )

    return pd.DataFrame(recommendations).sort_values(
        by=["Day_Type", "Weather_Condition"]
    ).reset_index(drop=True)


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
        logger.info("PART 3 - TASK 5 TRAFFIC RECOMMENDATION SYSTEM: STARTING")
        logger.info("=" * 60)

        OUTPUT_DIR.mkdir(exist_ok=True)
        ARTIFACT_DIR.mkdir(exist_ok=True)

        if not DATA_SOURCE.exists():
            raise FileNotFoundError(
                f"Inbound data error: '{DATA_SOURCE}' missing. Execute Proxy Label script first."
            )

        df = pd.read_csv(DATA_SOURCE)
        logger.info("Loaded dataset '%s' with shape %s.", DATA_SOURCE, df.shape)

        if "traffic_volume" not in df.columns:
            raise KeyError("Required column 'traffic_volume' missing from dataset.")

        reg_model, clf_model, feature_cols = load_models_and_features()
        hourly_profile = evaluate_hourly_profiles(df, reg_model, clf_model, feature_cols)
        rec_df = generate_scenario_recommendations(hourly_profile)

        # Save CSV recommendations
        rec_df.to_csv(RECOMMENDATIONS_CSV, index=False)
        logger.info("Saved scenario travel recommendations to '%s'.", RECOMMENDATIONS_CSV)

        # Save plain-language findings artifact
        with open(RECOMMENDATIONS_TXT, "w", encoding="utf-8") as f:
            f.write("TASK 5: SMART CORRIDOR TRAVEL TIMING RECOMMENDATIONS\n")
            f.write("=" * 65 + "\n\n")
            for _, row in rec_df.iterrows():
                f.write(f"[{row['Day_Type']} | {row['Weather_Condition']}]\n")
                f.write(f"  {row['Plain_Language_Recommendation']}\n\n")

        logger.info("Saved plain-language recommendations to '%s'.", RECOMMENDATIONS_TXT)

        # Log key scenario recommendations cleanly (Rule 5: zero print() statements)
        key_weather = ["Clear", "Clouds", "Rain", "Snow", "Mist"]
        sample_recs = rec_df[rec_df["Weather_Condition"].isin(key_weather)]
        if sample_recs.empty:
            sample_recs = rec_df.head(8)

        logger.info("\nSAMPLE PLAIN-LANGUAGE TRAVEL RECOMMENDATIONS:")
        for _, row in sample_recs.iterrows():
            logger.info(
                "[%s - %s] %s",
                row["Day_Type"],
                row["Weather_Condition"],
                row["Plain_Language_Recommendation"],
            )

        logger.info("=" * 60)
        logger.info("PART 3 - TASK 5 TRAFFIC RECOMMENDATION SYSTEM: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Task 5 aborted: required dataset not found.", exc_info=True)
        sys.exit(1)
    except KeyError:
        logger.error("Task 5 aborted: required column missing.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Task 5 aborted due to an unexpected error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()