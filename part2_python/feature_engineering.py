import os
import sys
import logging
import pandas as pd
import numpy as np

# ============================================================
# LOGGER INITIALISATION (MODULE-LEVEL ONLY; NO HANDLERS HERE)
# ============================================================

logger = logging.getLogger(__name__)

# ============================================================
# FILE CONFIGURATION
# ============================================================

INPUT_FILE = "cleaned_traffic_data.csv"
OUTPUT_FILE = "engineered_traffic_features.csv"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """
    Load cleaned dataset and log initial dimensions.
    """
    logger.info("Loading cleaned dataset.")

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Source file not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        rows, cols = df.shape

        logger.info(
            f"Dataset shape BEFORE feature engineering: "
            f"{rows} rows, {cols} columns."
        )

        return df

    except FileNotFoundError:
        logger.error(
            "Feature engineering terminated: "
            "source file not found.",
            exc_info=True
        )
        raise

    except pd.errors.ParserError:
        logger.error(
            "Feature engineering terminated: "
            "CSV parsing failure.",
            exc_info=True
        )
        raise


# ============================================================
# TIME FEATURES
# ============================================================

def create_time_features(df):
    """
    Extract temporal indicators and cyclical encodings.
    """
    logger.info("Creating time-based features.")

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    df["Hour"] = df["date_time"].dt.hour
    df["Day_of_Week"] = df["date_time"].dt.dayofweek

    df["Is_Weekend"] = np.where(
        df["Day_of_Week"] >= 5,
        1,
        0
    )

    max_hour = 24
    days_in_week = 7

    logger.debug(
        f"Cyclical boundaries -> max_hour = {max_hour}, "
        f"days_in_week = {days_in_week}"
    )

    df["Hour_Sin"] = np.sin(
        2 * np.pi * df["Hour"] / max_hour
    )
    df["Hour_Cos"] = np.cos(
        2 * np.pi * df["Hour"] / max_hour
    )

    df["Day_Sin"] = np.sin(
        2 * np.pi * df["Day_of_Week"] / days_in_week
    )
    df["Day_Cos"] = np.cos(
        2 * np.pi * df["Day_of_Week"] / days_in_week
    )

    logger.info("Time feature extraction completed.")

    return df


# ============================================================
# WEATHER FEATURES
# ============================================================

def create_weather_features(df):
    """
    One-hot encode weather_main (while preserving original column)
    and create derived freezing precipitation hazard indicator.
    """
    logger.info("Encoding weather features.")

    weather_dummies = pd.get_dummies(
        df["weather_main"],
        prefix="Weather",
        dtype=int
    )
    df = pd.concat([df, weather_dummies], axis=1)

    df["Freezing_Precipitation_Hazard"] = np.where(
        (df["temp"] < 273.15)
        & ((df["rain_1h"] > 0) | (df["snow_1h"] > 0)),
        1,
        0
    )

    logger.debug(
        f"Freezing precipitation hazard hours identified: "
        f"{int(df['Freezing_Precipitation_Hazard'].sum())}"
    )

    logger.info("Weather feature engineering completed.")

    return df


# ============================================================
# NUMERICAL FEATURES
# ============================================================

def create_numerical_features(df):
    """
    Convert Kelvin to Celsius and scale continuous variables with
    division-by-zero protection.
    """
    logger.info("Scaling continuous variables.")

    # Convert Kelvin to Celsius
    df["Temp_Celsius"] = df["temp"] - 273.15

    # Standardise temperature (Z-score) with zero-variance guard
    temp_mean = df["temp"].mean()
    temp_std = df["temp"].std()

    logger.debug(
        f"Temperature Mean = {temp_mean:.3f}, "
        f"Std = {temp_std:.3f}"
    )

    if temp_std != 0:
        df["Scaled_Temp"] = (df["temp"] - temp_mean) / temp_std
    else:
        df["Scaled_Temp"] = 0.0
        logger.warning("Temperature standard deviation is 0; Scaled_Temp set to 0.0.")

    # Normalise cloud cover (Min-Max) with zero-range guard
    cloud_min = df["clouds_all"].min()
    cloud_max = df["clouds_all"].max()

    logger.debug(
        f"Cloud Cover Min = {cloud_min}, "
        f"Max = {cloud_max}"
    )

    if cloud_max != cloud_min:
        df["Normalised_Clouds"] = (
            (df["clouds_all"] - cloud_min) / (cloud_max - cloud_min)
        )
    else:
        df["Normalised_Clouds"] = 0.0
        logger.warning("Cloud cover min equals max; Normalised_Clouds set to 0.0.")

    logger.info("Continuous variable scaling completed.")

    return df


# ============================================================
# CONGESTION TARGET
# ============================================================

def create_congestion_target(df):
    """
    Create data-driven congestion target based on interquartile thresholds.
    """
    logger.info("Creating congestion category.")

    q1 = df["traffic_volume"].quantile(0.25)
    q3 = df["traffic_volume"].quantile(0.75)

    logger.debug(
        f"Congestion Thresholds: "
        f"Q1 = {q1:.2f}, "
        f"Q3 = {q3:.2f}"
    )

    # --------------------------------------------------------
    # Congestion logic:
    #
    # Low Flow      <= Q1
    # Moderate Flow > Q1 and < Q3
    # High Flow     >= Q3
    # --------------------------------------------------------

    conditions = [
        df["traffic_volume"] <= q1,
        (df["traffic_volume"] > q1) & (df["traffic_volume"] < q3),
        df["traffic_volume"] >= q3
    ]

    categories = [
        "Low Flow",
        "Moderate Flow",
        "High Flow"
    ]

    df["Congestion_Category"] = np.select(
        conditions,
        categories,
        default="Moderate Flow"
    )

    logger.info("Congestion category successfully generated.")

    return df


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df, output_file):
    """
    Log final dataset shape and export engineered features to CSV.
    """
    final_rows, final_cols = df.shape

    logger.info(
        f"Dataset shape AFTER feature engineering: "
        f"{final_rows} rows, {final_cols} columns."
    )

    df.to_csv(
        output_file,
        index=False
    )

    logger.info(
        f"Feature-engineered dataset saved to "
        f"{output_file}"
    )


# ============================================================
# MAIN (ENTRY POINT - HANDLERS CONFIGURED HERE ONLY)
# ============================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                "pipeline.log",
                mode="a",
                encoding="utf-8"
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        logger.info("=" * 60)
        logger.info("PART 2 - TASK 2 FEATURE ENGINEERING: STARTING")
        logger.info("=" * 60)

        df = load_dataset(INPUT_FILE)
        df = create_time_features(df)
        df = create_weather_features(df)
        df = create_numerical_features(df)
        df = create_congestion_target(df)
        save_dataset(df, OUTPUT_FILE)

        logger.info("=" * 60)
        logger.info("PART 2 - TASK 2 FEATURE ENGINEERING: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error(
            "Feature engineering terminated: "
            "source file not found.",
            exc_info=True
        )
        sys.exit(1)

    except pd.errors.ParserError:
        logger.error(
            "Feature engineering terminated: "
            "CSV parsing failure.",
            exc_info=True
        )
        sys.exit(1)

    except ValueError:
        logger.error(
            "Feature engineering terminated: "
            "data validation failure.",
            exc_info=True
        )
        sys.exit(1)

    except Exception as error:
        logger.error(
            f"Feature engineering terminated due "
            f"to an unexpected error: {error}",
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()