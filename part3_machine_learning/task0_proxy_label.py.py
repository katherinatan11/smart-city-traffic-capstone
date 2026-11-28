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
# OUTPUT & FILE CONFIGURATION
# ============================================================

OUTPUT_DIR = "outputs"
INPUT_FILE = "engineered_traffic_features.csv"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_traffic_accident_proxy.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "proxy_label_summary.csv"
)

CLASS_DISTRIBUTION_FILE = os.path.join(
    OUTPUT_DIR,
    "class_distribution.csv"
)

DOCUMENTATION_FILE = os.path.join(
    OUTPUT_DIR,
    "proxy_label_documentation.txt"
)

# ============================================================
# WEATHER DEFINITIONS
# ============================================================

SEVERE_WEATHER = [
    "Squall",
    "Thunderstorm",
    "Snow",
    "Rain",
    "Drizzle"
]

LOW_VISIBILITY_WEATHER = [
    "Fog",
    "Haze",
    "Mist",
    "Smoke"
]


# ============================================================
# VALIDATION
# ============================================================

def validate_columns(df, required_columns):
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {', '.join(missing)}"
        )


# ============================================================
# CONGESTION CATEGORIES
# ============================================================

def create_congestion_category(df):
    q1, q2, q3 = (
        df["traffic_volume"]
        .quantile([0.25, 0.50, 0.75])
        .values
    )

    logger.debug(
        f"Traffic Quartiles | "
        f"Q1={q1:.0f}, "
        f"Median={q2:.0f}, "
        f"Q3={q3:.0f}"
    )

    def bucket(value):
        if pd.isna(value):
            return "Unknown"
        elif value <= q1:
            return "Low"
        elif value <= q2:
            return "Medium"
        elif value <= q3:
            return "High"
        else:
            return "Severe"

    df["congestion_category"] = (
        df["traffic_volume"]
        .apply(bucket)
    )

    logger.info("4-tier congestion_category generated successfully.")
    return df


# ============================================================
# WEATHER RECONSTRUCTION
# ============================================================

def reconstruct_weather_column(df):
    if "weather_main" in df.columns:
        return df

    weather_cols = [
        col
        for col in df.columns
        if col.startswith("Weather_")
    ]

    if not weather_cols:
        raise KeyError(
            "weather_main missing and "
            "Weather_* columns unavailable."
        )

    df["weather_main"] = (
        df[weather_cols]
        .idxmax(axis=1)
        .str.replace(
            "Weather_",
            "",
            regex=False
        )
    )

    logger.warning("Reconstructed missing 'weather_main' column from one-hot Weather_* indicators.")
    return df


# ============================================================
# DOCUMENTATION OUTPUT
# ============================================================

def generate_proxy_documentation():
    documentation = """PROXY ACCIDENT-RISK LABEL

IMPORTANT DISCLOSURE

No real accident dataset was used.

The Metro Interstate Traffic Volume dataset
contains traffic and weather observations
but does not contain accident outcomes.

A documented proxy accident-risk label
was generated using:

HIGH CONGESTION
AND
(SEVERE WEATHER OR LOW VISIBILITY)

Severe Weather:
- Rain
- Snow
- Thunderstorm
- Squall
- Drizzle

Low Visibility:
- Fog
- Haze
- Mist
- Smoke

Predictions represent elevated
traffic risk and not actual
accident probability.
"""

    with open(
        DOCUMENTATION_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(documentation)


# ============================================================
# PROXY LABEL
# ============================================================

def create_proxy_risk_label(df):
    df["is_low_visibility"] = np.where(
        df["weather_main"].isin(
            LOW_VISIBILITY_WEATHER
        ),
        1,
        0
    )

    high_congestion = (
        df["congestion_category"]
        .isin(["High", "Severe"])
    )

    risky_weather = (
        df["weather_main"].isin(SEVERE_WEATHER)
        | (df["is_low_visibility"] == 1)
    )

    df["high_risk"] = (
        high_congestion &
        risky_weather
    ).astype(int)

    high_risk_count = int(df["high_risk"].sum())
    low_risk_count = len(df) - high_risk_count
    risk_pct = (high_risk_count / len(df)) * 100

    logger.info(
        f"Proxy label 'high_risk' generated | "
        f"High Risk: {high_risk_count} ({risk_pct:.2f}%), "
        f"Normal Risk: {low_risk_count} ({100 - risk_pct:.2f}%)"
    )

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Records",
                "High Risk Records",
                "Low Risk Records",
                "High Risk %",
                "Low Risk %"
            ],
            "Value": [
                len(df),
                high_risk_count,
                low_risk_count,
                round(risk_pct, 2),
                round(100 - risk_pct, 2)
            ]
        }
    )

    summary_df.to_csv(SUMMARY_FILE, index=False)

    class_distribution = (
        df["high_risk"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    )

    class_distribution.columns = [
        "Class",
        "Percentage"
    ]

    class_distribution.to_csv(CLASS_DISTRIBUTION_FILE, index=False)

    return df


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
        logger.info("PART 3 - PROXY ACCIDENT-RISK LABEL: STARTING")
        logger.info("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(
                f"Input file not found: {INPUT_FILE}"
            )

        df = pd.read_csv(INPUT_FILE)

        validate_columns(df, ["traffic_volume"])
        df = reconstruct_weather_column(df)
        df = create_congestion_category(df)
        df = create_proxy_risk_label(df)
        generate_proxy_documentation()

        df.to_csv(OUTPUT_FILE, index=False)

        logger.info(f"Saved {OUTPUT_FILE}")
        logger.info(f"Saved {SUMMARY_FILE}")
        logger.info(f"Saved {CLASS_DISTRIBUTION_FILE}")
        logger.info(f"Saved {DOCUMENTATION_FILE}")
        logger.info(f"Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")
        logger.info("=" * 60)
        logger.info("PART 3 - PROXY ACCIDENT-RISK LABEL: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Processing failed: input CSV file not found.", exc_info=True)
        sys.exit(1)

    except KeyError:
        logger.error("Processing failed: required column missing.", exc_info=True)
        sys.exit(1)

    except Exception:
        logger.error("Processing failed due to an unexpected error.", exc_info=True)
        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()