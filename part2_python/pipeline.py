import sys
import os
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

CSV_FILENAME = "Metro_Interstate_Traffic_Volume"

OUTPUT_FILENAME = "cleaned_traffic_data.csv"

EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):
    """
    Load CSV file with robust exception handling.
    """
    logger.info("Loading raw CSV file.")

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"CSV file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        rows, cols = df.shape

        logger.info(
            f"Raw file successfully loaded "
            f"({rows} rows, {cols} columns)."
        )

        return df

    except FileNotFoundError:
        logger.error(
            "File loading failed: CSV file not found.",
            exc_info=True
        )
        raise

    except pd.errors.ParserError:
        logger.error(
            "File loading failed: CSV parsing error.",
            exc_info=True
        )
        raise

    except OSError:
        logger.error(
            "File loading failed: Operating system I/O error.",
            exc_info=True
        )
        raise

# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_schema(df):
    """
    Ensure all required columns are present.
    """
    logger.info("Validating schema.")

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        logger.error(
            f"Schema validation failed. "
            f"Missing columns: {missing_columns}"
        )
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logger.info("Schema validation successful.")


# ============================================================
# STANDARDISE CATEGORICAL VALUES
# ============================================================

def standardise_categories(df):
    """
    Standardise whitespace, casing, and missing values in categorical columns.
    """
    logger.info("Standardising categorical values.")

    categorical_columns = [
        "holiday",
        "weather_main",
        "weather_description"
    ]

    before = df[categorical_columns].copy()

    for column in categorical_columns:
        df[column] = (
            df[column]
            .fillna("None")
            .astype(str)
            .str.strip()
            .str.title()
        )

    modified_rows = (before.fillna("None") != df[categorical_columns]).any(axis=1).sum()

    if modified_rows > 0:
        logger.warning(
            f"Modified {modified_rows} rows during categorical standardisation "
            f"(whitespace, casing, or missing values)."
        )

    logger.info("Categorical value standardisation completed.")

    return df


# ============================================================
# DATETIME PARSING
# ============================================================

def parse_datetime(df):
    """
    Parse and validate date_time column and extract Month for imputation.
    """
    logger.info("Parsing and validating datetime values.")

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    invalid_dates = df["date_time"].isna().sum()

    if invalid_dates > 0:
        df = df.dropna(subset=["date_time"]).reset_index(drop=True)
        logger.warning(
            f"Dropped {invalid_dates} rows due to "
            f"invalid datetime values."
        )

    df["Month"] = df["date_time"].dt.month

    logger.info("Datetime parsing completed successfully.")

    return df


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(df):
    """
    Identify and drop exact duplicate rows.
    """
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.warning(
            f"Dropped {duplicate_count} duplicate rows."
        )
    else:
        logger.info("No duplicate rows detected.")

    return df


# ============================================================
# MONTHLY MEDIAN IMPUTATION
# ============================================================

def impute_by_monthly_median(
    df,
    column_name,
    invalid_mask,
    reason
):
    """
    Impute invalid values using monthly medians via explicit month loop.
    """
    affected_rows = invalid_mask.sum()

    if affected_rows == 0:
        logger.info(f"No invalid values detected in {column_name}.")
        return df

    for month in sorted(df["Month"].unique()):
        valid_values = df.loc[
            (df["Month"] == month)
            & (~invalid_mask),
            column_name
        ]

        if not valid_values.empty:
            monthly_median = valid_values.median()
            logger.debug(
                f"Month {month} median calculated for {column_name}: {monthly_median:.2f}"
            )
            df.loc[
                (df["Month"] == month)
                & invalid_mask,
                column_name
            ] = monthly_median

    logger.warning(
        f"Imputed {affected_rows} rows in "
        f"{column_name}. Reason: {reason}"
    )

    return df


# ============================================================
# OUTLIER HANDLING
# ============================================================

def clean_outliers(df):
    """
    Detect and impute physically impossible values across numeric features.
    """
    logger.info("Checking temperature outliers.")
    temp_invalid = df["temp"] <= 0
    df = impute_by_monthly_median(
        df,
        "temp",
        temp_invalid,
        "Temperature <= 0 Kelvin"
    )

    logger.info("Checking rainfall outliers.")
    rain_invalid = (df["rain_1h"] < 0) | (df["rain_1h"] > 9000)
    df = impute_by_monthly_median(
        df,
        "rain_1h",
        rain_invalid,
        "Rainfall outside valid range"
    )

    logger.info("Checking snowfall outliers.")
    snow_invalid = df["snow_1h"] < 0
    df = impute_by_monthly_median(
        df,
        "snow_1h",
        snow_invalid,
        "Negative snowfall detected"
    )

    logger.info("Checking cloud cover values.")
    cloud_invalid = (df["clouds_all"] < 0) | (df["clouds_all"] > 100)
    df = impute_by_monthly_median(
        df,
        "clouds_all",
        cloud_invalid,
        "Cloud cover outside 0-100%"
    )

    logger.info("Checking traffic volume values.")
    traffic_invalid = df["traffic_volume"] < 0
    df = impute_by_monthly_median(
        df,
        "traffic_volume",
        traffic_invalid,
        "Negative traffic volume detected"
    )

    return df


# ============================================================
# SAVE DATA
# ============================================================

def save_clean_data(df, output_file):
    """
    Save cleaned dataset to CSV.
    """
    df.to_csv(
        output_file,
        index=False
    )

    logger.info(f"Cleaned dataset saved to {output_file}")


# ============================================================
# MAIN (ENTRY POINT - HANDLERS CONFIGURED HERE ONLY)
# ============================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        logger.info("=" * 60)
        logger.info("PART 2 - TASK 1 DATA PIPELINE: STARTING")
        logger.info("=" * 60)

        df = load_data(CSV_FILENAME)
        validate_schema(df)
        df = standardise_categories(df)
        df = parse_datetime(df)
        df = remove_duplicates(df)
        df = clean_outliers(df)
        save_clean_data(df, OUTPUT_FILENAME)

        logger.info("=" * 60)
        logger.info("PART 2 - TASK 1 DATA PIPELINE: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error(
            "Pipeline terminated because the input file could not be found.",
            exc_info=True
        )
        sys.exit(1)

    except pd.errors.ParserError:
        logger.error(
            "Pipeline terminated due to a CSV parsing error.",
            exc_info=True
        )
        sys.exit(1)

    except ValueError:
        logger.error(
            "Pipeline terminated due to a data or schema validation error.",
            exc_info=True
        )
        sys.exit(1)

    except Exception:
        logger.error(
            "Pipeline terminated due to an unexpected error.",
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()