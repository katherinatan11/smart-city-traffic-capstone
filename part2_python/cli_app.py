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

DATA_SOURCE = "engineered_traffic_features.csv"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load the feature-engineered dataset for CLI queries.
    """
    logger.info("Loading feature-engineered dataset.")

    if not os.path.exists(DATA_SOURCE):
        logger.error(f"Dataset not found: {DATA_SOURCE}")
        raise FileNotFoundError(DATA_SOURCE)

    df = pd.read_csv(DATA_SOURCE)

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    logger.info(
        f"Dataset loaded successfully "
        f"({df.shape[0]} rows, {df.shape[1]} columns)."
    )

    return df


# ============================================================
# COMMAND 1: QUERY SPECIFIC DATETIME
# ============================================================

def query_specific_datetime(df, target_str):
    """
    Look up traffic and weather conditions for an exact timestamp.
    """
    logger.info(
        f"Command Invoked: datetime-lookup | target='{target_str}'"
    )

    if not target_str.strip():
        logger.error("Empty datetime string entered. Expected format: YYYY-MM-DD HH:MM:SS.")
        return

    try:
        target_datetime = pd.to_datetime(target_str.strip())

        if pd.isna(target_datetime):
            raise ValueError("Parsed timestamp is NaT.")

        record = df[df["date_time"] == target_datetime]

        if record.empty:
            print("\nNo matching data found for that exact date/time.")
            return

        if "weather_main" in record.columns:
            weather_type = record["weather_main"].iloc[0]
        else:
            weather_cols = [
                col for col in record.columns
                if col.startswith("Weather_")
            ]
            weather_type = "Unknown"
            for col in weather_cols:
                if record[col].iloc[0] == 1:
                    weather_type = col.replace("Weather_", "")
                    break

        print("\nTraffic Information")
        print("-" * 40)
        print(f"Date/Time:           {record['date_time'].iloc[0]}")
        print(f"Traffic Volume:      {record['traffic_volume'].iloc[0]}")
        print(f"Congestion Category: {record['Congestion_Category'].iloc[0]}")
        print(f"Weather Condition:   {weather_type}")

    except (ValueError, TypeError):
        logger.error(
            f"Invalid date format entered: '{target_str}'. Expected YYYY-MM-DD HH:MM:SS."
        )


# ============================================================
# COMMAND 2: HIGH TRAFFIC PERIODS
# ============================================================

def identify_high_traffic_periods(df, threshold=5500):
    """
    Identify hours where traffic volume exceeds a specified threshold.
    """
    logger.info(
        f"Command Invoked: high-traffic | threshold={threshold}"
    )

    if threshold < 0:
        logger.error(f"Invalid threshold entered: {threshold}. Threshold must be >= 0.")
        return

    results = df[df["traffic_volume"] > threshold]

    logger.debug(f"Filtered {len(results)} records exceeding {threshold} vehicles/hr.")

    print("\nHigh Traffic Analysis")
    print("-" * 40)
    print(f"Records above {threshold}: {len(results)}")

    if not results.empty:
        peak_hour = results["Hour"].mode().iloc[0]
        print(f"Most Common Peak Hour: {peak_hour:02d}:00")


# ============================================================
# COMMAND 3: WEEKDAY VS WEEKEND
# ============================================================

def compare_weekday_weekend_traffic(df):
    """
    Compare average traffic volume between weekdays and weekends.
    """
    logger.info("Command Invoked: weekday-weekend-comparison")

    weekday_avg = df[df["Is_Weekend"] == 0]["traffic_volume"].mean()
    weekend_avg = df[df["Is_Weekend"] == 1]["traffic_volume"].mean()

    print("\nWeekday vs Weekend")
    print("-" * 40)
    print(f"Weekday Average: {weekday_avg:.0f} vehicles/hr")
    print(f"Weekend Average: {weekend_avg:.0f} vehicles/hr")


# ============================================================
# COMMAND 4: TRAVEL RECOMMENDATIONS
# ============================================================

def recommend_travel_periods(df):
    """
    Display the top 3 lowest-traffic hours of the day.
    """
    logger.info("Command Invoked: recommend-travel")

    hourly_avg = df.groupby("Hour")["traffic_volume"].mean()
    best_hours = hourly_avg.sort_values().head(3)

    print("\nRecommended Low-Traffic Travel Hours")
    print("-" * 40)
    for hour, avg_vol in best_hours.items():
        print(f"{hour:02d}:00 (Avg Volume: {avg_vol:.0f} vehicles/hr)")


# ============================================================
# COMMAND 5: WEATHER LOOKUP
# ============================================================

def retrieve_weather_traffic_info(df, weather_query):
    """
    Retrieve average traffic volume for a given weather category.
    """
    logger.info(
        f"Command Invoked: weather-lookup | weather='{weather_query}'"
    )

    clean_query = weather_query.strip().title()
    target_col = f"Weather_{clean_query}"

    if not clean_query or target_col not in df.columns:
        available = [c.replace("Weather_", "") for c in df.columns if c.startswith("Weather_")]
        logger.error(
            f"Unknown weather type entered: '{weather_query}'. Available options: {', '.join(available)}."
        )
        return

    subset = df[df[target_col] == 1]

    print(f"\nWeather Report: {clean_query}")
    print("-" * 40)
    print(f"Matching Records: {len(subset)}")
    print(f"Average Traffic:  {subset['traffic_volume'].mean():.0f} vehicles/hr")


# ============================================================
# MENU LOOP
# ============================================================

def application_loop(df):
    """
    Interactive CLI menu loop with input validation.
    """
    while True:
        print("\n" + "=" * 40)
        print("Smart City Traffic Analytics Menu")
        print("=" * 40)
        print("1. Traffic at Date/Time")
        print("2. High Traffic Periods")
        print("3. Weekday vs Weekend")
        print("4. Travel Recommendation")
        print("5. Weather Lookup")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            target = input("Enter datetime (YYYY-MM-DD HH:MM:SS, e.g., 2017-07-15 16:00:00): ")
            query_specific_datetime(df, target)

        elif choice == "2":
            threshold = input("Threshold (press Enter for default=5500): ").strip()
            if threshold == "":
                identify_high_traffic_periods(df)
            else:
                try:
                    identify_high_traffic_periods(df, int(threshold))
                except ValueError:
                    logger.error(f"Invalid threshold entered: '{threshold}'. Expected a whole number.")

        elif choice == "3":
            compare_weekday_weekend_traffic(df)

        elif choice == "4":
            recommend_travel_periods(df)

        elif choice == "5":
            weather = input("Enter weather type (e.g., Clear, Rain, Snow, Clouds): ")
            retrieve_weather_traffic_info(df, weather)

        elif choice == "6":
            logger.info("Command Invoked: exit | Closing application.")
            break

        else:
            logger.error(f"Invalid menu option selected: '{choice}'. Please enter a number between 1 and 6.")


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
        logger.info("PART 2 - TASK 4 CLI APPLICATION: STARTING")
        logger.info("=" * 60)

        df = load_dataset()
        application_loop(df)

        logger.info("=" * 60)
        logger.info("PART 2 - TASK 4 CLI APPLICATION: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Application terminated: required dataset file is missing.")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C). Exiting cleanly.")
        sys.exit(0)

    except Exception as error:
        logger.error(f"Application terminated due to an unexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()