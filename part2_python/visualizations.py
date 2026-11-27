import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LOGGER INITIALISATION (MODULE-LEVEL ONLY; NO HANDLERS HERE)
# ============================================================

logger = logging.getLogger(__name__)

# ============================================================
# FILE CONFIGURATION
# ============================================================

INPUT_FILE = "engineered_traffic_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    """
    Load feature-engineered dataset for visualisation.
    """
    logger.info("Loading feature-engineered dataset.")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

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
# VISUALISATION 1: TRAFFIC DEMAND BY HOUR
# ============================================================

def create_hourly_traffic_chart(df):
    """
    Line chart showing average traffic demand across the 24-hour day.
    """
    logger.info("Generating Figure 1: Traffic Demand by Hour")

    hourly_avg = (
        df.groupby("Hour")["traffic_volume"]
        .mean()
        .reset_index()
    )

    logger.debug(
        f"Peak hourly average: {hourly_avg['traffic_volume'].max():.1f} "
        f"at Hour {int(hourly_avg.loc[hourly_avg['traffic_volume'].idxmax(), 'Hour'])}"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hourly_avg["Hour"],
        hourly_avg["traffic_volume"],
        marker="o",
        linewidth=2
    )

    plt.title("Average Traffic Volume by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Traffic Volume")
    plt.xticks(range(0, 24))
    plt.grid(True)

    output_path = "traffic_demand_by_hour.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(f"Figure saved: {output_path}")

    # Interpretation:
    # Traffic volume peaks during morning and evening commuter periods
    # and declines significantly overnight.


# ============================================================
# VISUALISATION 2: WEEKDAY VS WEEKEND TRAFFIC
# ============================================================

def create_weekday_weekend_chart(df):
    """
    Bar chart comparing average traffic volume on weekdays vs weekends.
    """
    logger.info("Generating Figure 2: Weekday vs Weekend Traffic")

    df["Day_Type"] = np.where(
        df["Is_Weekend"] == 1,
        "Weekend",
        "Weekday"
    )

    traffic_summary = (
        df.groupby("Day_Type")["traffic_volume"]
        .mean()
    )

    logger.debug(f"Weekday vs Weekend averages: {traffic_summary.to_dict()}")

    plt.figure(figsize=(8, 5))

    traffic_summary.plot(
        kind="bar",
        color=["steelblue", "orange"]
    )

    plt.title("Weekday vs Weekend Traffic")
    plt.ylabel("Average Traffic Volume")
    plt.xlabel("")
    plt.xticks(rotation=0)

    output_path = "weekday_vs_weekend_traffic.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(f"Figure saved: {output_path}")

    # Interpretation:
    # Average weekday traffic is higher than weekend traffic, suggesting
    # work-related commuting is a major contributor to congestion.


# ============================================================
# VISUALISATION 3: TEMPERATURE VS TRAFFIC
# ============================================================

def create_temperature_traffic_chart(df):
    """
    Scatter plot examining the relationship between temperature and traffic volume.
    """
    logger.info("Generating Figure 3: Temperature vs Traffic")

    df["Temp_Celsius"] = df["temp"] - 273.15

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["Temp_Celsius"],
        df["traffic_volume"],
        alpha=0.30
    )

    plt.title("Temperature vs Traffic Volume")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Traffic Volume")
    plt.grid(True)

    output_path = "temperature_vs_traffic.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(f"Figure saved: {output_path}")

    # Interpretation:
    # Traffic volume varies across temperatures, but no strong linear
    # relationship appears. Time-of-day factors likely have greater
    # influence on congestion.


# ============================================================
# GENERATE ALL VISUALISATIONS
# ============================================================

def generate_visualisations(df):
    create_hourly_traffic_chart(df)
    create_weekday_weekend_chart(df)
    create_temperature_traffic_chart(df)


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
        logger.info("PART 2 - TASK 3 VISUALISATIONS: STARTING")
        logger.info("=" * 60)

        df = load_dataset()
        generate_visualisations(df)

        logger.info("All visualisations generated successfully.")
        logger.info("=" * 60)
        logger.info("PART 2 - TASK 3 VISUALISATIONS: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error(
            "Visualisation process terminated: source file not found.",
            exc_info=True
        )
        sys.exit(1)

    except pd.errors.ParserError:
        logger.error(
            "Visualisation process terminated: parsing failure.",
            exc_info=True
        )
        sys.exit(1)

    except Exception as error:
        logger.error(
            f"Visualisation process terminated due to an unexpected error: {error}",
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()