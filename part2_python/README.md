# Smart City Traffic Analytics Pipeline (`part2_python`)

An end-to-end Python data cleaning, feature engineering, visualisation, and interactive CLI analytics pipeline built for the **Metro Interstate Traffic Volume** dataset.

---

## Project Structure

```text
part2_python/
├── pipeline.py                       # Task 1: Raw data ingestion, validation, and cleaning
├── feature_engineering.py            # Task 2: Temporal, cyclical, weather, and congestion features
├── visualizations.py                 # Task 3: High-resolution exploratory charts (300 DPI)
├── traffic_cli.py                    # Task 4: Interactive command-line traffic query tool
├── traffic_demand_by_hour.png        # Output Figure 1: 24-hour average traffic profile
├── weekday_vs_weekend_traffic.png    # Output Figure 2: Weekday vs. weekend comparison
├── temperature_vs_traffic.png        # Output Figure 3: Temperature (°C) vs. traffic scatter plot
├── pipeline.log                      # Sample execution log across Tasks 1–4
├── report.md                         # Written summary of methodology and key findings
└── README.md                         # Project documentation and setup instructions
```

---

## Prerequisites & Setup

1. **Python Version:** Python 3.9 or higher.
2. **Required Libraries:** Install dependencies using `pip`:
   ```bash
   pip install pandas numpy matplotlib
   ```
3. **Input Dataset:** Place `Metro_Interstate_Traffic_Volume.csv` in the `capstone_part2/` root directory (and ensure `CSV_FILENAME` in `pipeline.py` points to `"Metro_Interstate_Traffic_Volume.csv"`).

---

## How to Run the Pipeline

Execute the scripts in sequential order so each stage generates the input file required by the next step:

### Step 1: Data Cleaning (`pipeline.py`)
Validates schema, removes duplicate records, cleans anomalies, and outputs `cleaned_traffic_data.csv`.
```bash
python pipeline.py
```

### Step 2: Feature Engineering (`feature_engineering.py`)
Generates cyclical time encodings, weather one-hot indicators, freezing precipitation hazard flags, scaled continuous metrics, and quartile congestion labels (`engineered_traffic_features.csv`).
```bash
python feature_engineering.py
```

### Step 3: Generate Visualisations (`visualizations.py`)
Reads `engineered_traffic_features.csv` and exports three 300-DPI PNG charts to the working directory.
```bash
python visualizations.py
```

### Step 4: Launch the Interactive CLI App (`traffic_cli.py`)
Starts the Smart City Traffic Analytics terminal menu with full input validation and audit logging.
```bash
python traffic_cli.py
```

#### CLI Menu Options
1. **Traffic at Date/Time:** Look up exact traffic volume, congestion tier, and weather for a timestamp (`YYYY-MM-DD HH:MM:SS`).
2. **High Traffic Periods:** Count hours exceeding a user-defined vehicle threshold (default: `5500`) and identify the most frequent peak hour.
3. **Weekday vs Weekend:** Compare average hourly vehicle volumes between weekdays and weekends.
4. **Travel Recommendation:** Display the top 3 lowest-traffic hours of the day.
5. **Weather Lookup:** Retrieve record counts and average traffic volume for a specified weather condition (e.g., `Clear`, `Rain`, `Snow`, `Clouds`).
6. **Exit:** Safely terminate the session.

---

## Logging Architecture

All modules share a unified logging standard:
* Module-level loggers are initialised via `logging.getLogger(__name__)` without duplicate handlers.
* Entry points (`main()`) configure `logging.basicConfig` to append structured `INFO`, `DEBUG`, and `ERROR` events simultaneously to **`pipeline.log`** and `sys.stdout`.
