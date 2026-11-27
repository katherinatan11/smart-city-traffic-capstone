part2_python

# Smart City Mobility Analytics: Data Engineering, Feature Pipeline, and CLI Decision Support
**Capstone Project — Part 2 Summary Report**

---

## 1. Executive Summary
Urban traffic management relies on clean, structured telemetry to anticipate bottlenecks and guide commuter routing. In Part 2 of this capstone project, an end-to-end Python data engineering and analytics pipeline was built for the **Metro Interstate Traffic Volume** dataset. The system ingests raw sensor and weather records, enforces data quality standards, engineers domain-specific temporal and meteorological features, generates publication-ready visualisations, and exposes real-time analytical queries through an interactive command-line interface (CLI) backed by centralised execution logging.

---

## 2. Methodology

### 2.1 Data Ingestion & Cleaning (`pipeline.py`)
* **Validation & Error Handling:** The ingestion module validates file existence, verifies the 9-column expected schema (`holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`, `traffic_volume`), and wraps I/O operations in structured `try/except` blocks (`FileNotFoundError`, `ParserError`, `OSError`).
* **Deduplication & Anomaly Filtering:** Raw multi-weather entries sharing identical timestamps were deduplicated, reducing the dataset from **48,204 raw records** to **40,575 unique hourly observations**, and physical anomalies (such as `0 K` temperature sensor dropouts) were cleaned prior to exporting `cleaned_traffic_data.csv`.

### 2.2 Feature Engineering (`feature_engineering.py`)
The cleaned dataset was expanded into **31 analytical columns** (`engineered_traffic_features.csv`) across four transformation stages:
1. **Temporal & Cyclical Encoding:** Extracted `Hour` (0–23), `Day_of_Week` (0–6), and a binary `Is_Weekend` flag. Applied sine and cosine transformations (`Hour_Sin`, `Hour_Cos`, `Day_Sin`, `Day_Cos`) so downstream models preserve circular continuity between hour 23 and hour 0.
2. **Meteorological Encoding & Hazard Flagging:** One-hot encoded `weather_main` (`Weather_*`) while preserving the original label, and engineered a binary `Freezing_Precipitation_Hazard` indicator triggered when temperature drops below freezing (`temp < 273.15 K`) during active rain or snow (`rain_1h > 0` or `snow_1h > 0`).
3. **Unit Conversion & Scaling:** Converted Kelvin to `Temp_Celsius`, standardised temperature via Z-score (`Scaled_Temp`), and applied Min-Max scaling to cloud cover (`Normalised_Clouds`) with zero-variance guards.
4. **Target Categorisation:** Partitioned `traffic_volume` into three data-driven `Congestion_Category` tiers using interquartile boundaries: **Low Flow** ($\le Q_1$), **Moderate Flow** ($Q_1 < \text{volume} < Q_3$), and **High Flow** ($\ge Q_3$).

### 2.3 Visual Analytics & Interactive CLI (`visualizations.py` & `traffic_cli.py`)
* **Visual Reporting:** Generated three 300-DPI charts inspecting hourly demand curves, weekday vs. weekend volume shifts, and temperature dispersion.
* **CLI Mini-Application:** Built an interactive menu-driven utility allowing traffic operators to query exact timestamps, filter high-volume hours above custom thresholds, compare weekday/weekend demand, identify optimal off-peak travel windows, and inspect weather-specific traffic averages.
* **Observability:** All four scripts follow a uniform logging architecture (`logging.getLogger(__name__)` at module level and `logging.basicConfig` in `main()`), writing structured timestamps and severity levels to `pipeline.log`.

---

## 3. Key Analytical Findings

1. **Bimodal Commuter Demand (`traffic_demand_by_hour.png`):** Traffic volume follows a pronounced twin-peak curve driven by work commutes. Volume surges sharply during the **morning rush (07:00–08:00)**, moderates slightly midday, reaches its **daily peak during the evening rush (16:00–17:00)**, and drops to minimum levels overnight (**01:00–04:00**).
2. **Weekday vs. Weekend Shift (`weekday_vs_weekend_traffic.png`):** Average hourly traffic volume is substantially higher on **weekdays** than on **weekends**, confirming that weekday commuter flow—rather than leisure travel—is the primary driver of interstate congestion.
3. **Temperature Independence (`temperature_vs_traffic.png`):** Across the observed temperature range (from sub-zero winter conditions to warm summer highs), traffic volume exhibits broad vertical dispersion with no strong linear correlation. Time-of-day and day-of-week factors exert far stronger influence over highway demand than ambient temperature alone.
4. **Actionable Off-Peak Routing:** CLI travel-window analysis confirms that scheduling freight or non-urgent transit during the top three lowest-volume hours (**02:00, 03:00, and 04:00**) avoids congestion bottlenecks entirely.
