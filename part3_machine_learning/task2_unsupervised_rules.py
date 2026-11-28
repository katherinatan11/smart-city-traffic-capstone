# ============================================================
# PART 1 OF 3: IMPORTS, LOGGER, PATHS & FEATURE PREPARATION
# ============================================================

import os
import sys
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import logging
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Module-level logger only (handlers configured strictly in main())
logger = logging.getLogger(__name__)

# Directory & File Configuration
OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"
ARTIFACT_DIR = "artifacts"

DATA_SOURCE = os.path.join(OUTPUT_DIR, "ml_traffic_accident_proxy.csv")


def prepare_time_and_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure hour, day_of_week, _is_weekend, and weather_severity exist."""
    df = df.copy()

    # Map capitalized columns from Part 2 if present
    if "Hour" in df.columns and "hour" not in df.columns:
        df["hour"] = df["Hour"]
    if "Day_of_Week" in df.columns and "day_of_week" not in df.columns:
        df["day_of_week"] = df["Day_of_Week"]
    if "Is_Weekend" in df.columns and "is_weekend" not in df.columns:
        df["is_weekend"] = df["Is_Weekend"]

    # 1. Extract time features if raw date_time is present
    if "date_time" in df.columns:
        dt = pd.to_datetime(df["date_time"], errors="coerce")
        invalid_dt = int(dt.isna().sum())
        if invalid_dt > 0:
            logger.warning("Coerced %d invalid 'date_time' value(s) to NaT.", invalid_dt)

        if "hour" not in df.columns:
            df["hour"] = dt.dt.hour
        if "day_of_week" not in df.columns and "dayofweek" not in df.columns:
            df["day_of_week"] = dt.dt.dayofweek

    if "hour" not in df.columns:
        raise KeyError("Required column 'hour' (or 'Hour' / 'date_time') not found in dataset.")

    # 2. Determine weekday vs weekend
    if "is_weekend" in df.columns:
        df["_is_weekend"] = (df["is_weekend"] == 1)
    elif "day_of_week" in df.columns:
        df["_is_weekend"] = (df["day_of_week"] >= 5)
    elif "dayofweek" in df.columns:
        df["_is_weekend"] = (df["dayofweek"] >= 5)
    else:
        logger.warning("No weekday column found; defaulting all rows to 'Weekday' for association rules.")
        df["_is_weekend"] = False

    # 3. Cap extreme weather outliers (e.g., known rain_1h sensor spikes)
    for col in ["rain_1h", "snow_1h"]:
        if col in df.columns:
            upper_cap = float(df[col].quantile(0.999))
            if upper_cap > 0:
                outlier_mask = df[col] > upper_cap
                outlier_count = int(outlier_mask.sum())
                if outlier_count > 0:
                    df.loc[outlier_mask, col] = upper_cap
                    logger.warning(
                        "Capped %d extreme outlier(s) in '%s' at 99.9th percentile (%.2f).",
                        outlier_count,
                        col,
                        upper_cap,
                    )

    # 4. Build composite weather_severity if not already present
    if "weather_severity" not in df.columns:
        severity = np.zeros(len(df), dtype=float)
        components_used = []

        if "rain_1h" in df.columns:
            rain_max = df["rain_1h"].max()
            if rain_max > 0:
                severity += (df["rain_1h"].fillna(0) / rain_max) * 40.0
                components_used.append("rain_1h")

        if "snow_1h" in df.columns:
            snow_max = df["snow_1h"].max()
            if snow_max > 0:
                severity += (df["snow_1h"].fillna(0) / snow_max) * 40.0
                components_used.append("snow_1h")

        if "clouds_all" in df.columns:
            severity += (df["clouds_all"].fillna(0) / 100.0) * 20.0
            components_used.append("clouds_all")

        if "weather_main" in df.columns:
            weather_weights = {
                "Clear": 0.0,
                "Clouds": 15.0,
                "Mist": 30.0,
                "Haze": 30.0,
                "Fog": 45.0,
                "Drizzle": 40.0,
                "Rain": 60.0,
                "Snow": 75.0,
                "Thunderstorm": 95.0,
                "Squall": 95.0,
                "Smoke": 50.0,
                "Dust": 50.0,
            }
            severity += df["weather_main"].astype(str).map(weather_weights).fillna(25.0)
            components_used.append("weather_main")

        if not components_used:
            weather_ohe_cols = [c for c in df.columns if "weather" in c.lower()]
            if weather_ohe_cols:
                severe_keywords = ("rain", "snow", "storm", "thunder", "fog", "mist", "drizzle", "squall")
                for col in weather_ohe_cols:
                    weight = 60.0 if any(k in col.lower() for k in severe_keywords) else 15.0
                    severity += df[col].fillna(0).astype(float) * weight
                components_used.append("one-hot weather columns")
            else:
                logger.warning("No weather columns detected; setting 'weather_severity' to 0.")

        df["weather_severity"] = severity
        logger.info("Computed composite 'weather_severity' using: %s", components_used)

    return df


def interpret_cluster(row: pd.Series, vol_q33: float, vol_q66: float, sev_q75: float) -> str:
    """Generate a plain-language interpretation of a K-Means cluster centroid."""
    hr = row["hour"]
    vol = row["traffic_volume"]
    sev = row["weather_severity"]

    if 6 <= hr < 10:
        time_desc = "Morning rush hours"
    elif 10 <= hr < 16:
        time_desc = "Midday hours"
    elif 16 <= hr < 20:
        time_desc = "Evening rush hours"
    elif 20 <= hr <= 23:
        time_desc = "Late evening hours"
    else:
        time_desc = "Overnight / early-morning hours"

    if vol >= vol_q66:
        vol_desc = "heavy traffic congestion"
    elif vol >= vol_q33:
        vol_desc = "moderate traffic volume"
    else:
        vol_desc = "light free-flowing traffic"

    if sev > sev_q75 and sev > 0:
        weather_desc = "adverse/severe weather conditions"
    else:
        weather_desc = "mild/clear weather conditions"

    return f"{time_desc} with {vol_desc} under {weather_desc}."

# ============================================================
# PART 2 OF 3: DISCRETISATION & ASSOCIATION RULE MINING
# ============================================================

def discretise_for_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Discretise time of day, weekday type, weather, and congestion level into transaction items."""
    disc = pd.DataFrame(index=df.index)

    # 1. Time of day
    def map_time_of_day(h: float) -> str:
        if 0 <= h < 6:
            return "Time=Night(00-05)"
        elif 6 <= h < 10:
            return "Time=MorningRush(06-09)"
        elif 10 <= h < 16:
            return "Time=Midday(10-15)"
        elif 16 <= h < 20:
            return "Time=EveningRush(16-19)"
        return "Time=LateEvening(20-23)"

    disc["time_of_day"] = df["hour"].apply(map_time_of_day)

    # 2. Weekday type
    disc["weekday_type"] = np.where(df["_is_weekend"], "DayType=Weekend", "DayType=Weekday")

    # 3. Weather condition
    if "weather_main" in df.columns:
        def map_weather(w: str) -> str:
            w_low = str(w).lower()
            if any(k in w_low for k in ("thunder", "snow", "squall", "rain")):
                return "Weather=Severe(Rain/Snow/Storm)"
            if any(k in w_low for k in ("fog", "mist", "haze", "drizzle", "smoke", "dust")):
                return "Weather=ReducedVisibility(Fog/Mist/Drizzle)"
            if "cloud" in w_low:
                return "Weather=Cloudy"
            return "Weather=Clear"

        disc["weather"] = df["weather_main"].apply(map_weather)
    else:
        q33 = df["weather_severity"].quantile(0.33)
        q75 = df["weather_severity"].quantile(0.75)
        if q75 > q33:
            disc["weather"] = pd.cut(
                df["weather_severity"],
                bins=[-np.inf, q33, q75, np.inf],
                labels=["Weather=Clear/Mild", "Weather=Moderate", "Weather=Severe"],
            ).astype(str)
        else:
            disc["weather"] = np.where(
                df["weather_severity"] > df["weather_severity"].median(),
                "Weather=Adverse",
                "Weather=Clear/Mild",
            )

    # 4. Congestion level (prefer existing congestion_category if available)
    if "congestion_category" in df.columns:
        disc["congestion"] = "Congestion=" + df["congestion_category"].astype(str)
    else:
        disc["congestion"] = pd.qcut(
            df["traffic_volume"],
            q=3,
            labels=["Congestion=Low", "Congestion=Moderate", "Congestion=High"],
        ).astype(str)

    return disc


def mine_congestion_rules_mlxtend(
    disc: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.40,
) -> pd.DataFrame:
    """Mine association rules predicting congestion level using mlxtend FP-Growth."""
    basket = pd.get_dummies(disc, prefix="", prefix_sep="").astype(bool)
    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return rules

    mask = rules["consequents"].apply(
        lambda c: len(c) == 1 and next(iter(c)).startswith("Congestion=")
    ) & rules["antecedents"].apply(
        lambda a: all(not item.startswith("Congestion=") for item in a)
    )
    rules = rules[mask].copy()
    rules["antecedent_str"] = rules["antecedents"].apply(lambda x: " AND ".join(sorted(x)))
    rules["consequent_str"] = rules["consequents"].apply(lambda x: next(iter(x)))
    return rules.sort_values(by=["lift", "confidence"], ascending=False)


def mine_congestion_rules_fallback(
    disc: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.40,
) -> pd.DataFrame:
    """Exact pure-Python fallback miner for rules predicting congestion level."""
    n_rows = len(disc)
    feature_cols = ["time_of_day", "weekday_type", "weather"]
    grouped = disc.groupby(feature_cols + ["congestion"], observed=True).size().reset_index(name="count")

    consequent_counts = disc["congestion"].value_counts().to_dict()
    antecedent_counts = {}
    joint_counts = {}

    for row in grouped.itertuples(index=False):
        feats = [getattr(row, c) for c in feature_cols]
        cong = row.congestion
        cnt = row.count
        for r in range(1, len(feats) + 1):
            for ant in combinations(sorted(feats), r):
                antecedent_counts[ant] = antecedent_counts.get(ant, 0) + cnt
                joint_counts[(ant, cong)] = joint_counts.get((ant, cong), 0) + cnt

    records = []
    for (ant, cong), joint_cnt in joint_counts.items():
        support = joint_cnt / n_rows
        if support < min_support:
            continue
        ant_cnt = antecedent_counts[ant]
        confidence = joint_cnt / ant_cnt
        if confidence < min_confidence:
            continue
        cons_support = consequent_counts[cong] / n_rows
        lift = confidence / cons_support
        records.append(
            {
                "antecedent_str": " AND ".join(ant),
                "consequent_str": cong,
                "support": support,
                "confidence": confidence,
                "lift": lift,
            }
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(by=["lift", "confidence"], ascending=False)


def explain_rule_plain_language(row: pd.Series) -> str:
    """Convert an association rule row into a plain-language explanation."""
    ant_parts = [p.split("=", 1)[1] for p in row["antecedent_str"].split(" AND ")]
    cons_val = row["consequent_str"].split("=", 1)[1]
    return (
        f"When conditions are [{', '.join(ant_parts)}], congestion is '{cons_val}' "
        f"{row['confidence'] * 100:.1f}% of the time ({row['lift']:.2f}x more likely than baseline)."
    )

# ============================================================
# PART 3 OF 3: MAIN ENTRY POINT & PIPELINE EXECUTION
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

    mlflow_active = False

    try:
        logger.info("=" * 60)
        logger.info("PART 3 - TASK 2 UNSUPERVISED LEARNING: STARTING")
        logger.info("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FIGURE_DIR, exist_ok=True)
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        if HAS_MLFLOW:
            mlflow.start_run(run_name="Task2_Unsupervised")
            mlflow_active = True

        if not os.path.exists(DATA_SOURCE):
            raise FileNotFoundError(
                f"Inbound data error: '{DATA_SOURCE}' missing. Run Proxy Label script first."
            )

        df = pd.read_csv(DATA_SOURCE)
        logger.info("Loaded dataset '%s' with shape %s.", DATA_SOURCE, df.shape)

        if "traffic_volume" not in df.columns:
            raise KeyError("Required column 'traffic_volume' not found in dataset.")

        # Prepare time & weather features
        df = prepare_time_and_weather_features(df)

        # Drop rows with missing core clustering columns
        cluster_cols = ["hour", "weather_severity", "traffic_volume"]
        initial_rows = len(df)
        df = df.dropna(subset=cluster_cols).copy()
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.warning(
                "Dropped %d row(s) containing missing values in %s.",
                dropped,
                cluster_cols,
            )

        # ====================================================
        # PART 2A: K-MEANS CLUSTERING
        # ====================================================
        logger.info("Starting K-Means Clustering on features %s.", cluster_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[cluster_cols])

        sample_size = min(10000, len(X_scaled))
        k_scores = {}
        for k in (3, 4, 5):
            km_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels_tmp = km_tmp.fit_predict(X_scaled)
            sil = float(
                silhouette_score(
                    X_scaled, labels_tmp, sample_size=sample_size, random_state=42
                )
            )
            k_scores[k] = sil
            logger.debug(
                "K-Means evaluation: k=%d -> Silhouette Score=%.4f, Inertia=%.2f",
                k,
                sil,
                km_tmp.inertia_,
            )

        best_k = max(k_scores, key=k_scores.get)
        logger.info(
            "Selected optimal k=%d (highest Silhouette Score=%.4f).",
            best_k,
            k_scores[best_k],
        )

        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        df["traffic_cluster"] = kmeans.fit_predict(X_scaled)
        logger.info(
            "K-Means clustering complete. Assigned all %d rows to %d clusters.",
            len(df),
            best_k,
        )

        # Summarize clusters
        cluster_summary = (
            df.groupby("traffic_cluster")
            .agg(
                Count=("traffic_volume", "size"),
                Mean_Hour=("hour", "mean"),
                Mean_Weather_Severity=("weather_severity", "mean"),
                Mean_Traffic_Volume=("traffic_volume", "mean"),
            )
            .reset_index()
        )
        cluster_summary["Share_Pct"] = (cluster_summary["Count"] / len(df)) * 100.0

        vol_q33 = float(df["traffic_volume"].quantile(0.33))
        vol_q66 = float(df["traffic_volume"].quantile(0.66))
        sev_q75 = float(df["weather_severity"].quantile(0.75))

        cluster_summary["Interpretation"] = cluster_summary.apply(
            lambda r: interpret_cluster(
                pd.Series(
                    {
                        "hour": r["Mean_Hour"],
                        "weather_severity": r["Mean_Weather_Severity"],
                        "traffic_volume": r["Mean_Traffic_Volume"],
                    }
                ),
                vol_q33,
                vol_q66,
                sev_q75,
            ),
            axis=1,
        )

        summary_csv_path = os.path.join(OUTPUT_DIR, "kmeans_cluster_summary.csv")
        assignments_csv_path = os.path.join(OUTPUT_DIR, "traffic_cluster_assignments.csv")

        cluster_summary.to_csv(summary_csv_path, index=False)
        df.to_csv(assignments_csv_path, index=False)
        logger.info("Saved clustering outputs: %s and %s", summary_csv_path, assignments_csv_path)

        # Save Cluster Scatter Plot
        plt.figure(figsize=(10, 6))
        for cluster_id in sorted(df["traffic_cluster"].unique()):
            tmp = df[df["traffic_cluster"] == cluster_id]
            plt.scatter(
                tmp["hour"],
                tmp["traffic_volume"],
                alpha=0.4,
                label=f"Cluster {cluster_id}",
            )

        plt.xlabel("Hour")
        plt.ylabel("Traffic Volume")
        plt.title("Traffic Clusters")
        plt.legend()
        plt.tight_layout()

        cluster_fig_path = os.path.join(FIGURE_DIR, "traffic_clusters.png")
        plt.savefig(cluster_fig_path, dpi=300)
        plt.close()
        logger.info("Saved figure: %s", cluster_fig_path)

        # ====================================================
        # PART 2B: ASSOCIATION RULE MINING
        # ====================================================
        logger.info("Starting Association Rule Mining for congestion patterns.")
        disc = discretise_for_rules(df)
        logger.debug("Discretised dataset into categorical items: %s", disc.columns.tolist())

        min_support = 0.03
        min_confidence = 0.40

        if HAS_MLXTEND:
            logger.info(
                "Mining frequent itemsets using mlxtend FP-Growth (min_support=%.2f, min_confidence=%.2f).",
                min_support,
                min_confidence,
            )
            rules = mine_congestion_rules_mlxtend(
                disc, min_support=min_support, min_confidence=min_confidence
            )
        else:
            logger.warning(
                "Package 'mlxtend' is not installed; using built-in exact Python rule miner fallback."
            )
            rules = mine_congestion_rules_fallback(
                disc, min_support=min_support, min_confidence=min_confidence
            )

        if rules.empty:
            logger.warning(
                "No congestion rules met thresholds (support>=%.2f, confidence>=%.2f); relaxing thresholds.",
                min_support,
                min_confidence,
            )
            if HAS_MLXTEND:
                rules = mine_congestion_rules_mlxtend(
                    disc, min_support=0.01, min_confidence=0.25
                )
            else:
                rules = mine_congestion_rules_fallback(
                    disc, min_support=0.01, min_confidence=0.25
                )

        top_rules = rules.head(10).copy()
        rules_csv_path = os.path.join(OUTPUT_DIR, "association_rules.csv")

        if not top_rules.empty:
            top_rules["Plain_Language_Meaning"] = top_rules.apply(
                explain_rule_plain_language, axis=1
            )
            top_rules.to_csv(rules_csv_path, index=False)

            recommendations_df = pd.DataFrame(
                {"Recommendation": top_rules["Plain_Language_Meaning"]}
            )
            recommendations_df.to_csv(
                os.path.join(OUTPUT_DIR, "traffic_recommendations.csv"),
                index=False,
            )
            logger.info(
                "Mined %d congestion-predicting rule(s); saved top %d by lift to %s.",
                len(rules),
                len(top_rules),
                rules_csv_path,
            )
        else:
            logger.warning("No association rules could be generated even after relaxing thresholds.")

        # Save Findings Artifact
        findings_path = os.path.join(ARTIFACT_DIR, "task2_findings.txt")
        with open(findings_path, "w", encoding="utf-8") as f:
            f.write("K-MEANS CLUSTER FINDINGS\n\n")
            for _, row in cluster_summary.iterrows():
                f.write(
                    f"Cluster {int(row['traffic_cluster'])}: {row['Interpretation']}\n"
                )

            f.write("\nTOP ASSOCIATION RULES\n\n")
            if not top_rules.empty:
                for _, row in top_rules.iterrows():
                    f.write(row["Plain_Language_Meaning"] + "\n")

        logger.info("Saved Task 2 findings summary to %s", findings_path)

        # Log MLflow Artifacts
        if mlflow_active:
            mlflow.log_param("best_k", best_k)
            mlflow.log_metric("best_silhouette", k_scores[best_k])
            mlflow.log_artifact(summary_csv_path)
            if not top_rules.empty:
                mlflow.log_artifact(rules_csv_path)

        # Log Final Summaries (Rule 5: zero print() statements)
        logger.info(
            "\nK-MEANS CLUSTER SUMMARY (k=%d):\n%s",
            best_k,
            cluster_summary[
                [
                    "traffic_cluster",
                    "Count",
                    "Share_Pct",
                    "Mean_Hour",
                    "Mean_Weather_Severity",
                    "Mean_Traffic_Volume",
                    "Interpretation",
                ]
            ]
            .round(2)
            .to_string(index=False),
        )

        if not top_rules.empty:
            display_cols = [
                "antecedent_str",
                "consequent_str",
                "support",
                "confidence",
                "lift",
                "Plain_Language_Meaning",
            ]
            logger.info(
                "\nTOP ASSOCIATION RULES (SORTED BY LIFT):\n%s",
                top_rules[display_cols].round(4).to_string(index=False),
            )

        logger.info("=" * 60)
        logger.info("PART 3 - TASK 2 UNSUPERVISED LEARNING: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Task 2 aborted: required input dataset not found.", exc_info=True)
        sys.exit(1)
    except KeyError:
        logger.error("Task 2 aborted: required column missing.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Task 2 aborted due to an unexpected error.", exc_info=True)
        sys.exit(1)
    finally:
        if mlflow_active:
            mlflow.end_run()


if __name__ == "__main__":
    main()