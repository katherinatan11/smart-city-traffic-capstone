# Intelligent Smart City Mobility Solution (`capstone_part3`)

An end-to-end machine learning, deep learning, explainable AI, MLOps, and responsible AI pipeline for forecasting urban highway traffic demand, assessing compound weather-congestion risk, and generating commuter travel recommendations.

---

## Important Dataset & Proxy Accident-Risk Disclosure

* **Source Dataset:** **Metro Interstate Traffic Volume** (westbound I-94 hourly traffic and meteorological records, 2012–2017; `engineered_traffic_features.csv` generated in Part 2).
* **Proxy Accident-Risk Target (`high_risk`):** **No real accident or police collision dataset was used.** Because the Metro Interstate dataset contains traffic volume and weather observations without empirical crash outcomes, **Task 0** constructs a documented binary proxy target (`high_risk`) defined as:
  \[\text{high\_risk} = (\text{Congestion} \in \{\text{High}, \text{Severe}\}) \land (\text{Severe Weather} \lor \text{Low Visibility})\]
  * **High / Severe Congestion:** `traffic_volume > median` (above the 50th percentile).
  * **Severe Weather:** `Rain`, `Snow`, `Thunderstorm`, `Squall`, or `Drizzle`.
  * **Low Visibility:** `Fog`, `Haze`, `Mist`, or `Smoke`.
* **Interpretation Notice:** Model predictions represent **elevated traffic-weather risk exposure** rather than empirical collision probability.

---

## Project Structure

```text
part3_machine_learning/
├── engineered_traffic_features.csv             # From Part 2
├── README.md                                   # Project documentation, proxy disclosure & run guide
├── final_capstone_report.pdf                   # Final Capstone Report
├── bias_and_fairness_report.pdf                # Task 7: Responsible & Sustainable AI / Fairness audit
├── pipeline.log                                # Centralised execution log across all tasks
├── mlflow.db                                   # SQLite MLflow tracking & model registry database
├── mlruns/                                     # MLflow run metadata and skops model artifacts
│   ├── model.skops                             # skops model
├── task0_proxy_label.py                        # Task 0: Documented proxy risk label & class summary
├── task1_supervised_models.py                  # Task 1: Supervised Classification & Regression + plots
├── task2_unsupervised_rules.py                 # Task 2: K-Means clustering & FP-Growth association rules
├── task3_dl_explainability.py                  # Task 3: 2-layer PyTorch LSTM + SHAP/LIME explainability
├── task4_mlflow_tracking.py                    # Task 4: MLflow experiment tracking & model registry
├── task5_recommendations.py                    # Task 5: Scenario-based travel recommendation engine
├── task6_deployment_monitoring.py              # Task 6: FastAPI prediction API & drift monitoring
├── notebooks/
│   ├── task0_proxy_label.ipynb                 # Task 0: Documented proxy risk label & class summary
│   ├── task1_supervised_models.ipynb           # Task 1: Supervised Classification & Regression + plots
│   ├── task2_unsupervised_rules.ipynb          # Task 2: K-Means clustering & FP-Growth association rules
│   ├── task3_dl_explainability.ipynb           # Task 3: 2-layer PyTorch LSTM + SHAP/LIME explainability
│   ├── task4_mlflow_tracking.ipynb             # Task 4: MLflow experiment tracking & model registry
│   ├── task5_recommendations.ipynb             # Task 5: Scenario-based travel recommendation engine
│   ├── task6_deployment_monitoring.ipynb       # Task 6: FastAPI prediction API & drift monitoring
├── outputs/
│   ├── ml_traffic_accident_proxy.csv           # Task 0 proxy-labeled dataset
│   ├── proxy_label_summary.csv                 # High-risk vs. low-risk record summary
│   ├── class_distribution.csv                  # Percentage class distribution
│   ├── proxy_label_documentation.txt           # Formal proxy target disclosure
│   ├── classification_results.csv              # Task 1 Logistic Regression vs. Random Forest metrics
│   ├── regression_results.csv                  # Task 1 Linear Regression vs. Random Forest metrics
│   ├── logistic_regression_model.pkl           # Serialised baseline classifier
│   ├── random_forest_classifier.pkl            # Serialised production classifier
│   ├── linear_regression_model.pkl             # Serialised baseline regressor
│   ├── random_forest_regressor.pkl             # Serialised production regressor (file size too big cannot be loaded into GitHub)
│   ├── traffic_recommendations.csv             # Task 2 Inertia & Silhouette scores (k = 2..6)
│   ├── kmeans_cluster_summary.csv              # Task 2 cluster centroids & plain-language profiles
│   ├── association_rules.csv                   # Task 2 mined FP-Growth association rules
│   ├── task3_dl_comparison.csv                 # Task 3 PyTorch LSTM vs. Lagged Random Forest metrics
│   ├── task3_explainability_importance.csv     # Task 3 SHAP/LIME global feature attribution scores
│   ├── task4_mlflow_experiment_summary.csv     # Task 4 consolidated MLflow run comparison
│   └── task5_travel_recommendations.csv        # Task 5 scenario-by-scenario travel advisories
├── artifacts/
│   ├── selected_features.json                  # Leakage-free feature schema
│   ├── model_metadata.json                     # Supervised model metadata summary
│   ├── task4_features_used.json                # Feature schema logged to MLflow
│   ├── task3_explainability_notes.txt          # Plain-language SHAP/LIME driver summary
│   ├── task5_findings.txt                      # Human-readable commuter & logistics advisories
│   └── task6_mlops_deployment_report.json      # FastAPI TestClient & drift alert simulation output
└── figures/
    ├── roc_curves.png                          # Task 1 ROC curve comparison
    ├── actual_vs_predicted.png                 # Task 1 Random Forest actual vs. predicted scatter
    ├── regression_model_comparison.png         # Task 1 MAE & R² bar comparison
    ├── rf_feature_importance.png               # Task 1 Top 15 Random Forest feature importances
    ├── classification_model_comparison.png     # Task 2 K-Means Elbow & Silhouette diagnostics
    ├── traffic_clusters.png                    # Task 2 Traffic regime scatter plot
    └── task3_shap_summary.png                  # Task 3 Top 12 SHAP/LIME demand drivers

```

---

## Prerequisites & Environment Setup

1. **Python Version:** Python 3.9 or higher.
2. **Install Core Dependencies:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn fastapi httpx pydantic reportlab
   ```
3. **Install Optional Deep Learning, Explainability, Association Rules & MLOps Libraries:**
   *(Every script includes built-in fallbacks if optional packages are not installed.)*
   ```bash
   pip install torch shap mlxtend mlflow skops
   ```
4. **Input Dataset:** Place `engineered_traffic_features.csv` (from Part 2) in the `capstone_part3/` root directory.

---

## How to Run the Pipeline (Tasks 0–7)

Execute the scripts sequentially from inside the `capstone_part3/` directory so each stage generates the models, logs, and artefacts required by downstream tasks:

1. **Task 0 — Generate Documented Proxy Accident-Risk Target:**
   ```bash
   python task0_proxy_label.py
   ```
   Creates `outputs/ml_traffic_accident_proxy.csv`, class distribution summaries, and `outputs/proxy_label_documentation.txt`.

2. **Task 1 — Train Supervised Classification & Regression Models:**
   ```bash
   python task1_supervised_models.py
   ```
   Trains leakage-free `LogisticRegression`, `RandomForestClassifier`, `LinearRegression`, and `RandomForestRegressor` pipelines, saving `.pkl` models, CSV metrics, and ROC/feature-importance plots.

3. **Task 2 — Unsupervised K-Means Clustering & Association Rule Mining:**
   ```bash
   python task2_unsupervised_rules.py
   ```
   Caps extreme precipitation outliers, evaluates K-Means (`k = 2..6`) via Silhouette scores, profiles traffic regimes, and mines FP-Growth association rules.

4. **Task 3 — Deep Learning Demand Forecasting (LSTM) & Explainability (SHAP/LIME):**
   ```bash
   python task3_dl_explainability.py
   ```
   Trains a 2-layer PyTorch LSTM (`SEQ_LEN = 24` hours) on chronologically ordered sequences, benchmarks against an aligned lagged Random Forest, and extracts global SHAP/LIME attributions.

5. **Task 4 — MLflow Experiment Tracking & Model Registry:**
   ```bash
   python task4_mlflow_tracking.py
   ```
   Logs parameters, metrics, schemas, and `skops`-serialised models to `sqlite:///mlflow.db` and registers winning models (`TrafficRiskClassifier` and `TrafficVolumeRegressor`).
   *(Optional: Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to inspect runs in your browser.)*

6. **Task 5 — Scenario-Based Travel Recommendation Engine:**
   ```bash
   python task5_recommendations.py
   ```
   Scores `(Day_Type, Weather_Condition, Hour)` scenarios using a weighted Suitability Penalty (`80%` predicted volume + `20%` risk probability) to generate practical daytime and off-peak travel advisories (`outputs/task5_travel_recommendations.csv` and `artifacts/task5_findings.txt`).

7. **Task 6 — FastAPI Deployment Simulation & Drift Monitoring:**
   ```bash
   python task6_deployment_monitoring.py
   ```
   Runs automated `TestClient` checks across `/`, `/predict_traffic`, `/predict_risk`, and `/experiment_tracking`, and evaluates prediction MAE degradation (`> 15%`) and weather covariate shift alerts (`artifacts/task6_mlops_deployment_report.json`).

8. **Task 7 — Export Final Reports & Responsible AI Audit to PDF:**
   ```bash
   python build_part3_pdfs.py
   ```
   Converts `final_capstone_report.md` and `bias_and_fairness_report.md` (covering Bias, Fairness, Governance, and Sustainability) into formatted `.pdf` documents (`final_capstone_report.pdf` and `bias_and_fairness_report.pdf`).

---

## Logging & Reproducibility

* **Unified Audit Trail:** Every script initialises `logging.getLogger(__name__)` at the module level and appends timestamped `INFO`, `WARNING`, and `ERROR` records to **`pipeline.log`**.
* **Deterministic Splits:** All train/test splits, K-Means initialisations, and PyTorch/NumPy routines fix `random_state=42` (`torch.manual_seed(42)`) so results are fully reproducible across runs.
