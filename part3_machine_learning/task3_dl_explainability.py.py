# ============================================================
# PART 1 OF 5: IMPORTS, BACKEND DETECTION, LOGGER & PATHS
# ============================================================

import os
import sys
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

# Optional Deep Learning frameworks (PyTorch preferred, then TensorFlow)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    DL_BACKEND = "pytorch"
except ImportError:
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        DL_BACKEND = "tensorflow"
    except ImportError:
        DL_BACKEND = "sklearn_mlp"

# Optional Explainability frameworks (SHAP preferred, then LIME)
try:
    import shap
    EXPLAIN_BACKEND = "shap"
except ImportError:
    try:
        import lime.lime_tabular
        EXPLAIN_BACKEND = "lime"
    except ImportError:
        EXPLAIN_BACKEND = "permutation"

# Module-level logger only (handlers configured strictly in main())
logger = logging.getLogger(__name__)

# Directory & File Configuration
OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"
ARTIFACT_DIR = "artifacts"

DATA_SOURCE = os.path.join(OUTPUT_DIR, "ml_traffic_accident_proxy.csv")
DL_COMPARISON_FILE = os.path.join(OUTPUT_DIR, "task3_dl_comparison.csv")
EXPLAIN_IMPORTANCE_FILE = os.path.join(OUTPUT_DIR, "task3_explainability_importance.csv")
EXPLAIN_NOTES_FILE = os.path.join(ARTIFACT_DIR, "task3_explainability_notes.txt")

SEQ_LEN = 24

# ============================================================
# PART 2 OF 5: FEATURE ENGINEERING & SEQUENCE UTILITIES
# ============================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically, cap extreme outliers, and ensure time/cyclical/holiday features exist."""
    df = df.copy()

    # 1. Chronological ordering
    if "date_time" in df.columns:
        dt = pd.to_datetime(df["date_time"], errors="coerce")
        invalid_dt = int(dt.isna().sum())
        if invalid_dt > 0:
            logger.warning("Coerced %d invalid 'date_time' value(s) to NaT.", invalid_dt)

        df["_dt"] = dt
        df = df.sort_values("_dt").reset_index(drop=True)

        if "Hour" not in df.columns and "hour" not in df.columns:
            df["hour"] = df["_dt"].dt.hour
        if "Day_of_Week" not in df.columns and "day_of_week" not in df.columns:
            df["day_of_week"] = df["_dt"].dt.dayofweek
        if "Month" not in df.columns and "month" not in df.columns:
            df["month"] = df["_dt"].dt.month
        if "Is_Weekend" not in df.columns and "is_weekend" not in df.columns:
            dow_col_tmp = "Day_of_Week" if "Day_of_Week" in df.columns else "day_of_week"
            df["is_weekend"] = (df[dow_col_tmp] >= 5).astype(int)

        df = df.drop(columns=["_dt"])
        logger.info("Sorted dataset chronologically using 'date_time'.")
    else:
        logger.warning("'date_time' column not found; assuming dataset rows are already chronologically ordered.")

    # 2. Cyclical encodings (only add if neither capitalized nor lowercase exists)
    hr_col = "Hour" if "Hour" in df.columns else ("hour" if "hour" in df.columns else None)
    if hr_col is not None:
        if "Hour_Sin" not in df.columns and "hour_sin" not in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df[hr_col] / 24.0)
        if "Hour_Cos" not in df.columns and "hour_cos" not in df.columns:
            df["hour_cos"] = np.cos(2 * np.pi * df[hr_col] / 24.0)

    dow_col = "Day_of_Week" if "Day_of_Week" in df.columns else ("day_of_week" if "day_of_week" in df.columns else None)
    if dow_col is not None:
        if "Day_Sin" not in df.columns and "dow_sin" not in df.columns:
            df["dow_sin"] = np.sin(2 * np.pi * df[dow_col] / 7.0)
        if "Day_Cos" not in df.columns and "dow_cos" not in df.columns:
            df["dow_cos"] = np.cos(2 * np.pi * df[dow_col] / 7.0)

    # 3. Holiday flag
    if "holiday" in df.columns and not pd.api.types.is_numeric_dtype(df["holiday"]):
        df["is_holiday"] = (df["holiday"].notna() & (df["holiday"].astype(str) != "None")).astype(int)
        df = df.drop(columns=["holiday"])

    # 4. Cap extreme sensor outliers (e.g., rain_1h / snow_1h spikes)
    for col in ["rain_1h", "snow_1h"]:
        if col in df.columns:
            cap = float(df[col].quantile(0.999))
            if cap > 0:
                mask = df[col] > cap
                n_out = int(mask.sum())
                if n_out > 0:
                    df.loc[mask, col] = cap
                    logger.warning("Capped %d extreme outlier(s) in '%s' at %.2f.", n_out, col, cap)

    return df


def build_sequences(X_mat: np.ndarray, y_vec: np.ndarray, seq_len: int):
    """Build 3D sliding windows [t - seq_len : t] to predict target y[t]."""
    combined = np.column_stack([X_mat, y_vec])
    X_seq, y_seq = [], []
    for t in range(seq_len, len(combined)):
        X_seq.append(combined[t - seq_len : t])
        y_seq.append(y_vec[t])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def add_lag_features(df_split: pd.DataFrame, exog_cols: list, seq_len: int):
    """Create tabular lagged features aligned with the exact same t >= seq_len horizon as the LSTM."""
    work = df_split[exog_cols].copy()
    work["vol_lag_1"] = df_split["traffic_volume"].shift(1)
    work["vol_lag_2"] = df_split["traffic_volume"].shift(2)
    work["vol_lag_24"] = df_split["traffic_volume"].shift(24)

    X_tab = work.iloc[seq_len:].reset_index(drop=True)
    y_tab = df_split["traffic_volume"].iloc[seq_len:].to_numpy(dtype=np.float32)
    return X_tab, y_tab

# ============================================================
# PART 3 OF 5: DEEP LEARNING MODEL RUNNERS
# ============================================================

def train_pytorch_lstm(X_train_seq, y_train_seq, X_test_seq, epochs: int = 15, batch_size: int = 128):
    """Train a 2-layer PyTorch LSTM and return test predictions."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training PyTorch LSTM on device '%s' (epochs=%d, batch_size=%d).", device, epochs, batch_size)

    class TrafficLSTM(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    n_features = X_train_seq.shape[2]
    model = TrafficLSTM(input_size=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq)),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(bx)
        if epoch in (1, 5, 10, epochs):
            logger.info(
                "LSTM Epoch %02d/%02d - Train MSE (scaled): %.4f",
                epoch,
                epochs,
                total_loss / len(X_train_seq),
            )

    model_path = os.path.join(OUTPUT_DIR, "traffic_lstm.pt")
    torch.save(model.state_dict(), model_path)
    logger.info("Saved trained PyTorch LSTM weights to '%s'.", model_path)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_test_seq).to(device)).cpu().numpy()

    return preds_scaled, "PyTorch LSTM"


def train_tf_lstm(X_train_seq, y_train_seq, X_test_seq, epochs: int = 15, batch_size: int = 128):
    """Train a 2-layer TensorFlow/Keras LSTM and return test predictions."""
    tf.random.set_seed(42)
    logger.info("Training TensorFlow/Keras LSTM (epochs=%d, batch_size=%d).", epochs, batch_size)

    seq_steps = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_steps, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0)

    model_path = os.path.join(OUTPUT_DIR, "traffic_lstm.keras")
    model.save(model_path)
    logger.info("Saved trained Keras LSTM model to '%s'.", model_path)

    preds_scaled = model.predict(X_test_seq, verbose=0).ravel()
    return preds_scaled, "TensorFlow/Keras LSTM"


def train_mlp_fallback(X_train_tab_scaled, y_train_scaled, X_test_tab_scaled):
    """Fallback Option 1 Neural Network (MLPRegressor) if neither PyTorch nor TensorFlow is installed."""
    logger.warning(
        "Neither PyTorch nor TensorFlow is installed; falling back to scikit-learn MLPRegressor "
        "(Option 1 Neural Network for demand prediction)."
    )
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, early_stopping=True, random_state=42)
    mlp.fit(X_train_tab_scaled, y_train_scaled)

    model_path = os.path.join(OUTPUT_DIR, "traffic_mlp_nn.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(mlp, f)
    logger.info("Saved trained MLPRegressor neural network to '%s'.", model_path)

    preds_scaled = mlp.predict(X_test_tab_scaled)
    return preds_scaled, "MLP Neural Network (Fallback)"

# ============================================================
# PART 4 OF 5: MAIN PIPELINE SETUP, SPLITTING & MODEL TRAINING
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

    try:
        logger.info("=" * 60)
        logger.info("PART 3 - TASK 3 DEEP LEARNING & EXPLAINABILITY: STARTING")
        logger.info("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FIGURE_DIR, exist_ok=True)
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        if not os.path.exists(DATA_SOURCE):
            raise FileNotFoundError(
                f"Inbound data error: '{DATA_SOURCE}' missing. Execute Proxy Label script first."
            )

        df = pd.read_csv(DATA_SOURCE)
        logger.info("Loaded dataset '%s' with shape %s.", DATA_SOURCE, df.shape)

        if "traffic_volume" not in df.columns:
            raise KeyError("Required target column 'traffic_volume' missing from dataset.")

        initial_rows = len(df)
        df = df.dropna(subset=["traffic_volume"]).copy()
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.warning("Dropped %d row(s) with missing 'traffic_volume'.", dropped)

        df = prepare_features(df)

        # Exclude target, proxy target, target-derived categories, and raw timestamp
        exclude_cols = {
            "traffic_volume",
            "high_risk",
            "congestion_category",
            "Congestion_Category",
            "is_low_visibility",
            "date_time",
        }
        exog_cols = [c for c in df.columns if c not in exclude_cols]

        numeric_exog = df[exog_cols].select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_exog = df[exog_cols].select_dtypes(include=["object", "str", "category"]).columns.tolist()

        logger.info(
            "Exogenous feature count: %d (numeric=%d, categorical=%d)",
            len(exog_cols),
            len(numeric_exog),
            len(categorical_exog),
        )

        # Chronological 80/20 split (preserves sequential order)
        split_idx = int(len(df) * 0.80)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        logger.info(
            "Chronological split complete. Train rows: %d, Test rows: %d",
            len(train_df),
            len(test_df),
        )

        if len(test_df) <= SEQ_LEN:
            raise ValueError(
                f"Test split ({len(test_df)} rows) is too small for sequence length {SEQ_LEN}."
            )

        # Preprocess exogenous features for LSTM
        transformers = []
        if numeric_exog:
            transformers.append(
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                    numeric_exog,
                )
            )
        if categorical_exog:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_exog,
                )
            )

        seq_prep = ColumnTransformer(transformers=transformers)
        X_train_exog = seq_prep.fit_transform(train_df[exog_cols])
        X_test_exog = seq_prep.transform(test_df[exog_cols])

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(train_df[["traffic_volume"]]).ravel()
        y_test_scaled = y_scaler.transform(test_df[["traffic_volume"]]).ravel()

        # Build 24-hour sequential windows for LSTM
        X_train_seq, y_train_seq = build_sequences(X_train_exog, y_train_scaled, SEQ_LEN)
        X_test_seq, _ = build_sequences(X_test_exog, y_test_scaled, SEQ_LEN)
        y_test_actual = test_df["traffic_volume"].iloc[SEQ_LEN:].to_numpy(dtype=float)

        logger.info(
            "Built sequential tensors (seq_len=%d): X_train_seq=%s, X_test_seq=%s",
            SEQ_LEN,
            X_train_seq.shape,
            X_test_seq.shape,
        )

        # Build aligned tabular dataset (for comparable Random Forest & Explainability)
        X_tab_train, y_tab_train = add_lag_features(train_df, exog_cols, SEQ_LEN)
        X_tab_test, y_tab_test = add_lag_features(test_df, exog_cols, SEQ_LEN)

        tab_numeric = X_tab_train.select_dtypes(include=["number", "bool"]).columns.tolist()
        tab_categorical = X_tab_train.select_dtypes(include=["object", "str", "category"]).columns.tolist()

        tab_transformers = []
        if tab_numeric:
            tab_transformers.append(("num", SimpleImputer(strategy="median"), tab_numeric))
        if tab_categorical:
            tab_transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    tab_categorical,
                )
            )

        tab_prep = ColumnTransformer(transformers=tab_transformers)
        X_tab_train_mat = tab_prep.fit_transform(X_tab_train)
        X_tab_test_mat = tab_prep.transform(X_tab_test)
        tab_feature_names = [
            name.split("__", 1)[-1] for name in tab_prep.get_feature_names_out().tolist()
        ]

        # ----------------------------------------------------
        # 1. Train Deep Learning Model
        # ----------------------------------------------------
        if DL_BACKEND == "pytorch":
            dl_preds_scaled, dl_name = train_pytorch_lstm(X_train_seq, y_train_seq, X_test_seq)
        elif DL_BACKEND == "tensorflow":
            dl_preds_scaled, dl_name = train_tf_lstm(X_train_seq, y_train_seq, X_test_seq)
        else:
            mlp_scaler = StandardScaler()
            X_mlp_tr = mlp_scaler.fit_transform(X_tab_train_mat)
            X_mlp_te = mlp_scaler.transform(X_tab_test_mat)
            y_tab_train_scaled = y_scaler.transform(y_tab_train.reshape(-1, 1)).ravel()
            dl_preds_scaled, dl_name = train_mlp_fallback(X_mlp_tr, y_tab_train_scaled, X_mlp_te)

        dl_preds = y_scaler.inverse_transform(dl_preds_scaled.reshape(-1, 1)).ravel()
        dl_mae = mean_absolute_error(y_test_actual, dl_preds)
        dl_r2 = r2_score(y_test_actual, dl_preds)
        logger.info("%s Evaluation -> MAE: %.2f, R-squared: %.4f", dl_name, dl_mae, dl_r2)

        # ----------------------------------------------------
        # 2. Train Comparable Tree-Based Model for Explainability
        # ----------------------------------------------------
        logger.info("Training comparable RandomForestRegressor on identical horizon for explainability.")
        rf_comp = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        rf_comp.fit(X_tab_train_mat, y_tab_train)
        rf_preds = rf_comp.predict(X_tab_test_mat)
        rf_mae = mean_absolute_error(y_tab_test, rf_preds)
        rf_r2 = r2_score(y_tab_test, rf_preds)
        logger.info("Comparable Random Forest Evaluation -> MAE: %.2f, R-squared: %.4f", rf_mae, rf_r2)


        # ----------------------------------------------------
        # 3. Apply Explainability (SHAP / LIME / Permutation)
        # ----------------------------------------------------
        explain_sample_size = min(500, len(X_tab_test_mat))
        X_explain = X_tab_test_mat[:explain_sample_size]
        explain_method_label = ""

        if EXPLAIN_BACKEND == "shap":
            logger.info("Computing SHAP TreeExplainer values on %d test samples...", explain_sample_size)
            explainer = shap.TreeExplainer(rf_comp)
            shap_vals = explainer.shap_values(X_explain, check_additivity=False)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": tab_feature_names,
                        "Mean_Abs_SHAP": mean_abs_shap,
                    }
                )
                .sort_values("Mean_Abs_SHAP", ascending=False)
                .head(10)
            )
            explain_method_label = "SHAP (TreeExplainer on Comparable Random Forest)"
            logger.info("SHAP computation complete.")

            try:
                shap_fig_path = os.path.join(FIGURE_DIR, "task3_shap_summary.png")
                shap.summary_plot(
                    shap_vals,
                    X_explain,
                    feature_names=tab_feature_names,
                    plot_type="bar",
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(shap_fig_path, dpi=300)
                plt.close()
                logger.info("Saved SHAP feature importance plot to '%s'.", shap_fig_path)
            except Exception as exc:
                logger.warning("Could not save SHAP summary plot: %s", exc)

        elif EXPLAIN_BACKEND == "lime":
            logger.warning("'shap' package not installed; using LIME TabularExplainer instead.")
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_tab_train_mat,
                feature_names=tab_feature_names,
                mode="regression",
                random_state=42,
            )
            n_lime = min(20, len(X_explain))
            agg_weights = np.zeros(len(tab_feature_names))
            for i in range(n_lime):
                exp = lime_explainer.explain_instance(
                    X_explain[i], rf_comp.predict, num_features=len(tab_feature_names)
                )
                local_pairs = next(iter(exp.as_map().values()))
                for feat_idx, weight in local_pairs:
                    agg_weights[feat_idx] += abs(weight)

            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": tab_feature_names,
                        "Mean_Abs_LIME_Weight": agg_weights / n_lime,
                    }
                )
                .sort_values("Mean_Abs_LIME_Weight", ascending=False)
                .head(10)
            )
            explain_method_label = "LIME (TabularExplainer on Comparable Random Forest)"
            logger.info("LIME computation complete.")

        else:
            logger.warning(
                "Neither 'shap' nor 'lime' is installed; falling back to Permutation Importance."
            )
            perm = permutation_importance(
                rf_comp,
                X_explain,
                y_tab_test[:explain_sample_size],
                n_repeats=5,
                random_state=42,
                n_jobs=-1,
            )
            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": tab_feature_names,
                        "Permutation_Importance_R2_Drop": perm.importances_mean,
                    }
                )
                .sort_values("Permutation_Importance_R2_Drop", ascending=False)
                .head(10)
            )
            explain_method_label = "Permutation Importance Fallback (Comparable Random Forest)"
            logger.info("Permutation importance computation complete.")

        # ----------------------------------------------------
        # 4. Save Artifacts & Log Results (Rule 5: No print())
        # ----------------------------------------------------
        comparison = pd.DataFrame(
            {
                "Model": [dl_name, "Comparable Random Forest (Explainability Surrogate)"],
                "MAE": [dl_mae, rf_mae],
                "R-squared": [dl_r2, rf_r2],
            }
        )
        comparison.to_csv(DL_COMPARISON_FILE, index=False)
        importance_df.to_csv(EXPLAIN_IMPORTANCE_FILE, index=False)
        logger.info("Saved comparison CSV: %s", DL_COMPARISON_FILE)
        logger.info("Saved explainability importance CSV: %s", EXPLAIN_IMPORTANCE_FILE)

        justification_text = (
            "EXPLAINABILITY METHODOLOGICAL JUSTIFICATION\n\n"
            f"Deep Learning Model: {dl_name} (24-hour lookback window)\n"
            f"Explainability Method: {explain_method_label}\n\n"
            "Rationale:\n"
            "Multi-step 3D recurrent sequences (24 time steps x features) in an LSTM are\n"
            "difficult to interpret directly because temporal feature attributions span\n"
            "across lagged hidden states. In accordance with task guidelines, SHAP/LIME\n"
            "attributions were computed on a comparable Random Forest regressor trained on\n"
            "the exact same chronological forecast horizon using 1-hour, 2-hour, and 24-hour\n"
            "traffic volume lags alongside exogenous time and weather features.\n"
        )
        with open(EXPLAIN_NOTES_FILE, "w", encoding="utf-8") as f:
            f.write(justification_text)
        logger.info("Saved explainability justification notes: %s", EXPLAIN_NOTES_FILE)

        logger.info(
            "\nTASK 3: DEEP LEARNING & COMPARABLE SURROGATE PERFORMANCE:\n%s",
            comparison.round(4).to_string(index=False),
        )
        logger.info(
            "\nEXPLAINABILITY RESULTS (%s):\n%s",
            explain_method_label,
            importance_df.round(4).to_string(index=False),
        )

        logger.info("=" * 60)
        logger.info("PART 3 - TASK 3 DEEP LEARNING & EXPLAINABILITY: COMPLETED")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Task 3 aborted: required dataset not found.", exc_info=True)
        sys.exit(1)
    except KeyError:
        logger.error("Task 3 aborted: required column missing.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Task 3 aborted due to an unexpected error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()