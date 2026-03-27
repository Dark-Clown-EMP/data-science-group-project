"""
RF_v2: Honest Day-Ahead Random Forest for GB National Demand Forecasting
=========================================================================
- Target: ND
- Train: data before 2025-01-01
- Test:  data from 2025-01-01 onward
- 51 features: 5 demand lags, 6 rolling stats, 5 calendar, 4 cyclical,
               31 lagged weather (all shifted 24h, no same-time weather)
- Optuna tuning: 500 trials, 5-fold TimeSeriesSplit, TPE sampler
- SQLite-backed Optuna persistence with resume support
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_data.csv"

TARGET = "ND"
TRAIN_CUTOFF = "2025-01-01"

N_TRIALS = 500
N_SPLITS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_regression(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": _mape(np.asarray(y_true), np.asarray(y_pred)),
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    """Build the 51-feature RF_v2 feature set. Returns (df_model, feature_cols)."""

    df = df.copy()

    # --- Demand lags (5) ---------------------------------------------------
    for lag in [24, 48, 72, 168, 336]:
        df[f"ND_lag_{lag}"] = df[TARGET].shift(lag)

    lag_cols = [f"ND_lag_{lag}" for lag in [24, 48, 72, 168, 336]]

    # --- Rolling demand stats (6) ------------------------------------------
    # Shift by 24 first, then roll => no future leakage
    shifted = df[TARGET].shift(24)
    for win in [24, 48, 168]:
        df[f"ND_rmean_{win}h"] = shifted.rolling(win).mean()
        df[f"ND_rstd_{win}h"] = shifted.rolling(win).std()

    rolling_cols = [
        f"ND_rmean_{w}h" for w in [24, 48, 168]
    ] + [
        f"ND_rstd_{w}h" for w in [24, 48, 168]
    ]

    # --- Calendar features (5) ---------------------------------------------
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    calendar_cols = ["hour", "day_of_week", "month", "day_of_year", "is_weekend"]

    # --- Cyclical features (4) ----------------------------------------------
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    cyclical_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos"]

    # --- 24h-lagged weather (31) --------------------------------------------
    temp_cols = [c for c in df.columns if c.startswith("Temp_")]
    wind_cols = [c for c in df.columns if c.startswith("Wind10m_")]
    solar_cols = [c for c in df.columns if c.startswith("Solar_")]
    raw_weather = temp_cols + wind_cols + solar_cols

    lagged_weather_cols = []
    for col in raw_weather:
        new_name = f"{col}_lag24"
        df[new_name] = df[col].shift(24)
        lagged_weather_cols.append(new_name)

    # --- Assemble feature list ----------------------------------------------
    feature_cols = lag_cols + rolling_cols + calendar_cols + cyclical_cols + lagged_weather_cols

    # --- Drop NaN rows introduced by shifts / rolling -----------------------
    df_model = df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)

    return df_model, feature_cols


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def prepare_data(df):
    df_model, feature_cols = engineer_features(df)

    train_df = df_model[df_model["datetime"] < TRAIN_CUTOFF].copy()
    test_df = df_model[df_model["datetime"] >= TRAIN_CUTOFF].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test, train_df, test_df, feature_cols


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------
def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------
def tune_rf(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = list(tscv.split(X_train))

    print("=" * 70)
    print(f"   HYPERPARAMETER TUNING - Optuna {N_TRIALS} trials x {N_SPLITS} folds")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Features: {X_train.shape[1]}")
    print("=" * 70)
    print()

    overall_start = time.time()

    def objective(trial):
        use_none_depth = trial.suggest_categorical("max_depth_none", [True, False])
        if use_none_depth:
            max_depth = None
        else:
            max_depth = trial.suggest_int("max_depth", 10, 50)

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
            "max_depth": max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.15, 0.80),
            "max_samples": trial.suggest_float("max_samples", 0.70, 1.00),
            "bootstrap": True,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        fold_maes = []
        for tr_idx, val_idx in folds:
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            rf = RandomForestRegressor(**params)
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_val)
            fold_maes.append(mean_absolute_error(y_val, y_pred))

        return float(np.mean(fold_maes))

    def trial_callback(study, trial):
        elapsed = time.time() - overall_start
        eta = (elapsed / (trial.number + 1)) * (N_TRIALS - trial.number - 1)
        best = study.best_trial
        marker = "  * NEW BEST" if trial.number == best.number else ""
        sys.stdout.write(
            f"   Trial {trial.number + 1:>3}/{N_TRIALS}  |  "
            f"MAE: {trial.value:>10.2f}  |  "
            f"Best: {best.value:>10.2f}  |  "
            f"ETA: {_format_time(eta)}{marker}\n"
        )
        sys.stdout.flush()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{PROJECT_ROOT / 'optuna_rf.db'}"
    study_name = "rf_v2_day_ahead_tuning"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )

    completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    if completed > 0:
        print(f"   Resuming study '{study_name}' with {completed} completed trials")
        print(f"   Current best MAE: {study.best_value:.2f}")
    else:
        print(f"   Created new study '{study_name}'")
    print()

    remaining = max(0, N_TRIALS - completed)
    if remaining == 0:
        print("   All trials already completed, skipping optimization")
    else:
        study.optimize(objective, n_trials=remaining, callbacks=[trial_callback])

    total_time = time.time() - overall_start

    # Extract best params (reconstruct max_depth properly)
    best_trial = study.best_trial
    bp = dict(best_trial.params)

    # Reconstruct clean param dict
    use_none = bp.pop("max_depth_none", False)
    if use_none:
        bp["max_depth"] = None
        bp.pop("max_depth", None)  # remove the int suggestion if present
    # If max_depth_none is False, max_depth int is already in bp

    total_completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )

    print()
    print("=" * 70)
    print(f"   TUNING COMPLETE in {_format_time(total_time)}")
    print(f"   Total completed trials: {total_completed}")
    print(f"   Best CV MAE: {study.best_value:.2f}")
    print(f"   Best Trial: {best_trial.number + 1}")
    print(f"   Best Parameters:")
    for k, v in sorted(bp.items()):
        if isinstance(v, float):
            print(f"     {k}: {v:.6f}")
        else:
            print(f"     {k}: {v}")
    print("=" * 70)

    return bp, study


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(test_df, y_test, y_pred, save_path):
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    dt = pd.to_datetime(test_df["datetime"].values)

    ax = axes[0]
    ax.plot(dt, y_test, linewidth=0.6, alpha=0.85, label="Actual ND", color="#2196F3")
    ax.plot(dt, y_pred, linewidth=0.6, alpha=0.85, label="Predicted ND", color="#FF5722")
    ax.set_ylabel("National Demand (MW)", fontsize=11)
    ax.set_title(
        "RF_v2 Day-Ahead: Actual vs Predicted National Demand (2025 Test Set)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    ax2.fill_between(dt, residuals, 0, alpha=0.4, color="#9C27B0", label="Residual")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Residual (MW)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance_df, save_path, top_n=20):
    top = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(top)))
    ax.barh(top["feature"], top["importance"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Importance (Gini / Impurity)", fontsize=11)
    ax.set_title(
        f"RF_v2 Day-Ahead: Top {top_n} Feature Importances",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(True, axis="x", alpha=0.3)

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["importance"] + importance_df["importance"].max() * 0.01,
            i, f'{row["importance"]:.4f}', va="center", fontsize=8, color="#333",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_results_png(results, save_path):
    test_m = results["test_metrics"]
    train_m = results["train_metrics"]
    bp = results.get("best_params") or {}

    lines = [
        "RF_v2 Day-Ahead Forecasting Results",
        "=" * 45,
        "",
        "TEST SET (2025):",
        f"  RMSE : {test_m['RMSE']:.2f}",
        f"  MAE  : {test_m['MAE']:.2f}",
        f"  R\u00b2   : {test_m['R2']:.4f}",
        f"  MAPE : {test_m['MAPE']:.2f}%",
        "",
        "TRAINING SET:",
        f"  RMSE : {train_m['RMSE']:.2f}",
        f"  MAE  : {train_m['MAE']:.2f}",
        f"  R\u00b2   : {train_m['R2']:.4f}",
        f"  MAPE : {train_m['MAPE']:.2f}%",
        "",
        "BEST HYPERPARAMETERS:",
    ]
    for k, v in sorted(bp.items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Features: {len(results['feature_cols'])}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    text = "\n".join(lines)
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0", edgecolor="#333"),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_rf_v2(path=DATA_PATH):
    df = load_data(path)

    X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = \
        prepare_data(df)

    print(f"   Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Feature columns:")
    for col in feature_cols:
        print(f"     - {col}")
    print()

    # --- Tune ---------------------------------------------------------------
    best_params, study = tune_rf(X_train, y_train)

    # --- Final model --------------------------------------------------------
    print()
    print("=" * 70)
    print("   FINAL MODEL TRAINING")

    final_params = {
        **best_params,
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**final_params)

    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start

    print(f"   Final RF trained in {_format_time(train_time)}")
    print(f"   n_estimators: {model.n_estimators}")
    print(f"   max_depth: {model.max_depth}")
    print("=" * 70)

    # --- Predict & evaluate -------------------------------------------------
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = evaluate_regression(y_train.values, y_pred_train)
    test_metrics = evaluate_regression(y_test.values, y_pred_test)

    print()
    print("RF_v2 DAY-AHEAD RESULTS (Test 2025):")
    print(f"   RMSE: {test_metrics['RMSE']:.2f}")
    print(f"   MAE : {test_metrics['MAE']:.2f}")
    print(f"   R2  : {test_metrics['R2']:.3f}")
    print(f"   MAPE: {test_metrics['MAPE']:.2f} %")

    print()
    print("TRAINING DATA RESULTS:")
    print(f"   RMSE: {train_metrics['RMSE']:.2f}")
    print(f"   MAE : {train_metrics['MAE']:.2f}")
    print(f"   R2  : {train_metrics['R2']:.3f}")
    print(f"   MAPE: {train_metrics['MAPE']:.2f} %")

    # --- Feature importance -------------------------------------------------
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # --- Build result frames ------------------------------------------------
    train_results = pd.DataFrame({
        "datetime": train_df["datetime"].values,
        "actual_ND": y_train.values,
        "predicted_ND": y_pred_train,
        "dataset": "train",
    })
    test_results = pd.DataFrame({
        "datetime": test_df["datetime"].values,
        "actual_ND": y_test.values,
        "predicted_ND": y_pred_test,
        "dataset": "test",
    })

    return {
        "model": model,
        "study": study,
        "best_params": best_params,
        "feature_cols": feature_cols,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_results": train_results,
        "test_results": test_results,
        "feature_importance": importance_df,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "test_df": test_df,
    }


def save_outputs(results):
    data_dir = PROJECT_ROOT / "data" / "processed"
    fig_dir = PROJECT_ROOT / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results["train_results"].to_csv(
        data_dir / "rf_v2_train_predictions.csv", index=False
    )
    results["test_results"].to_csv(
        data_dir / "rf_v2_test_predictions.csv", index=False
    )
    results["feature_importance"].to_csv(
        data_dir / "rf_v2_feature_importance.csv", index=False
    )

    bp_serializable = {
        k: (v if v is not None else "None") for k, v in results["best_params"].items()
    }
    with open(data_dir / "rf_v2_best_params.json", "w") as f:
        json.dump(bp_serializable, f, indent=2)

    plot_actual_vs_predicted(
        results["test_df"], results["y_test"].values,
        results["y_pred_test"], fig_dir / "rf_v2_actual_vs_predicted.png",
    )
    plot_feature_importance(
        results["feature_importance"],
        fig_dir / "rf_v2_feature_importance.png",
    )
    save_results_png(results, fig_dir / "rf_v2_results_summary.png")

    print()
    print(f"   Outputs saved to {data_dir}/ and {fig_dir}/")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline_start = time.time()
    print()
    print("=" * 70)
    print("   RF_v2: HONEST DAY-AHEAD RANDOM FOREST PIPELINE")
    print("=" * 70)
    print()

    results = run_rf_v2()
    save_outputs(results)

    total = time.time() - pipeline_start
    print()
    print("=" * 70)
    print(f"   PIPELINE COMPLETE  |  Total time: {_format_time(total)}")
    print("=" * 70)