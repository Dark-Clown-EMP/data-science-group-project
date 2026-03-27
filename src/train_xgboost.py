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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from evaluation import evaluate_regression
from feature_engineering import prepare_model_frame


DATA_PATH = "data/processed/final_model_data.csv"
TARGET = "ND"
TRAIN_CUTOFF = "2025-01-01"

N_TRIALS = 500
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
MAX_BOOSTING_ROUNDS = 5000
VAL_FRACTION = 0.10


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def prepare_xgboost_data(df, target_col=TARGET):
    df_model, feature_cols = prepare_model_frame(
        df, include_weather=True, target_col=target_col
    )

    train_df = df_model[df_model["datetime"] < TRAIN_CUTOFF].copy()
    test_df = df_model[df_model["datetime"] >= TRAIN_CUTOFF].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, train_df, test_df, feature_cols


def _build_model(**extra_params):
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_estimators=MAX_BOOSTING_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=42,
        verbosity=0,
        **extra_params,
    )


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


def tune_xgboost(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = list(tscv.split(X_train))

    print("=" * 70)
    print(f"   HYPERPARAMETER TUNING - Optuna {N_TRIALS} trials x {N_SPLITS} folds")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Early stopping patience: {EARLY_STOPPING_ROUNDS}")
    print(f"   Max boosting rounds per trial: {MAX_BOOSTING_ROUNDS}")
    print("=" * 70)
    print()

    overall_start = time.time()

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }

        fold_maes = []
        for tr_idx, val_idx in folds:
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            mdl = _build_model(**params)
            mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_val_pred = mdl.predict(X_val)
            fold_maes.append(mean_absolute_error(y_val, y_val_pred))

        return np.mean(fold_maes)

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
    storage = "sqlite:///optuna_xgb.db"
    study_name = "xgb_day_ahead_tuning"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
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
    best_params = study.best_trial.params
    total_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    print()
    print("=" * 70)
    print(f"   TUNING COMPLETE in {_format_time(total_time)}")
    print(f"   Total completed trials: {total_completed}")
    print(f"   Best CV MAE: {study.best_value:.2f}")
    print(f"   Best Trial: {study.best_trial.number + 1}")
    print(f"   Best Parameters:")
    for k, v in sorted(best_params.items()):
        if isinstance(v, float):
            print(f"     {k}: {v:.6f}")
        else:
            print(f"     {k}: {v}")
    print("=" * 70)

    return best_params, study



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
        "XGBoost Day-Ahead: Actual vs Predicted National Demand (2025 Test Set)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    residuals = y_test - y_pred
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
    ax.set_xlabel("Importance (Gain)", fontsize=11)
    ax.set_title(
        f"XGBoost Day-Ahead: Top {top_n} Feature Importances",
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



def run_xgboost(path=DATA_PATH, tuned=True):

    df = load_data(path)

    X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = \
        prepare_xgboost_data(df)

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Num features: {len(feature_cols)}")
    print("Feature columns:")
    for col in feature_cols:
        print(f"  - {col}")
    print()

    if tuned:
        best_params, study = tune_xgboost(X_train, y_train)
    else:
        best_params = {}
        study = None

    print()
    print("=" * 70)
    print("   FINAL MODEL TRAINING (with early stopping)")

    split_idx = int(len(X_train) * (1 - VAL_FRACTION))
    X_fit = X_train.iloc[:split_idx]
    y_fit = y_train.iloc[:split_idx]
    X_es_val = X_train.iloc[split_idx:]
    y_es_val = y_train.iloc[split_idx:]

    print(f"   Fit samples: {len(X_fit)} | ES validation samples: {len(X_es_val)}")
    print(f"   Max boosting rounds: {MAX_BOOSTING_ROUNDS}")
    print(f"   Early stopping patience: {EARLY_STOPPING_ROUNDS}")
    print("-" * 70)

    model = _build_model(**best_params)

    train_start = time.time()
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_fit, y_fit), (X_es_val, y_es_val)],
        verbose=False,
    )
    train_time = time.time() - train_start

    best_iteration = model.best_iteration
    print(f"   Early stopping at iteration: {best_iteration + 1}")

    evals = model.evals_result()
    train_rmse_list = evals["validation_0"]["rmse"]
    val_rmse_list = evals["validation_1"]["rmse"]
    total_rounds = len(train_rmse_list)
    step = max(1, total_rounds // 10)
    for epoch in range(0, total_rounds, step):
        sys.stdout.write(
            f"   [Epoch {epoch + 1:>5}/{total_rounds}]  "
            f"Train RMSE: {train_rmse_list[epoch]:>10.2f}  |  "
            f"Val RMSE: {val_rmse_list[epoch]:>10.2f}\n"
        )
    if (total_rounds - 1) % step != 0:
        sys.stdout.write(
            f"   [Epoch {total_rounds:>5}/{total_rounds}]  "
            f"Train RMSE: {train_rmse_list[-1]:>10.2f}  |  "
            f"Val RMSE: {val_rmse_list[-1]:>10.2f}\n"
        )
    print(f"\n   Training completed in {_format_time(train_time)}")
    print("=" * 70)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_metrics = evaluate_regression(y_train.values, y_pred_train)
    test_metrics = evaluate_regression(y_test.values, y_pred_test)

    print()
    print("DAY-AHEAD XGBOOST RESULTS (Test 2025):")
    print(f"RMSE: {test_metrics['RMSE']:.2f}")
    print(f"MAE : {test_metrics['MAE']:.2f}")
    print(f"R2  : {test_metrics['R2']:.3f}")
    print(f"MAPE: {test_metrics['MAPE']:.2f} %")

    print()
    print("TRAINING DATA RESULTS:")
    print(f"RMSE: {train_metrics['RMSE']:.2f}")
    print(f"MAE : {train_metrics['MAE']:.2f}")
    print(f"R2  : {train_metrics['R2']:.3f}")
    print(f"MAPE: {train_metrics['MAPE']:.2f} %")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

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
        "best_iteration": best_iteration,
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



def save_outputs(results, data_dir="data/processed", fig_dir="figures"):
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results["train_results"].to_csv(
        data_dir / "xgb_train_predictions.csv", index=False
    )
    results["test_results"].to_csv(
        data_dir / "xgb_test_predictions.csv", index=False
    )
    results["feature_importance"].to_csv(
        data_dir / "xgb_feature_importance.csv", index=False
    )

    if results["best_params"] is not None:
        with open(data_dir / "xgb_best_params.json", "w") as f:
            json.dump(results["best_params"], f, indent=2)

    plot_actual_vs_predicted(
        results["test_df"], results["y_test"].values,
        results["y_pred_test"], fig_dir / "xgb_actual_vs_predicted.png",
    )
    plot_feature_importance(
        results["feature_importance"],
        fig_dir / "xgb_feature_importance.png",
    )

    print()
    print(f"\U0001f4be Outputs saved to {data_dir}/ and {fig_dir}/")


def save_results_png(results, save_path):
    test_m = results["test_metrics"]
    train_m = results["train_metrics"]
    bp = results.get("best_params") or {}
    best_iter = results.get("best_iteration", "N/A")

    lines = [
        "XGBoost Day-Ahead Forecasting Results",
        "" + "=" * 45,
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
        f"Early stopping iteration: {best_iter}",
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
    print(f"\U0001f4f8 Results summary saved to {save_path}")



if __name__ == "__main__":
    pipeline_start = time.time()
    print()
    print("\u2588" * 70)
    print("   XGBOOST DAY-AHEAD DEMAND FORECASTING PIPELINE")
    print("\u2588" * 70)
    print()

    results = run_xgboost(tuned=True)
    save_outputs(results)

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    save_results_png(results, fig_dir / "xgb_results_summary.png")

    total = time.time() - pipeline_start
    print()
    print("\u2588" * 70)
    print(f"   PIPELINE COMPLETE  |  Total time: {_format_time(total)}")
    print("\u2588" * 70)
