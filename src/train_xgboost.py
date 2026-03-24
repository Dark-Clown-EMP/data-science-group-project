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
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from evaluation import evaluate_regression
from feature_engineering import prepare_model_frame


DATA_PATH = "data/processed/final_model_data.csv"
TARGET = "ND"
TRAIN_CUTOFF = "2025-01-01"



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



def build_base_xgb():
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


PARAM_DIST = {
    "n_estimators": [500, 700, 1000, 1500, 2000],
    "max_depth": [4, 5, 6, 7, 8, 10, 12],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.8, 0.85, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma": [0, 0.05, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.05, 0.1, 1],
    "reg_lambda": [0.5, 1, 2, 3, 5],
}

N_ITER = 25
N_SPLITS = 5


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

    rng = np.random.RandomState(42)
    candidates = list(ParameterSampler(PARAM_DIST, n_iter=N_ITER, random_state=rng))

    print("=" * 70)
    print(f"\u23f3  HYPERPARAMETER TUNING — {N_ITER} candidates x {N_SPLITS} folds")
    print(f"   Training samples: {len(X_train)}")
    print("=" * 70)
    print()

    best_score = -np.inf
    best_params = None
    best_estimator = None
    all_results = []
    overall_start = time.time()

    for i, params in enumerate(candidates, 1):
        cand_start = time.time()
        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(folds, 1):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            mdl = build_base_xgb()
            mdl.set_params(**params)
            mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_val_pred = mdl.predict(X_val)
            fold_mae = mean_absolute_error(y_val, y_val_pred)
            fold_scores.append(-fold_mae)

            sys.stdout.write(
                f"\r   Candidate {i:>2}/{N_ITER}  |  "
                f"Fold {fold_idx}/{N_SPLITS}  |  "
                f"Fold MAE: {fold_mae:>10.2f}"
            )
            sys.stdout.flush()

        mean_score = np.mean(fold_scores)
        cand_time = time.time() - cand_start
        elapsed = time.time() - overall_start
        eta = (elapsed / i) * (N_ITER - i)

        marker = ""
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_estimator = mdl
            marker = "  \u2b50 NEW BEST"

        sys.stdout.write(
            f"\r   Candidate {i:>2}/{N_ITER}  |  "
            f"Mean MAE: {-mean_score:>10.2f}  |  "
            f"Time: {_format_time(cand_time)}  |  "
            f"ETA: {_format_time(eta)}{marker}\n"
        )
        sys.stdout.flush()

        all_results.append({
            "candidate": i,
            "params": params,
            "mean_mae": -mean_score,
            "time": cand_time,
        })

    total_time = time.time() - overall_start
    print()
    print("=" * 70)
    print(f"\U0001f3c6  TUNING COMPLETE in {_format_time(total_time)}")
    print(f"   Best CV MAE: {-best_score:.2f}")
    print(f"   Best Parameters:")
    for k, v in sorted(best_params.items()):
        print(f"     {k}: {v}")
    print("=" * 70)

    class TuneResult:
        pass
    result = TuneResult()
    result.best_estimator_ = build_base_xgb()
    result.best_estimator_.set_params(**best_params)
    result.best_params_ = best_params
    result.best_score_ = best_score
    result.cv_results_ = all_results
    return result



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
        search = tune_xgboost(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        search = None
        model = build_base_xgb()
        best_params = None

    print()
    print("=" * 70)
    print("\U0001f680  FINAL MODEL TRAINING")
    n_est = model.get_params().get("n_estimators", 100)
    print(f"   Boosting rounds: {n_est}")
    print("-" * 70)

    train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    train_time = time.time() - train_start

    evals = model.evals_result()
    train_rmse_list = evals["validation_0"]["rmse"]
    test_rmse_list = evals["validation_1"]["rmse"]
    total_rounds = len(train_rmse_list)
    step = max(1, total_rounds // 10)
    for epoch in range(0, total_rounds, step):
        sys.stdout.write(
            f"   [Epoch {epoch + 1:>5}/{total_rounds}]  "
            f"Train RMSE: {train_rmse_list[epoch]:>10.2f}  |  "
            f"Test RMSE: {test_rmse_list[epoch]:>10.2f}\n"
        )
    if (total_rounds - 1) % step != 0:
        sys.stdout.write(
            f"   [Epoch {total_rounds:>5}/{total_rounds}]  "
            f"Train RMSE: {train_rmse_list[-1]:>10.2f}  |  "
            f"Test RMSE: {test_rmse_list[-1]:>10.2f}\n"
        )
    print(f"\n   Training completed in {_format_time(train_time)}")
    print("=" * 70)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_metrics = evaluate_regression(y_train.values, y_pred_train)
    test_metrics = evaluate_regression(y_test.values, y_pred_test)

    print()
    print("\u2705 DAY-AHEAD XGBOOST RESULTS (Test 2025):")
    print(f"RMSE: {test_metrics['RMSE']:.2f}")
    print(f"MAE : {test_metrics['MAE']:.2f}")
    print(f"R2  : {test_metrics['R2']:.3f}")
    print(f"MAPE: {test_metrics['MAPE']:.2f} %")

    print()
    print("\U0001f4ca TRAINING DATA RESULTS:")
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
        "search": search,
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
