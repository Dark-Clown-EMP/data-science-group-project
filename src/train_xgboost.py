"""XGBoost training pipeline for GB electricity demand forecasting."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from rich import box

from evaluation import evaluate_regression
from feature_engineering import prepare_model_frame

console = Console()

DATA_PATH = "data/processed/final_model_data.csv"
TARGET = "ND"
TRAIN_CUTOFF = "2025-01-01"


# ── Data loading & preparation ───────────────────────────────────────

def load_data(path=DATA_PATH):
    """Load the merged modelling dataset."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def prepare_xgboost_data(df, target_col=TARGET):
    """Engineer features and create the chronological train/test split."""
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


# ── Model building & tuning ──────────────────────────────────────────

def build_base_xgb():
    """Return a base XGBRegressor with sensible defaults."""
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


PARAM_DIST = {
    "n_estimators": [200, 300, 500, 700, 1000],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [0.5, 1, 2, 5],
}

N_ITER = 5
N_SPLITS = 5


def tune_xgboost(X_train, y_train):
    """Tune hyperparameters with time-series-aware cross-validation."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    total_fits = N_ITER * N_SPLITS

    # ── Config summary ──
    console.print(Panel.fit(
        f"[bold cyan]Hyperparameter Tuning[/]\n"
        f"  Candidates : [yellow]{N_ITER}[/]\n"
        f"  CV Folds   : [yellow]{N_SPLITS}[/]\n"
        f"  Total Fits : [yellow]{total_fits}[/]\n"
        f"  Scoring    : [yellow]neg_mean_absolute_error[/]",
        title="⚙️  RandomizedSearchCV",
        border_style="blue",
    ))

    space_table = Table(title="Search Space", box=box.ROUNDED, border_style="dim")
    space_table.add_column("Parameter", style="cyan")
    space_table.add_column("Values", style="green")
    for k, v in PARAM_DIST.items():
        space_table.add_row(k, str(v))
    console.print(space_table)
    console.print()

    # ── Run search ──
    start = time.time()
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Tuning hyperparameters...", total=100)

        search = RandomizedSearchCV(
            estimator=build_base_xgb(),
            param_distributions=PARAM_DIST,
            n_iter=N_ITER,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        progress.update(task, completed=100)

    elapsed = time.time() - start
    console.print(f"\n[bold green]✓[/] Tuning completed in [cyan]{elapsed:.1f}s[/]")

    # ── Best params table ──
    best_table = Table(
        title="🏆 Best Parameters", box=box.HEAVY_HEAD, border_style="green"
    )
    best_table.add_column("Parameter", style="cyan bold")
    best_table.add_column("Value", style="yellow")
    for k, v in search.best_params_.items():
        best_table.add_row(k, str(v))
    best_table.add_row(
        "Best CV MAE", f"{-search.best_score_:.2f}", style="bold green"
    )
    console.print(best_table)

    return search


# ── Figures ───────────────────────────────────────────────────────────

def plot_actual_vs_predicted(test_df, y_test, y_pred, save_path):
    """Line plot of actual vs predicted demand on the test set."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    dt = pd.to_datetime(test_df["datetime"].values)

    # ── Top: actual vs predicted ──
    ax = axes[0]
    ax.plot(dt, y_test, linewidth=0.6, alpha=0.85, label="Actual ND", color="#2196F3")
    ax.plot(dt, y_pred, linewidth=0.6, alpha=0.85, label="Predicted ND", color="#FF5722")
    ax.set_ylabel("National Demand (MW)", fontsize=11)
    ax.set_title("XGBoost — Actual vs Predicted National Demand (2025 Test Set)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)

    # ── Bottom: residuals ──
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
    """Horizontal bar chart of feature importances."""
    top = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(top)))
    ax.barh(top["feature"], top["importance"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Importance (Gain)", fontsize=11)
    ax.set_title(f"XGBoost — Top {top_n} Feature Importances",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.3)

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["importance"] + importance_df["importance"].max() * 0.01,
                i, f'{row["importance"]:.4f}', va="center", fontsize=8, color="#333")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main pipeline ─────────────────────────────────────────────────────

def run_xgboost(path=DATA_PATH, tuned=True):
    """End-to-end XGBoost training, evaluation, and output pipeline."""

    console.print(Panel.fit(
        "[bold white]XGBoost Regression Training Pipeline[/]",
        title="🚀 XGBoost Trainer",
        subtitle="GB Day-Ahead Demand Forecasting",
        border_style="bold magenta",
    ))

    # ── Step 1: Load data ──
    console.rule("[bold blue]Step 1 · Load Data", style="blue")
    with console.status("[cyan]Loading data..."):
        df = load_data(path)
    console.print(
        f"[green]✓[/] Loaded [cyan]{len(df):,}[/] rows × "
        f"[cyan]{len(df.columns)}[/] columns"
    )
    console.print(
        f"  Date range: [yellow]{df['datetime'].min()}[/] → "
        f"[yellow]{df['datetime'].max()}[/]"
    )

    # ── Step 2: Feature engineering + split ──
    console.rule("[bold blue]Step 2 · Feature Engineering & Split", style="blue")
    with console.status("[cyan]Engineering features..."):
        X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = \
            prepare_xgboost_data(df)

    split_table = Table(title="Dataset Split", box=box.ROUNDED, border_style="cyan")
    split_table.add_column("Set", style="bold")
    split_table.add_column("Samples", justify="right", style="yellow")
    split_table.add_column("Features", justify="right", style="green")
    split_table.add_column("Target", style="cyan")
    split_table.add_column("Period", style="dim")
    split_table.add_row(
        "Train", f"{len(X_train):,}", str(len(feature_cols)),
        TARGET, f"< {TRAIN_CUTOFF}"
    )
    split_table.add_row(
        "Test", f"{len(X_test):,}", str(len(feature_cols)),
        TARGET, f"≥ {TRAIN_CUTOFF}"
    )
    console.print(split_table)

    # ── Step 3: Hyperparameter tuning ──
    if tuned:
        console.rule("[bold blue]Step 3 · Hyperparameter Tuning", style="blue")
        search = tune_xgboost(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        search = None
        console.rule("[bold blue]Step 3 · Training (default params)", style="blue")
        model = build_base_xgb()
        best_params = None

    # ── Step 4: Final model fit ──
    console.rule("[bold blue]Step 4 · Final Model Training", style="blue")
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        n_est = (best_params or {}).get("n_estimators", 500)
        task = progress.add_task(
            f"Training final model ({n_est} rounds)...", total=100
        )
        model.fit(X_train, y_train)
        progress.update(task, completed=100)

    console.print(f"[green]✓[/] Final model trained — [cyan]{n_est}[/] boosting rounds")

    # ── Step 5: Predict & evaluate ──
    console.rule("[bold blue]Step 5 · Prediction & Evaluation", style="blue")
    with console.status("[cyan]Generating predictions..."):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_metrics = evaluate_regression(y_train.values, y_pred_train)
        test_metrics = evaluate_regression(y_test.values, y_pred_test)

    metrics_table = Table(
        title="📈 Model Performance", box=box.HEAVY_HEAD, border_style="green"
    )
    metrics_table.add_column("Metric", style="cyan bold")
    metrics_table.add_column("Train", justify="right", style="yellow")
    metrics_table.add_column("Test", justify="right", style="bold green")
    for metric in train_metrics:
        metrics_table.add_row(
            metric,
            f"{train_metrics[metric]:,.4f}",
            f"{test_metrics[metric]:,.4f}",
        )
    console.print(metrics_table)

    # ── Feature importance ──
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    feat_table = Table(
        title="🔑 Top 15 Feature Importances",
        box=box.ROUNDED, border_style="yellow",
    )
    feat_table.add_column("#", style="dim", justify="right")
    feat_table.add_column("Feature", style="cyan")
    feat_table.add_column("Importance", justify="right", style="green")
    feat_table.add_column("", style="yellow")
    max_imp = importance_df["importance"].max()
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        bar_len = int((row["importance"] / max_imp) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        feat_table.add_row(str(i), row["feature"], f"{row['importance']:.4f}", bar)
    console.print(feat_table)

    # ── Build result DataFrames ──
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


# ── Save outputs ──────────────────────────────────────────────────────

def save_outputs(results, data_dir="data/processed", fig_dir="figures"):
    """Save predictions, feature importance, best params, and figures."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Step 6 · Saving Outputs", style="blue")

    with console.status("[cyan]Saving CSVs and figures..."):
        # CSVs
        results["train_results"].to_csv(
            data_dir / "xgb_train_predictions.csv", index=False
        )
        results["test_results"].to_csv(
            data_dir / "xgb_test_predictions.csv", index=False
        )
        results["feature_importance"].to_csv(
            data_dir / "xgb_feature_importance.csv", index=False
        )

        # Best params JSON
        if results["best_params"] is not None:
            with open(data_dir / "xgb_best_params.json", "w") as f:
                json.dump(results["best_params"], f, indent=2)

        # Figures
        plot_actual_vs_predicted(
            results["test_df"], results["y_test"].values,
            results["y_pred_test"], fig_dir / "xgb_actual_vs_predicted.png",
        )
        plot_feature_importance(
            results["feature_importance"],
            fig_dir / "xgb_feature_importance.png",
        )

    save_table = Table(box=box.ROUNDED, border_style="green")
    save_table.add_column("File", style="cyan")
    save_table.add_column("Location", style="dim")
    save_table.add_column("Status", style="green")
    save_table.add_row("xgb_train_predictions.csv", str(data_dir), "✓")
    save_table.add_row("xgb_test_predictions.csv", str(data_dir), "✓")
    save_table.add_row("xgb_feature_importance.csv", str(data_dir), "✓")
    if results["best_params"] is not None:
        save_table.add_row("xgb_best_params.json", str(data_dir), "✓")
    save_table.add_row("xgb_actual_vs_predicted.png", str(fig_dir), "✓")
    save_table.add_row("xgb_feature_importance.png", str(fig_dir), "✓")
    console.print(Panel(save_table, title="💾 Saved Outputs", border_style="green"))


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print()
    pipeline_start = time.time()

    results = run_xgboost(tuned=True)
    save_outputs(results)

    total = time.time() - pipeline_start
    console.print(Panel.fit(
        f"[bold green]Pipeline completed successfully![/]\n"
        f"Total time: [cyan]{total:.1f}s[/]",
        title="✅ Done",
        border_style="bold green",
    ))
