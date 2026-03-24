"""XGBoost training utilities."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box

from evaluation import evaluate_regression
from feature_engineering import prepare_model_frame

console = Console()


DATA_PATH = "data/processed/final_model_data.csv"
TARGET = "ND"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def prepare_xgboost_data(df, target_col=TARGET):
    df_model, feature_cols = prepare_model_frame(
        df,
        include_weather=True,
        target_col=target_col
    )

    train_df = df_model[df_model["datetime"] < "2025-01-01"].copy()
    test_df = df_model[df_model["datetime"] >= "2025-01-01"].copy()

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
        n_jobs=-1
    )


def tune_xgboost(X_train, y_train):
    param_dist = {
        "n_estimators": [200, 300, 500, 700, 1000],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.5, 1, 2, 5]
    }

    n_iter = 5
    tscv = TimeSeriesSplit(n_splits=5)
    total_fits = n_iter * tscv.n_splits

    console.print(Panel.fit(
        f"[bold cyan]Hyperparameter Tuning[/]\n"
        f"  Candidates:  [yellow]{n_iter}[/]\n"
        f"  CV Folds:    [yellow]{tscv.n_splits}[/]\n"
        f"  Total Fits:  [yellow]{total_fits}[/]\n"
        f"  Scoring:     [yellow]neg_mean_absolute_error[/]",
        title="⚙️  RandomizedSearchCV Config",
        border_style="blue"
    ))

    # Show search space
    space_table = Table(title="Search Space", box=box.ROUNDED, border_style="dim")
    space_table.add_column("Parameter", style="cyan")
    space_table.add_column("Values", style="green")
    for k, v in param_dist.items():
        space_table.add_row(k, str(v))
    console.print(space_table)

    console.print()
    start_time = time.time()

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("🔍 Tuning hyperparameters...", total=total_fits)

        class ProgressCallback:
            def __init__(self):
                self.count = 0
            def __call__(self, *args, **kwargs):
                self.count += 1
                progress.update(task, completed=self.count)

        callback = ProgressCallback()

        search = RandomizedSearchCV(
            estimator=build_base_xgb(),
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=1,
        )

        # Monkey-patch to track progress per fold
        original_fit = search.fit
        def tracked_fit(X, y, **kwargs):
            from sklearn.model_selection._search import _check_refit
            result = original_fit(X, y, **kwargs)
            return result
        search.fit(X_train, y_train)
        progress.update(task, completed=total_fits)

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]✓[/] Tuning completed in [cyan]{elapsed:.1f}s[/]")

    # Show best params
    best_table = Table(title="🏆 Best Parameters", box=box.HEAVY_HEAD, border_style="green")
    best_table.add_column("Parameter", style="cyan bold")
    best_table.add_column("Value", style="yellow")
    for k, v in search.best_params_.items():
        best_table.add_row(k, str(v))
    best_table.add_row("Best CV MAE", f"{-search.best_score_:.2f}", style="bold green")
    console.print(best_table)

    return search


def run_xgboost(path=DATA_PATH, tuned=True):
    console.print(Panel.fit(
        "[bold white]XGBoost Regression Training Pipeline[/]",
        title="🚀 XGBoost Trainer",
        subtitle="Energy Demand Forecasting",
        border_style="bold magenta",
    ))

    # --- Step 1: Load Data ---
    with console.status("[bold cyan]📂 Loading data...", spinner="dots"):
        df = load_data(path)
    console.print(f"[green]✓[/] Data loaded: [cyan]{len(df):,}[/] rows, [cyan]{len(df.columns)}[/] columns")
    console.print(f"  Date range: [yellow]{df['datetime'].min()}[/] → [yellow]{df['datetime'].max()}[/]")

    # --- Step 2: Prepare Features ---
    with console.status("[bold cyan]🔧 Engineering features...", spinner="dots"):
        X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = prepare_xgboost_data(df)

    data_table = Table(title="📊 Dataset Split", box=box.ROUNDED, border_style="cyan")
    data_table.add_column("Set", style="bold")
    data_table.add_column("Samples", justify="right", style="yellow")
    data_table.add_column("Features", justify="right", style="green")
    data_table.add_column("Target", style="cyan")
    data_table.add_row("Train", f"{len(X_train):,}", str(len(feature_cols)), TARGET)
    data_table.add_row("Test", f"{len(X_test):,}", str(len(feature_cols)), TARGET)
    console.print(data_table)

    # --- Step 3: Train / Tune ---
    if tuned:
        console.rule("[bold blue]Hyperparameter Tuning Phase", style="blue")
        search = tune_xgboost(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        search = None
        console.rule("[bold blue]Training Phase", style="blue")
        model = build_base_xgb()
        with console.status("[bold cyan]🏋️ Training base model...", spinner="dots"):
            model.fit(X_train, y_train)
        console.print("[green]✓[/] Base model trained")
        best_params = None

    # --- Step 4: Final training with best params ---
    if tuned:
        console.rule("[bold blue]Final Model Training", style="blue")
        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            n_est = best_params.get("n_estimators", 500)
            task = progress.add_task("🏋️ Training final model...", total=n_est)

            eval_callback = {}
            model.set_params(callbacks=[
                lambda env: progress.update(task, completed=env.iteration + 1)
            ])
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )
        console.print(f"[green]✓[/] Final model trained with [cyan]{n_est}[/] boosting rounds")

    # --- Step 5: Predict ---
    console.rule("[bold blue]Prediction & Evaluation", style="blue")
    with console.status("[bold cyan]🔮 Generating predictions...", spinner="dots"):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_metrics = evaluate_regression(y_train.values, y_pred_train)
        test_metrics = evaluate_regression(y_test.values, y_pred_test)

    # --- Metrics Table ---
    metrics_table = Table(title="📈 Model Performance", box=box.HEAVY_HEAD, border_style="green")
    metrics_table.add_column("Metric", style="cyan bold")
    metrics_table.add_column("Train", justify="right", style="yellow")
    metrics_table.add_column("Test", justify="right", style="bold green")
    for metric in train_metrics:
        train_val = f"{train_metrics[metric]:,.4f}"
        test_val = f"{test_metrics[metric]:,.4f}"
        metrics_table.add_row(metric, train_val, test_val)
    console.print(metrics_table)

    # --- Feature Importance ---
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    feat_table = Table(title="🔑 Top 10 Feature Importances", box=box.ROUNDED, border_style="yellow")
    feat_table.add_column("#", style="dim", justify="right")
    feat_table.add_column("Feature", style="cyan")
    feat_table.add_column("Importance", justify="right", style="green")
    feat_table.add_column("Bar", style="yellow")
    max_imp = importance_df["importance"].max()
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        bar_len = int((row["importance"] / max_imp) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        feat_table.add_row(str(i), row["feature"], f"{row['importance']:.4f}", bar)
    console.print(feat_table)

    train_results = pd.DataFrame({
        "datetime": train_df["datetime"].values,
        "actual_ND": y_train.values,
        "predicted_ND": y_pred_train,
        "dataset": "train"
    })

    test_results = pd.DataFrame({
        "datetime": test_df["datetime"].values,
        "actual_ND": y_test.values,
        "predicted_ND": y_pred_test,
        "dataset": "test"
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
        "feature_importance": importance_df
    }


def save_xgboost_outputs(results, output_dir="data/processed"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with console.status("[bold cyan]💾 Saving outputs...", spinner="dots"):
        results["train_results"].to_csv(output_dir / "xgb_train_predictions.csv", index=False)
        results["test_results"].to_csv(output_dir / "xgb_test_predictions.csv", index=False)
        results["feature_importance"].to_csv(output_dir / "xgb_feature_importance.csv", index=False)

        if results["best_params"] is not None:
            with open(output_dir / "xgb_best_params.json", "w") as f:
                json.dump(results["best_params"], f, indent=2)

    save_table = Table(box=box.ROUNDED, border_style="green")
    save_table.add_column("File", style="cyan")
    save_table.add_column("Status", style="green")
    save_table.add_row("xgb_train_predictions.csv", "✓ Saved")
    save_table.add_row("xgb_test_predictions.csv", "✓ Saved")
    save_table.add_row("xgb_feature_importance.csv", "✓ Saved")
    if results["best_params"] is not None:
        save_table.add_row("xgb_best_params.json", "✓ Saved")
    console.print(Panel(save_table, title="💾 Output Files", subtitle=str(output_dir), border_style="green"))


if __name__ == "__main__":
    console.print()
    start = time.time()
    results = run_xgboost(tuned=True)
    save_xgboost_outputs(results)
    total = time.time() - start
    console.print(Panel.fit(
        f"[bold green]Pipeline completed successfully![/]\n"
        f"Total time: [cyan]{total:.1f}s[/]",
        title="✅ Done",
        border_style="bold green",
    ))