from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/Processed Data/final_model_data.csv"
PREDICTIONS_OUTPUT_PATH = BASE_DIR / "code/RF/outputs/tuned_rf_predictions.csv"
TARGET = "ND"

def mean_abs_error(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    sum_err = 0
    for i in range(n):
        if y_true[i] != 0:
            sum_err += abs(y_pred[i] - y_true[i]) / y_true[i]
    return sum_err / n if n > 0 else 0


def build_predictions_frame(
    split_name: str,
    frame: pd.DataFrame,
    actual_values: pd.Series,
    predicted_values: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": split_name,
            "datetime": frame["datetime"].values,
            f"actual_{TARGET}": actual_values.values,
            f"predicted_{TARGET}": predicted_values,
        }
    )

def main():
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # CALENDAR-ALIGNED LAGS
    # Short-Term Lags (Days)
    df["ND_lag_24"] = df[TARGET].shift(24)   # 1 Day
    df["ND_lag_48"] = df[TARGET].shift(48)   # 2 Days
    df["ND_lag_72"] = df[TARGET].shift(72)   # 3 Days

    # Medium-Term Lags (Weeks)
    df["ND_lag_168"] = df[TARGET].shift(168) # 1 Week
    df["ND_lag_336"] = df[TARGET].shift(336) # 2 Weeks

    # Long-Term Lags (Perfectly aligned by exactly 4 weeks and 52 weeks)
    df["ND_lag_672"] = df[TARGET].shift(672)   # Exactly 4 weeks
    df["ND_lag_8736"] = df[TARGET].shift(8736) # Exactly 52 weeks

    lag_cols = [
        "ND_lag_24", "ND_lag_48", "ND_lag_72", 
        "ND_lag_168", "ND_lag_336", 
        "ND_lag_672", "ND_lag_8736"
    ]

    temp_cols = [c for c in df.columns if c.startswith("Temp_")]
    wind_cols = [c for c in df.columns if c.startswith("Wind10m_")]
    solar_cols = [c for c in df.columns if c.startswith("Solar_")]

    feature_cols = lag_cols + temp_cols + wind_cols + solar_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_model = df.dropna(subset=[TARGET] + lag_cols).copy()

    # Time-based split 
    train_df = df_model[df_model["datetime"] < "2025-01-01"].copy()
    test_df  = df_model[df_model["datetime"] >= "2025-01-01"].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET]

    print("Train rows:", len(train_df), "| Test rows:", len(test_df))
    print("Num features:", len(feature_cols))

    # 1) Hyperparameter search
    tscv = TimeSeriesSplit(n_splits=3)

    param_dist = {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.3, 0.5, 0.8],
        "bootstrap": [True]
    }
    base_rf = RandomForestRegressor(
        random_state=42,
        n_jobs=1
    )

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=15,  
        scoring="neg_mean_absolute_error",
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    print("\n⏳ Running hyperparameter tuning (training set only)...")
    search.fit(X_train, y_train)

    print("\n🏆 BEST PARAMETERS:")
    print(search.best_params_)
    print("Best CV MAE:", round(-search.best_score_, 2))

    best_rf = search.best_estimator_

    # 2) Final test evaluation
    pred = best_rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    mape = mean_abs_error(y_test.tolist(), pred.tolist()) * 100

    print("\n✅ TUNED RANDOM FOREST RESULTS (Test 2025):")
    print("RMSE:", round(rmse, 2))
    print("MAE :", round(mae, 2))
    print("R2  :", round(r2, 3))
    print("MAPE:", round(mape, 2), "%")

    # 3) Predict on the Training Data
    train_pred = best_rf.predict(X_train)

    # Calculate Train Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    train_mape = mean_abs_error(y_train.tolist(), train_pred.tolist()) * 100

    print("\n📊 TRAINING DATA RESULTS:")
    print(f"RMSE: {train_rmse:.2f}")
    print(f"MAE : {train_mae:.2f}")
    print(f"R2  : {train_r2:.3f}")
    print(f"MAPE: {train_mape:.2f} %")

    train_predictions_df = build_predictions_frame(
        split_name="train",
        frame=train_df,
        actual_values=y_train,
        predicted_values=train_pred,
    )
    test_predictions_df = build_predictions_frame(
        split_name="test",
        frame=test_df,
        actual_values=y_test,
        predicted_values=pred,
    )
    predictions_df = pd.concat(
        [train_predictions_df, test_predictions_df],
        ignore_index=True,
    )

    PREDICTIONS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH.parent / "test_predictions.csv", index=False)
    predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"\nSaved predictions to: {PREDICTIONS_OUTPUT_PATH}")

    # Plot first 7 days of test
    n = min(len(test_df), 24 * 7)  
    plt.figure(figsize=(14, 5))
    plt.plot(test_df["datetime"].iloc[:n], y_test.iloc[:n], label="Actual")
    plt.plot(test_df["datetime"].iloc[:n], pred[:n], label="Tuned RF Pred", linestyle="--")
    plt.title("ND Forecast (Tuned RF) — first 7 days of test")
    plt.xlabel("Datetime")
    plt.ylabel("ND")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()