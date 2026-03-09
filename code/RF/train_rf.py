from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/Processed Data/final_model_data.csv"
TARGET = "ND"

# Define your custom MAPE function OUTSIDE of main()
def mean_abs_error(y_true : List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    sum_err = 0  # renamed from 'sum' to avoid overriding Python's built-in sum()
    for i in range(n):
        if y_true[i] != 0:
            sum_err += abs(y_pred[i] - y_true[i]) / y_true[i]
    return sum_err / n if n > 0 else 0

def main():
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

   # Short-Term Lags (Days)
    df["ND_lag_24"] = df[TARGET].shift(24)   # 1 Day
    df["ND_lag_48"] = df[TARGET].shift(48)   # 2 Days
    df["ND_lag_72"] = df[TARGET].shift(72)   # 3 Days

    # Medium-Term Lags (Weeks)
    df["ND_lag_168"] = df[TARGET].shift(168) # 1 Week
    df["ND_lag_336"] = df[TARGET].shift(336) # 2 Weeks

    # Long-Term Lags (Months & Years)
    df["ND_lag_720"] = df[TARGET].shift(720)   # 30 Days (~1 Month)
    df["ND_lag_8760"] = df[TARGET].shift(8760) # 365 Days (1 Year)
   

    lag_cols = [
        "ND_lag_24", "ND_lag_48", "ND_lag_72", 
        "ND_lag_168", "ND_lag_336", 
        "ND_lag_720", "ND_lag_8760"
    ]
    temp_cols = [c for c in df.columns if c.startswith("Temp_")]
    wind_cols = [c for c in df.columns if c.startswith("Wind10m_")]
    solar_cols = [c for c in df.columns if c.startswith("Solar_")]

    feature_cols = lag_cols + temp_cols + wind_cols + solar_cols 
    feature_cols = [c for c in feature_cols if c in df.columns]

    # 3) Drop rows where target or required lags are missing
    df_model = df.dropna(subset=[TARGET] + lag_cols).copy()

    # 4) Train/Test split (time-based)
    
    train_df = df_model[df_model["datetime"] < "2025-01-01"].copy()
    test_df  = df_model[df_model["datetime"] >= "2025-01-01"].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    print("Train rows:", len(train_df), " Test rows:", len(test_df))
    print("Num features:", len(feature_cols))
    print("Example features:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")

    # 5) Model
    rf = RandomForestRegressor(
    n_estimators=400, 
    max_depth=25, 
    min_samples_leaf=5, 
    n_jobs=-1, 
    random_state=42
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    # 6) Metrics (Now properly indented inside main!)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)

    print("\nRANDOM FOREST RESULTS:")
    print("RMSE:", round(rmse, 2))
    print("MAE :", round(mae, 2))
    print("R2  :", round(r2, 3))
    
    MAPE = mean_abs_error(y_test.tolist(), pred.tolist())
    print("MAPE :", round(MAPE*100, 2), "%")



    # 7) Predict on the Training Data
    train_pred = rf.predict(X_train)

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



    # 8) Plot: actual vs predicted (first 7 days of test)
    n = min(len(test_df), 48 * 7)
    plt.figure(figsize=(14, 5))
    plt.plot(test_df["datetime"].iloc[:n], y_test.iloc[:n], label="Actual")
    plt.plot(test_df["datetime"].iloc[:n], pred[:n], label="RF Pred", linestyle="--")
    plt.title("ND Day-ahead Forecast (RF) — first 7 days of test")
    plt.xlabel("Datetime")
    plt.ylabel("ND")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()