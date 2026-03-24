import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# 1. Load Data
# -------------------------
DATA_PATH = "data/Processed Data/final_model_data.csv"
TARGET = "ND"

print("🚀 Loading data for Time Series CV...")
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# -------------------------
# 2. Generate Calendar-Aligned Lags
# -------------------------
# Short-Term Lags (Days)
df["ND_lag_24"] = df[TARGET].shift(24)   
df["ND_lag_48"] = df[TARGET].shift(48)   
df["ND_lag_72"] = df[TARGET].shift(72)   

# Medium-Term Lags (Weeks)
df["ND_lag_168"] = df[TARGET].shift(168) 
df["ND_lag_336"] = df[TARGET].shift(336) 

# Long-Term Lags (4 weeks and 52 weeks)
df["ND_lag_672"] = df[TARGET].shift(672)   
df["ND_lag_8736"] = df[TARGET].shift(8736) 

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

# Drop rows with missing values caused by the 1-year shift
df_model = df.dropna(subset=[TARGET] + lag_cols).copy()

# -------------------------
# 3. Initialize Tuned Model
# -------------------------
rf = RandomForestRegressor(
    n_estimators=800,
    min_samples_split=10, 
    max_features=0.3,
    max_depth=None, 
    min_samples_leaf=1, 
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# -------------------------
# 4. Expanding Window Cross Validation
# -------------------------
print("\n⏳ Running Expanding Window Cross Validation...")
results = []

# Test years: evaluating on 2022, 2023, 2024, and 2025
for test_year in [2022, 2023, 2024, 2025]:
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year+1}-01-01")

    # Expanding window: Train on everything BEFORE the test year
    train = df_model[df_model["datetime"] < test_start]
    
    # Test strictly on the test year
    test = df_model[(df_model["datetime"] >= test_start) & (df_model["datetime"] < test_end)]

    if len(train) == 0 or len(test) == 0:
        continue

    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    print(f"Training up to {test_year-1} ({len(train)} rows) -> Testing on {test_year} ({len(test)} rows)")
    
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    # Custom MAPE
    def mean_abs_error_perc(y_t, y_p):
        y_t, y_p = np.array(y_t), np.array(y_p)
        return np.mean(np.abs((y_t - y_p) / y_t)) * 100
        
    mape = mean_abs_error_perc(y_test, pred)

    results.append([test_year, len(train), len(test), mae, rmse, r2, mape])

# -------------------------
# 5. Print Results
# -------------------------
res_df = pd.DataFrame(results, columns=["Test Year", "Train Rows", "Test Rows", "MAE", "RMSE", "R2", "MAPE (%)"])
print("\n✅ Expanding-Window CV Results:")
print(res_df.to_string(index=False))

print("\n📊 Average Overall Performance:")
print(res_df[["MAE", "RMSE", "R2", "MAPE (%)"]].mean().round(3).to_string())