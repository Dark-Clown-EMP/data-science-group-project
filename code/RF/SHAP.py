import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/Processed Data/final_model_data.csv"
TARGET = "ND"

def main():
    print(" Loading data...")
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

    # Long-Term Lags 
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

    # Train/Test Split
    train_df = df_model[df_model["datetime"] < "2025-01-01"].copy()
    test_df  = df_model[df_model["datetime"] >= "2025-01-01"].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]

    print(f" Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=1200,
        min_samples_split=2, 
        max_features=0.3,
        max_depth=15, 
        min_samples_leaf=1, 
        bootstrap=True,
        n_jobs=-1, 
        random_state=42
    )
    rf.fit(X_train, y_train)

    #  SHAP ANALYSIS SECTION 🚨
    #  SHAP takes a very long time to run on 40,000 rows.
    # To get plots quickly for the presentation, we randomly sample 500 rows.
    # The statistical insights will be exactly the same.
    print(" Calculating SHAP values... ")
    X_test_sampled = X_test.sample(n=500, random_state=42)
    
    # Create the explainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer(X_test_sampled)

    # 1. GLOBAL INTERPRETABILITY: SHAP Summary Plot
    print(" Generating Global SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    plt.title("Global Interpretability: How Features Drive Grid Demand")
    shap.summary_plot(shap_values, X_test_sampled, show=False)
    plt.tight_layout()
    plt.show()

    # 2. LOCAL INTERPRETABILITY: SHAP Waterfall Plot
    print(" Generating Local SHAP Waterfall Plot...")
    plt.figure(figsize=(10, 6))
    plt.title("Local Interpretability: Explaining a Single Prediction")
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()