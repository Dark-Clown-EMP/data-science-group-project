import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. SETUP AND DATA LOADING
# ==========================================
DATA_PATH = "data/Processed Data/final_model_data.csv"
TARGET = "ND"

def main():
    print("🚀 Loading data and engineering features...")
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

    # Drop missing rows
    df_model = df.dropna(subset=[TARGET] + lag_cols).copy()

    # ==========================================
    # 2. TRAIN THE MODEL (Using Training Data Only)
    # ==========================================
    # We only need to train the model to get feature importances. 
    # We use the training set (pre-2025) to prevent any data leakage.
    train_df = df_model[df_model["datetime"] < "2025-01-01"].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    print(f"🌲 Training Random Forest on {len(X_train)} rows to calculate importance...")
    
    # Using your tuned parameters
    rf = RandomForestRegressor(
        n_estimators=800,
        min_samples_split=10, 
        max_features=0.3,
        max_depth=None, 
        min_samples_leaf=1, 
        bootstrap=True,
        n_jobs=-1, 
        random_state=42
    )

    rf.fit(X_train, y_train)

    # ==========================================
    # 3. EXTRACT AND PLOT FEATURE IMPORTANCES
    # ==========================================
    print("📊 Generating Feature Importance Plot...")
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1] # Sort from highest to lowest

    # Get the top 15 features so the chart isn't overly crowded
    top_n = 15
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Top {top_n} Drivers of UK Grid Demand (Random Forest Global Importance)", fontsize=14, fontweight='bold')
    
    # Create the bar chart
    bars = plt.bar(
        range(top_n), 
        importances[top_indices], 
        align="center", 
        color="#2c7bb6", 
        edgecolor="black"
    )
    
    # Label the x-axis with the actual feature names
    plt.xticks(
        range(top_n), 
        [feature_cols[i] for i in top_indices], 
        rotation=45, 
        ha="right",
        fontsize=10
    )
    
    plt.ylabel("Relative Importance Score (Gini)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Optional: Add the exact numbers on top of the bars for clarity
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()