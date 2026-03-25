import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/Processed Data/final_model_data.csv"
TARGET = "ND"

def main():
    print(" Loading data for Feature Ablation Study...")
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
    
    weather_cols = temp_cols + wind_cols + solar_cols
    
    # Define the two different models
    full_features = lag_cols + weather_cols
    full_features = [c for c in full_features if c in df.columns]
    
    ablated_features = lag_cols  # NO WEATHER DATA HERE!
    ablated_features = [c for c in ablated_features if c in df.columns]

    df_model = df.dropna(subset=[TARGET] + lag_cols).copy()

    # Train/Test Split
    train_df = df_model[df_model["datetime"] < "2025-01-01"].copy()
    test_df  = df_model[df_model["datetime"] >= "2025-01-01"].copy()

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    print(f" Initializing Random Forest...")
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

    #  RUN 1: THE FULL MODEL
    print(f"\n Training FULL Model (Lags + {len(weather_cols)} Weather Features)...")
    rf.fit(train_df[full_features], y_train)
    pred_full = rf.predict(test_df[full_features])
    mae_full = mean_absolute_error(y_test, pred_full)
    print(f"Full Model MAE: {mae_full:.2f} MW")

    #  RUN 2: THE ABLATED MODEL
    print(f"\n Training ABLATED Model (Lags ONLY, Zero Weather Data)...")
    rf.fit(train_df[ablated_features], y_train)
    pred_ablated = rf.predict(test_df[ablated_features])
    mae_ablated = mean_absolute_error(y_test, pred_ablated)
    print(f"Ablated Model MAE: {mae_ablated:.2f} MW")

    # CALCULATE VALUE AND PLOT
    diff = mae_ablated - mae_full
    
    print("\n" + "="*50)
    print(f"THE VALUE OF WEATHER: Weather data prevents {diff:.2f} MW of error!")
    print("="*50)

    # Create a professional bar chart
    labels = ['Full Model\n(With Weather)', 'Ablated Model\n(No Weather)']
    maes = [mae_full, mae_ablated]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, maes, color=['#2c7bb6', '#d7191c'], edgecolor='black', width=0.5)
    
    plt.title("Feature Ablation: Quantifying the Value of Weather Data", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Absolute Error (MW) - Lower is Better", fontsize=12)
    plt.ylim(0, max(maes) * 1.2)

    # Add the MAE numbers on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 15, f'{yval:.0f} MW', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add a highlight box showing the exact value
    plt.text(0.5, max(maes) * 1.05, f"Adding weather features reduces\nforecasting error by {diff:.0f} Megawatts", 
             ha='center', va='center', fontsize=12, bbox=dict(facecolor='#ffffcc', alpha=0.8, edgecolor='black'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()