import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from LSTM import tune_lstm_model

# 1. Load and Clean Data
df = pd.read_csv("./data/final_model_data.csv")
columns_drop = ['TSD', 'ENGLAND_WALES_DEMAND','EMBEDDED_WIND_GENERATION',
    'EMBEDDED_WIND_CAPACITY','EMBEDDED_SOLAR_GENERATION','EMBEDDED_SOLAR_CAPACITY',
    'NON_BM_STOR','PUMP_STORAGE_PUMPING','NET_IMPORTS','SCOTTISH_TRANSFER', 'datetime']

# FIX: Reassign the dataframe or use inplace=True
df = df.drop(columns=columns_drop)

data = df.values
scaler = MinMaxScaler(feature_range=(0,1))

# --- Configuration ---
# Assuming you want to predict the feature at column index 0. Change this to your actual target index.
target_col_index = 0
time_steps = [3, 6, 12, 24, 48, 3*30*24]

# 2. Sequential Train/Test Split
split_idx = int(len(data) * 0.8)

train_data = data[:split_idx]
test_data = data[split_idx:]

# 3. Scale Data (Fit only on training data)
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# 4. Sequence Generation Function for Discrete Lags
def create_y_lags_2d(dataset, target_index, lags):
    X, y = [], []
    max_lag = max(lags)
    
    # Ensure lags are sorted to maintain a logical column order (oldest to newest)
    sorted_lags = sorted(lags, reverse=True)
    
    for i in range(max_lag, len(dataset)):
        # 1. Get the current weather features, dropping the target column to prevent data leakage
        current_weather_features = np.delete(dataset[i], target_index)
        
        # 2. Extract ONLY the lagged values of the target variable (y)
        lagged_y_features = [dataset[i - lag, target_index] for lag in sorted_lags]
        
        # 3. Concatenate them into a single, flat 1D array for this specific row
        combined_features = np.concatenate([current_weather_features, lagged_y_features])
        
        X.append(combined_features)
        y.append(dataset[i, target_index])
        
    return np.array(X), np.array(y)

# 5. Build Tensors
X_train, y_train = create_y_lags_2d(train_data_scaled, target_col_index, time_steps)
X_test, y_test = create_y_lags_2d(test_data_scaled, target_col_index, time_steps)

print("--- Tensor Shapes ---")
# The 'Timesteps' dimension will be exactly 5 (because you provided 5 specific lags)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")

best_hps, best_model, X_train_3d, X_test_3d = tune_lstm_model(X_train, y_train, X_test, y_test)