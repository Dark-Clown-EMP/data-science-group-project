from typing import List
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import tensorflow as tf
import json
from GWO_LSTM_tuner import tune_lstm_with_gwo_advanced

version_num = '5.0'

def set_global_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_global_determinism(seed=42)

# 1. Load and Clean Data
df_original = pd.read_csv("./data/final_model_data.csv")
columns_drop = ['TSD', 'ENGLAND_WALES_DEMAND','EMBEDDED_WIND_GENERATION',
    'EMBEDDED_WIND_CAPACITY','EMBEDDED_SOLAR_GENERATION','EMBEDDED_SOLAR_CAPACITY',
    'NON_BM_STOR','PUMP_STORAGE_PUMPING','NET_IMPORTS','SCOTTISH_TRANSFER', 'datetime']

df = df_original.drop(columns=columns_drop)

data = df.values
scaler = MinMaxScaler(feature_range=(0,1))

# --- Configuration ---
target_col_index = 0
time_steps = [3, 6, 12, 24, 48, 3*30*24, 24*365]

# 2. Sequential Train/Test Split
split_idx = int(len(data) * 0.8)

train_data = data[:split_idx]
test_data = data[split_idx:]

# 3. Scale Data
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

with open('minmax_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved successfully to 'minmax_scaler.pkl'")

# 4. Sequence Generation Function
def create_y_lags_2d(dataset, target_index, lags):
    X, y = [], []
    max_lag = max(lags)
    sorted_lags = sorted(lags, reverse=True)
    
    for i in range(max_lag, len(dataset)):
        current_weather_features = np.delete(dataset[i], target_index)
        lagged_y_features = [dataset[i - lag, target_index] for lag in sorted_lags]
        combined_features = np.concatenate([current_weather_features, lagged_y_features])
        
        X.append(combined_features)
        y.append(dataset[i, target_index])
        
    return np.array(X), np.array(y)

# 5. Build Tensors
X_train, y_train = create_y_lags_2d(train_data_scaled, target_col_index, time_steps)
X_test, y_test = create_y_lags_2d(test_data_scaled, target_col_index, time_steps)

print("--- Tensor Shapes ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")

# 6. Run Tuner (Downsampling handled internally)
final_model, params, y_train_pred, y_test_pred, X_train_3d, X_test_3d = tune_lstm_with_gwo_advanced(
    X_train, y_train, X_test, y_test
)

# 7. Unscale Predictions
def inverse_transform_target(scaled_1d_array, scaler, target_index, n_features):
    dummy_matrix = np.zeros((len(scaled_1d_array), n_features))
    dummy_matrix[:, target_index] = scaled_1d_array.flatten()
    unscaled_matrix = scaler.inverse_transform(dummy_matrix)
    return unscaled_matrix[:, target_index]

def mean_abs_error(y_true : List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    sum = 0
    for i in range(n):
        sum += abs(y_pred[i] - y_true[i])/y_true[i]
    return sum / n

num_features = train_data.shape[1]

y_train_unscaled = inverse_transform_target(y_train, scaler, target_col_index, num_features)
y_train_pred_unscaled = inverse_transform_target(y_train_pred, scaler, target_col_index, num_features)
y_test_unscaled = inverse_transform_target(y_test, scaler, target_col_index, num_features)
y_test_pred_unscaled = inverse_transform_target(y_test_pred, scaler, target_col_index, num_features)

mape_train = mean_abs_error(y_train_unscaled, y_train_pred_unscaled)
mape_test = mean_abs_error(y_test_unscaled, y_test_pred_unscaled)

print("MAPE training: ", mape_train)
print("MAPE test: ", mape_test)

final_model.save(f'GWO_best_model_v{version_num}.keras')

# 8. Align Datetime Indices
max_lag = 365 * 24
split_idx = int(len(df_original) * 0.8)

train_dates = df_original['datetime'].iloc[max_lag : split_idx].values
test_dates = df_original['datetime'].iloc[split_idx + max_lag : ].values

train_results_df = pd.DataFrame({
    'Datetime': train_dates,
    'Actual_Train': y_train_unscaled.flatten(),
    'Predicted_Train': y_train_pred_unscaled.flatten()
})
train_results_df.to_csv(f'LSTM_GWO_training_results_with_dates_v{version_num}.csv', index=False)
print("Training results with dates successfully saved!")

test_results_df = pd.DataFrame({
    'Datetime': test_dates,
    'Actual_Test': y_test_unscaled.flatten(),
    'Predicted_Test': y_test_pred_unscaled.flatten()
})
test_results_df.to_csv(f'LSTM_GWO_testing_results_with_dates_v{version_num}.csv', index=False)
print("Testing results with dates successfully saved!")