from typing import List
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from LSTM import tune_lstm_model
import pickle
import tensorflow as tf
import json
from GWO_LSTM_tuner import tune_lstm_with_gwo_advanced
from linear_regression_baseline import train_baseline_linear_model

def set_global_determinism(seed=42):
    # 1. Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Set Python's built-in random number generator
    random.seed(seed)
    
    # 3. Set NumPy's random number generator
    np.random.seed(seed)
    
    # 4. Set TensorFlow's random number generator
    tf.random.set_seed(seed)
    
    # 5. (Optional but recommended) Force TensorFlow to use deterministic GPU operations
    # Note: This might make training slightly slower, but guarantees identical runs
    tf.config.experimental.enable_op_determinism()

# Execute the lockdown
set_global_determinism(seed=42)


# 1. Load and Clean Data
df_original = pd.read_csv("./data/final_model_data.csv")
columns_drop = ['TSD', 'ENGLAND_WALES_DEMAND','EMBEDDED_WIND_GENERATION',
    'EMBEDDED_WIND_CAPACITY','EMBEDDED_SOLAR_GENERATION','EMBEDDED_SOLAR_CAPACITY',
    'NON_BM_STOR','PUMP_STORAGE_PUMPING','NET_IMPORTS','SCOTTISH_TRANSFER', 'datetime']

# FIX: Reassign the dataframe or use inplace=True
df = df_original.drop(columns=columns_drop)

data = df.values
scaler = MinMaxScaler(feature_range=(0,1))

# --- Configuration ---
# Assuming you want to predict the feature at column index 0. Change this to your actual target index.
target_col_index = 0
time_steps = [3, 6, 12, 24, 48, 3*30*24, 24*365]

# 2. Sequential Train/Test Split
split_idx = int(len(data) * 0.8)

train_data = data[:split_idx]
test_data = data[split_idx:]

# 3. Scale Data (Fit only on training data)
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

with open('minmax_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved successfully to 'minmax_scaler.pkl'")

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


# best_hps, best_model, X_train_3d, X_test_3d, y_test_pred, y_train_pred = tune_lstm_model(X_train, y_train, X_test, y_test)

# baseline_mode, y_test_pred, y_train_pred = train_baseline_linear_model(X_train, y_train, X_test)

final_model, params, y_train_pred, y_test_pred, X_train_3d, X_test_3d = tune_lstm_with_gwo_advanced(
    X_train, y_train, X_test, y_test
)


def inverse_transform_target(scaled_1d_array, scaler, target_index, n_features):
    """
    Reverses the MinMaxScaler for a single column by using a dummy array.
    """
    # 1. Create a dummy matrix of zeros with the same shape the scaler expects
    dummy_matrix = np.zeros((len(scaled_1d_array), n_features))
    
    # 2. Place your scaled 1D values into the target column
    dummy_matrix[:, target_index] = scaled_1d_array.flatten()
    
    # 3. Inverse transform the entire matrix
    unscaled_matrix = scaler.inverse_transform(dummy_matrix)
    
    # 4. Extract and return just your unscaled target column
    return unscaled_matrix[:, target_index]


def mean_abs_error(y_true : List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    sum = 0
    for i in range(n):
        sum += abs(y_pred[i] - y_true[i])/y_true[i]
    return sum / n

# Get the total number of columns your scaler was originally trained on
num_features = train_data.shape[1]

# Unscale the Training Data
y_train_unscaled = inverse_transform_target(y_train, scaler, target_col_index, num_features)
y_train_pred_unscaled = inverse_transform_target(y_train_pred, scaler, target_col_index, num_features)

# Unscale the Test Data
y_test_unscaled = inverse_transform_target(y_test, scaler, target_col_index, num_features)
y_test_pred_unscaled = inverse_transform_target(y_test_pred, scaler, target_col_index, num_features)

mape_train = mean_abs_error(y_train_unscaled, y_train_pred_unscaled)
mape_test = mean_abs_error(y_test_unscaled, y_test_pred_unscaled)

print("MAPE training: ", mape_train)
print("MAPE test: ", mape_test)

# best_model.save('best_lstm_model.keras')
# print("Model saved successfully to 'best_lstm_model.keras'")

final_model.save('GWO_best_model.keras')

max_lag = 365 * 24  # Your largest lag (8760 hours)
split_idx = int(len(df_original) * 0.8) # Your train/test split point

# 2. Slice the datetimes to match the LSTM sequences perfectly
# Training dates: Start after the initial max_lag, stop at the split index
train_dates = df_original['datetime'].iloc[max_lag : split_idx].values

# Testing dates: Start at the split index PLUS the max_lag, go to the end
test_dates = df_original['datetime'].iloc[split_idx + max_lag : ].values

# --- 3. Build the Training DataFrame ---
train_results_df = pd.DataFrame({
    'Datetime': train_dates,
    'Actual_Train': y_train_unscaled.flatten(),
    'Predicted_Train': y_train_pred_unscaled.flatten()
})

# Save to CSV
train_results_df.to_csv('LSTM_GWO_training_results_with_dates.csv', index=False)
print("Training results with dates successfully saved!")

# --- 4. Build the Testing DataFrame ---
test_results_df = pd.DataFrame({
    'Datetime': test_dates,
    'Actual_Test': y_test_unscaled.flatten(),
    'Predicted_Test': y_test_pred_unscaled.flatten()
})

# Save to CSV
test_results_df.to_csv('LSTM_GWO_testing_results_with_dates.csv', index=False)
print("Testing results with dates successfully saved!")


# # 1. Print the hyperparameters neatly to the console
# print("--- Best Hyperparameters ---")
# best_params_dict = best_hps.values

# for param_name, param_value in best_params_dict.items():
#     print(f"{param_name}: {param_value}")
# print("----------------------------")

# # 2. Save the hyperparameters to a JSON file for future reference
# with open('best_hyperparameters.json', 'w') as f:
#     json.dump(best_params_dict, f, indent=4)
    
# print("Hyperparameters successfully saved to 'best_hyperparameters.json'")