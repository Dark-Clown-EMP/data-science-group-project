import pandas as pd
import tensorflow as tf
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

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


def run_permutation_importance(model, X_test_3d, y_test, feature_names):
    print("Calculating baseline performance...")
    
    # 1. Get Baseline Error (Fixed the tuple bug here)
    baseline_preds = model.predict(X_test_3d, verbose=0)
    baseline_mape = mean_absolute_error(y_test.flatten(), baseline_preds.flatten())
    print(f"Baseline Test MAE: {baseline_mape:.4f}\n")

    importances = []
    num_features = X_test_3d.shape[2] 

    print("Running permutation tests (this may take a minute)...")
    for i in range(num_features):
        X_test_shuffled = X_test_3d.copy()
        
        # Shuffle ONLY the current feature
        np.random.shuffle(X_test_shuffled[:, 0, i])
        
        # Predict with sabotaged feature
        shuffled_preds = model.predict(X_test_shuffled, verbose=0)
        shuffled_mape = mean_absolute_error(y_test.flatten(), shuffled_preds.flatten())
        
        importance_score = shuffled_mape - baseline_mape
        importances.append(importance_score)
        
        print(f"Sabotaged {feature_names[i]}: New MAE = {shuffled_mape:.4f} (+{importance_score:.4f})")

    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Score_MAE_Increase': importances
    })
    
    # Sort ascending (least important at the top of the dataframe, most important at the bottom)
    results_df = results_df.sort_values(by='Importance_Score_MAE_Increase', ascending=True)
    
    # --- NEW: Extract only the top 10 most important features for the plot ---
    top_10_df = results_df.tail(15)
    
    # 3. Plot the Results
    plt.figure(figsize=(10, 6)) # Reduced height since there are fewer bars
    plt.barh(top_10_df['Feature'], top_10_df['Importance_Score_MAE_Increase'], color='darkred')
    
    # Add data labels to the bars for a more professional look
    for index, value in enumerate(top_10_df['Importance_Score_MAE_Increase']):
        plt.text(value, index, f' {value:.4f}', va='center', fontsize=10)

    plt.xlabel('Increase in Error (MAE) when feature is shuffled')
    plt.title('Top 10 LSTM Permutation Feature Importances')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Pad the x-axis limits slightly so the text labels don't get cut off
    plt.xlim(0, top_10_df['Importance_Score_MAE_Increase'].max() * 1.15)
    
    plt.tight_layout()
    plt.savefig('LSTM_Top10_Feature_Importance.png', dpi=300)
    plt.show()

    # Still return the full dataframe so your CSV saves all 43 features
    return results_df

if __name__ == "__main__":
    # --- 1. Load Data ---
    print("Loading data...")
    df_original = pd.read_csv("./data/final_model_data.csv")
    columns_drop = ['TSD', 'ENGLAND_WALES_DEMAND','EMBEDDED_WIND_GENERATION',
        'EMBEDDED_WIND_CAPACITY','EMBEDDED_SOLAR_GENERATION','EMBEDDED_SOLAR_CAPACITY',
        'NON_BM_STOR','PUMP_STORAGE_PUMPING','NET_IMPORTS','SCOTTISH_TRANSFER', 'datetime']

    feature_list = ['Temp_Scot_Highlands','Wind10m_Scot_Highlands','Temp_Scot_Aberdeenshire',
        'Wind10m_Scot_Aberdeenshire','Temp_Scot_Glasgow_West','Temp_Scot_Edinburgh_East',
        'Temp_Scot_Borders','Wind10m_Scot_Borders','Temp_Wales_North_Gwynedd',
        'Wind10m_Wales_North_Gwynedd','Temp_Wales_South_Cardiff','Temp_Eng_North_Tyne',
        'Temp_Eng_North_Cumbria','Wind10m_Eng_North_Cumbria','Temp_Eng_Yorkshire',
        'Wind10m_Eng_Yorkshire','Temp_Eng_Manchester','Temp_Eng_West_Midlands',
        'Temp_Eng_East_Midlands','Solar_Eng_East_Midlands','Temp_Eng_East_Norfolk',
        'Wind10m_Eng_East_Norfolk','Solar_Eng_East_Norfolk','Temp_Eng_East_Suffolk',
        'Wind10m_Eng_East_Suffolk','Solar_Eng_East_Suffolk','Temp_Eng_London',
        'Solar_Eng_London','Temp_Eng_South_Kent','Solar_Eng_South_Kent',
        'Temp_Eng_South_Hampshire','Solar_Eng_South_Hampshire','Temp_Eng_South_Cornwall',
        'Solar_Eng_South_Cornwall','Temp_Eng_South_Bristol','Solar_Eng_South_Bristol',
        'lag_3','lag_6','lag_12','lag_24','lag_48','lag_3m','lag_8760']
        # 'lag_24', 'lag_48', 'lag_72', 'lag_168', 'lag_720', 'lag_8760']

    df = df_original.drop(columns=columns_drop)
    data = df.values

    # --- 2. Split First ---
    split_idx = int(len(data) * 0.8)
    test_data = data[split_idx:]

    # --- 3. Scale Second ---
    with open('minmax_scaler.pkl', 'rb') as f:
        scaler = pk.load(f)
    print("Scaler loaded successfully.")

    test_data_scaled = scaler.transform(test_data)

    # --- 4. Generate Lags Third ---
    target_col_index = 0
    # time_steps = [24, 48, 72, 7*24, 30*24, 24*365]
    time_steps = [3, 6, 12, 24, 48, 3*30*24, 24*365]

    print("Generating lags...")
    X_test_scaled, y_test_scaled = create_y_lags_2d(test_data_scaled, target_col_index, time_steps)
    X_test_3d = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Sanity Check
    assert len(feature_list) == X_test_3d.shape[2], f"CRITICAL: You listed {len(feature_list)} features, but the model expects {X_test_3d.shape[2]}."

    # --- 5. Load Model and Execute ---
    champion_model = tf.keras.models.load_model('GWO_best_model.keras')
    print("Model loaded successfully. Running Permutation Importance...")

    results_df = run_permutation_importance(champion_model, X_test_3d, y_test_scaled, feature_list)

    results_df.to_csv('./feature_importance_results.csv', index=False)
    print("Results saved successfully.")