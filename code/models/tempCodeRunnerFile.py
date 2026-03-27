import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

csvList = ['training', 'testing']
columnList = ['Train', 'Test']

for i, str in enumerate(csvList):
# 1. Load the dataset
    df = pd.read_csv(f'LSTM_GWO_{str}_results_with_dates.csv')


    y_true = df[f'Actual_{columnList[i]}']
    y_pred = df[f'Predicted_{columnList[i]}']
    print(f'\n\n{str} scores')
    # 3. Calculate Scores
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # 4. Print results
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"MAPE:     {mape:.4%}") # Displayed as a percentage