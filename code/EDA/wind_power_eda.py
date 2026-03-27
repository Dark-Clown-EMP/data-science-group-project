from typing import List

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
df = pd.read_csv("./baseline_training_results_with_dates.csv")
def mean_abs_error(y_true : List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    sum = 0
    for i in range(n):
        sum += abs(y_pred[i] - y_true[i])/y_true[i]
    return sum / n
y_pred = df['Predicted_Train']
y_actual = df['Actual_Train']
rmse = root_mean_squared_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)
mae = mean_absolute_error(y_actual,y_pred)
mape = mean_abs_error(y_actual, y_pred)
print("RMSE", rmse)
print("R2", r2)
print("MAE",mae)
print("MAPE",mape)