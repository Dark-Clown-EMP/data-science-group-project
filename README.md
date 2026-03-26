# Forecasting Day-Ahead Net Electricity Demand in Great Britain

University of Birmingham group data science project.

## Setup

```bash
pip install -r requirements.txt
```

## Running the XGBoost pipeline

```bash
cd xgboost
export PYTHONPATH=src
python src/train_xgboost.py
```

This runs Optuna hyperparameter tuning (500 trials, 5-fold TimeSeriesSplit),
trains the final XGBoost model with early stopping, evaluates on the 2025
test set, and saves all outputs to `data/processed/` and `figures/`.

## Project structure

```
src/
  feature_engineering.py   Feature construction (lags, rolling stats, calendar, weather)
  evaluation.py            Evaluation metrics (RMSE, MAE, R², MAPE)
  train_xgboost.py         XGBoost training, tuning, and output pipeline
data/processed/
  final_model_data.csv     Authoritative hourly modelling dataset (2020-2025)
figures/
  XGBoost output figures
```
