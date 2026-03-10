import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# ────────────────────────────────────────────────
# 1. Load and prepare data
# ────────────────────────────────────────────────

# RF test predictions
df_rf = pd.read_csv('../data/rf_test_predictions.csv')
df_rf = df_rf[df_rf['dataset'] == 'test'].copy()
df_rf['datetime'] = pd.to_datetime(df_rf['datetime'])
df_rf = df_rf.set_index('datetime')
df_rf = df_rf.rename(columns={'actual_ND': 'actual', 'predicted_ND': 'rf_pred'})
df_rf = df_rf[['actual', 'rf_pred']]

# LSTM test predictions
df_lstm = pd.read_csv('../data/lstm_testing_results_with_dates.csv')
df_lstm['Datetime'] = pd.to_datetime(df_lstm['Datetime'])
df_lstm = df_lstm.set_index('Datetime')
df_lstm = df_lstm.rename(columns={'Actual_Test': 'actual', 'Predicted_Test': 'lstm_pred'})
df_lstm = df_lstm[['actual', 'lstm_pred']]

# LSTM + GWO test predictions
df_lstm_GWO = pd.read_csv('../data/lstm_GWO_testing_results_with_dates.csv')
df_lstm_GWO['Datetime'] = pd.to_datetime(df_lstm_GWO['Datetime'])
df_lstm_GWO = df_lstm_GWO.set_index('Datetime')
df_lstm_GWO = df_lstm_GWO.rename(columns={'Actual_Test': 'actual', 'Predicted_Test': 'lstm_GWO_pred'})
df_lstm_GWO = df_lstm_GWO[['actual', 'lstm_GWO_pred']]

# Merge on overlapping datetime index (inner join → Oct-Dec 2025 mostly)
test_df = df_rf.join(df_lstm, how='inner', rsuffix='_lstm')
test_df_GWO = df_rf.join(df_lstm_GWO, how='inner', rsuffix='_lstm_GWO')

# If actual values differ slightly → prefer one (here: use RF's actual, or average, etc.)
# For simplicity we keep RF actual; adjust if needed
test_df = test_df.rename(columns={'actual': 'actual_rf'}).rename(columns={'actual_lstm': 'actual'})
test_df_GWO = test_df_GWO.rename(columns={'actual': 'actual_rf'}).rename(columns={'actual_lstm_GWO': 'actual'})

print("Overlapping test set shape:", test_df.shape)
print("Date range:", test_df.index.min(), "to", test_df.index.max())

# Optional: full RF year metrics (without LSTM)
full_rf = df_rf.copy()

# ────────────────────────────────────────────────
# 2. Metrics table (MAE, RMSE, sMAPE, Peak MAE)
# ────────────────────────────────────────────────

def smape(a, p):
    return 100 * np.mean(2 * np.abs(p - a) / (np.abs(a) + np.abs(p) + 1e-10))

metrics = []

# Overlapping period (both models)
metrics.append({
    'Model': 'Random Forest (overlap)',
    'MAE': mean_absolute_error(test_df['actual_rf'], test_df['rf_pred']),
    'RMSE': np.sqrt(mean_squared_error(test_df['actual_rf'], test_df['rf_pred'])),
    'sMAPE (%)': smape(test_df['actual_rf'], test_df['rf_pred']),
    'Peak MAE': np.abs(test_df['actual_rf'].max() - test_df['rf_pred'][test_df['actual_rf'].idxmax()])
})

metrics.append({
    'Model': 'LSTM (overlap)',
    'MAE': mean_absolute_error(test_df['actual_rf'], test_df['lstm_pred']),
    'RMSE': np.sqrt(mean_squared_error(test_df['actual_rf'], test_df['lstm_pred'])),
    'sMAPE (%)': smape(test_df['actual_rf'], test_df['lstm_pred']),
    'Peak MAE': np.abs(test_df['actual_rf'].max() - test_df['lstm_pred'][test_df['actual_rf'].idxmax()])
})

metrics.append({
    'Model': 'LSTM + GWO(overlap)',
    'MAE': mean_absolute_error(test_df_GWO['actual_rf'], test_df_GWO['lstm_GWO_pred']),
    'RMSE': np.sqrt(mean_squared_error(test_df_GWO['actual_rf'], test_df_GWO['lstm_GWO_pred'])),
    'sMAPE (%)': smape(test_df_GWO['actual_rf'], test_df_GWO['lstm_GWO_pred']),
    'Peak MAE': np.abs(test_df_GWO['actual_rf'].max() - test_df_GWO['lstm_GWO_pred'][test_df_GWO['actual_rf'].idxmax()])
})

# Full RF test year (for reference)
metrics.append({
    'Model': 'Random Forest (full year 2025)',
    'MAE': mean_absolute_error(full_rf['actual'], full_rf['rf_pred']),
    'RMSE': np.sqrt(mean_squared_error(full_rf['actual'], full_rf['rf_pred'])),
    'sMAPE (%)': smape(full_rf['actual'], full_rf['rf_pred']),
    'Peak MAE': np.abs(full_rf['actual'].max() - full_rf['rf_pred'][full_rf['actual'].idxmax()])
})

metrics_df = pd.DataFrame(metrics).round(2)
print("\nTest Metrics:")
print(metrics_df.to_string(index=False))
metrics_df.to_csv('../data/final_test_metrics.csv', index=False)

# ────────────────────────────────────────────────
# 3. Overlay time series (only overlap period)
# ────────────────────────────────────────────────

# Pick a week with interesting variation (adjust dates as needed)
start = '2025-12-15'
end   = '2025-12-22'

week = test_df.loc[start:end]

plt.figure(figsize=(14, 6))
plt.plot(week.index, week['actual_rf'], label='Actual', color='black', lw=2)
plt.plot(week.index, week['rf_pred'],   label='RF', color='forestgreen', lw=2.2)
plt.plot(week.index, week['lstm_pred'], label='LSTM', color='dodgerblue', lw=2.2)

plt.title(f'Actual vs Predicted: Week ({start} to {end})')
plt.xlabel('Date & Time')
plt.ylabel('National Demand (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('Figures/overlay_one_week.png', dpi=180)

# With LSTM + GWO instead

# Pick a week with interesting variation (adjust dates as needed)
start = '2025-12-15'
end   = '2025-12-22'

week = test_df_GWO.loc[start:end]

plt.figure(figsize=(14, 6))
plt.plot(week.index, week['actual_rf'], label='Actual', color='black', lw=2)
plt.plot(week.index, week['rf_pred'],   label='RF', color='forestgreen', lw=2.2)
plt.plot(week.index, week['lstm_GWO_pred'], label='LSTM + GWO', color='orange', lw=2.2)

plt.title(f'Actual vs Predicted: Week ({start} to {end})')
plt.xlabel('Date & Time')
plt.ylabel('National Demand (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('Figures/overlay_one_week_GWO.png', dpi=180)

# Comparing LSTMs

# Pick a week with interesting variation (adjust dates as needed)
start = '2025-12-15'
end   = '2025-12-22'

week = test_df_GWO.loc[start:end]
week_1 = test_df.loc[start:end]

plt.figure(figsize=(14, 6))
plt.plot(week.index, week['actual_rf'], label='Actual', color='black', lw=2)
plt.plot(week.index, week['rf_pred'],   label='RF', color='forestgreen', lw=2.2)
plt.plot(week_1.index, week_1['lstm_pred'], label='LSTM', color='blue', lw=2.2)
plt.plot(week.index, week['lstm_GWO_pred'], label='LSTM + GWO', color='orange', lw=2.2)

plt.title(f'Actual vs Predicted: Week ({start} to {end})')
plt.xlabel('Date & Time')
plt.ylabel('National Demand (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
#plt.savefig('Figures/overlay_one_week_GWO.png', dpi=180)

# ────────────────────────────────────────────────
# 4. Parity plots
# ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

minv, maxv = test_df['actual_rf'].min(), test_df['actual_rf'].max()

axes[0].scatter(test_df['actual_rf'], test_df['rf_pred'], alpha=0.5, s=35, color='green')
axes[0].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[0].set_title('Random Forest – Parity Plot (overlap)')
axes[0].set_xlabel('Actual ND (MW)')
axes[0].set_ylabel('Predicted ND (MW)')
axes[0].grid(alpha=0.3)

axes[1].scatter(test_df['actual_rf'], test_df['lstm_pred'], alpha=0.5, s=35, color='blue')
axes[1].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[1].set_title('LSTM – Parity Plot (overlap)')
axes[1].set_xlabel('Actual ND (MW)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/parity_both.png', dpi=180)

# With LSTM + GWO instead
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

minv, maxv = test_df_GWO['actual_rf'].min(), test_df_GWO['actual_rf'].max()

axes[0].scatter(test_df_GWO['actual_rf'], test_df_GWO['rf_pred'], alpha=0.5, s=35, color='green')
axes[0].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[0].set_title('Random Forest – Parity Plot (overlap)')
axes[0].set_xlabel('Actual ND (MW)')
axes[0].set_ylabel('Predicted ND (MW)')
axes[0].grid(alpha=0.3)

axes[1].scatter(test_df_GWO['actual_rf'], test_df_GWO['lstm_GWO_pred'], alpha=0.5, s=35, color='orange')
axes[1].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[1].set_title('LSTM + GWO – Parity Plot (overlap)')
axes[1].set_xlabel('Actual ND (MW)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/parity_both_GWO.png', dpi=180)
# ────────────────────────────────────────────────
# 5. Residual plots (overlap)
# ────────────────────────────────────────────────

test_df['rf_resid']   = test_df['actual_rf'] - test_df['rf_pred']
test_df['lstm_resid'] = test_df['actual_rf'] - test_df['lstm_pred']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# RF residual vs time
axes[0,0].plot(test_df.index, test_df['rf_resid'], color='green', lw=1)
axes[0,0].axhline(0, color='red', ls='--', alpha=0.6)
axes[0,0].set_title('RF Residuals vs Time')
axes[0,0].grid(alpha=0.3)

# RF residual vs actual
axes[0,1].scatter(test_df['actual_rf'], test_df['rf_resid'], alpha=0.5, color='green')
axes[0,1].axhline(0, color='red', ls='--', alpha=0.6)
axes[0,1].set_title('RF Residuals vs Actual')
axes[0,1].set_xlabel('Actual ND')
axes[0,1].grid(alpha=0.3)

# LSTM versions
axes[1,0].plot(test_df.index, test_df['lstm_resid'], color='blue', lw=1)
axes[1,0].axhline(0, color='red', ls='--', alpha=0.6)
axes[1,0].set_title('LSTM Residuals vs Time')
axes[1,0].grid(alpha=0.3)

axes[1,1].scatter(test_df['actual_rf'], test_df['lstm_resid'], alpha=0.5, color='blue')
axes[1,1].axhline(0, color='red', ls='--', alpha=0.6)
axes[1,1].set_title('LSTM Residuals vs Actual')
axes[1,1].set_xlabel('Actual ND')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/residuals_both.png', dpi=180)

# With LSTM + GWO instead
test_df_GWO['rf_resid']   = test_df_GWO['actual_rf'] - test_df_GWO['rf_pred']
test_df_GWO['lstm_GWO_resid'] = test_df_GWO['actual_rf'] - test_df_GWO['lstm_GWO_pred']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# RF residual vs time
axes[0,0].plot(test_df_GWO.index, test_df_GWO['rf_resid'], color='green', lw=1)
axes[0,0].axhline(0, color='red', ls='--', alpha=0.6)
axes[0,0].set_title('RF Residuals vs Time')
axes[0,0].grid(alpha=0.3)

# RF residual vs actual
axes[0,1].scatter(test_df_GWO['actual_rf'], test_df_GWO['rf_resid'], alpha=0.5, color='green')
axes[0,1].axhline(0, color='red', ls='--', alpha=0.6)
axes[0,1].set_title('RF Residuals vs Actual')
axes[0,1].set_xlabel('Actual ND')
axes[0,1].grid(alpha=0.3)

# LSTM + GWO versions
axes[1,0].plot(test_df_GWO.index, test_df_GWO['lstm_GWO_resid'], color='orange', lw=1)
axes[1,0].axhline(0, color='red', ls='--', alpha=0.6)
axes[1,0].set_title('LSTM + GWO Residuals vs Time')
axes[1,0].grid(alpha=0.3)

axes[1,1].scatter(test_df_GWO['actual_rf'], test_df_GWO['lstm_GWO_resid'], alpha=0.5, color='orange')
axes[1,1].axhline(0, color='red', ls='--', alpha=0.6)
axes[1,1].set_title('LSTM + GWO Residuals vs Actual')
axes[1,1].set_xlabel('Actual ND')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/residuals_both_GWO.png', dpi=180)

# ────────────────────────────────────────────────
# 6. MAE by hour of day (overlap)
# ────────────────────────────────────────────────

test_df['hour'] = test_df.index.hour

hourly = test_df.groupby('hour').agg({
    'rf_resid': lambda x: np.mean(np.abs(x)),
    'lstm_resid': lambda x: np.mean(np.abs(x))
}).rename(columns={'rf_resid': 'RF_MAE', 'lstm_resid': 'LSTM_MAE'})

plt.figure(figsize=(10, 6))
hourly.plot(kind='bar', width=0.75)
plt.title('MAE by Hour of Day – Overlap Test Period')
plt.ylabel('MAE (MW)')
plt.grid(axis='y', alpha=0.3)
#plt.savefig('mae_by_hour_both.png', dpi=180)
plt.show()

print("All plots and metrics saved. Check 'final_test_metrics.csv' for numbers.")