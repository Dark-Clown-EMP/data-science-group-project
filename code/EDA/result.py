import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# ────────────────────────────────────────────────
# 1. Load and prepare data
# ────────────────────────────────────────────────
# Baseline test predictions
df_b = pd.read_csv('../data/baseline_test_predictions.csv')
df_b['Datetime'] = pd.to_datetime(df_b['Datetime'])
df_b = df_b.set_index('Datetime')
df_b = df_b.rename(columns={'Actual_Test': 'actual_b', 'Predicted_Test': 'b_pred'})
df_b = df_b[['actual_b', 'b_pred']]

# RF test predictions
df_rf = pd.read_csv('../data/rf_test_predictions.csv')
df_rf['datetime'] = pd.to_datetime(df_rf['datetime'])
df_rf = df_rf.set_index('datetime')
df_rf = df_rf.rename(columns={'actual_ND': 'actual_rf', 'predicted_ND': 'rf_pred'})
df_rf = df_rf[['actual_rf', 'rf_pred']]

# XGBoost test predictions
df_xgb = pd.read_csv('../data/xgb_test_predictions.csv')
df_xgb['datetime'] = pd.to_datetime(df_xgb['datetime'])
df_xgb = df_xgb.set_index('datetime')
df_xgb = df_xgb.rename(columns={'actual_ND': 'actual_xgb', 'predicted_ND': 'xgb_pred'})
df_xgb = df_xgb[['actual_xgb', 'xgb_pred']]

# LSTM test predictions
df_lstm = pd.read_csv('../data/lstm_testing_results_with_dates.csv')
df_lstm['Datetime'] = pd.to_datetime(df_lstm['Datetime'])
df_lstm = df_lstm.set_index('Datetime')
df_lstm = df_lstm.rename(columns={'Actual_Test': 'actual_lstm', 'Predicted_Test': 'lstm_pred'})
df_lstm = df_lstm[['actual_lstm', 'lstm_pred']]

# LSTM + GWO test predictions
df_lstm_GWO = pd.read_csv('../data/lstm_GWO_testing_results_with_dates.csv')
df_lstm_GWO['Datetime'] = pd.to_datetime(df_lstm_GWO['Datetime'])
df_lstm_GWO = df_lstm_GWO.set_index('Datetime')
df_lstm_GWO = df_lstm_GWO.rename(columns={'Actual_Test': 'actual_lstm_GWO', 'Predicted_Test': 'lstm_GWO_pred'})
df_lstm_GWO = df_lstm_GWO[['actual_lstm_GWO', 'lstm_GWO_pred']]

# Merge on overlapping datetime index (inner join → Oct-Dec 2025 mostly)
test_df = df_rf.join(df_lstm, how='inner', rsuffix='_lstm')
test_df = test_df.join(df_lstm_GWO, how='inner', rsuffix='_lstm_GWO')
test_df = test_df.join(df_b, how='inner', rsuffix='_b')
test_df = test_df.join(df_xgb, how='inner', rsuffix='_xgb')

# If actual values differ slightly → prefer one (here: use RF's actual, or average, etc.)
# For simplicity we keep RF actual; adjust if needed
test_df = test_df.rename(columns={'actual': 'actual_rf'}).rename(columns={'actual_lstm': 'actual'})

print("Overlapping test set shape:", test_df.shape)
print("Date range:", test_df.index.min(), "to", test_df.index.max())

# Optional: full RF year metrics (without LSTM)
full_rf = df_rf.copy()
full_xgb = df_xgb.copy()

# ────────────────────────────────────────────────
# Metrics table (MAE, RMSE, R², MAPE)
# ────────────────────────────────────────────────

def mape(a, p):
    return 100 * np.mean(np.abs((a - p) / (a + 1e-10)))

metrics = []

# Baseline
metrics.append({
    'Model': 'Baseline', 
    'MAE': mean_absolute_error(test_df['actual_rf'], test_df['b_pred']),
    'RMSE': np.sqrt(mean_squared_error(test_df['actual_rf'], test_df['b_pred'])),
    'R²': r2_score(test_df['actual_rf'],test_df['b_pred']),
    'MAPE (%)': mape(test_df['actual_rf'],test_df['b_pred'])
})

# Random Forest
metrics.append({
    'Model': 'Random Forest',
    'MAE': mean_absolute_error(full_rf['actual_rf'],full_rf['rf_pred']),
    'RMSE': np.sqrt(mean_squared_error(full_rf['actual_rf'],full_rf['rf_pred'])),
    'R²': r2_score(full_rf['actual_rf'],full_rf['rf_pred']),
    'MAPE (%)': mape(full_rf['actual_rf'],full_rf['rf_pred'])
})

# XGBoost
metrics.append({
    'Model': 'XGBoost',
    'MAE': mean_absolute_error(full_xgb['actual_xgb'],full_xgb['xgb_pred']),
    'RMSE': np.sqrt(mean_squared_error(full_xgb['actual_xgb'],full_xgb['xgb_pred'])),
    'R²': r2_score(full_xgb['actual_xgb'],full_xgb['xgb_pred']),
    'MAPE (%)': mape(full_xgb['actual_xgb'],full_xgb['xgb_pred'])
})

# LSTM + GWO
metrics.append({
    'Model': 'LSTM + GWO',
    'MAE': mean_absolute_error(test_df['actual_rf'],test_df['lstm_GWO_pred']),
    'RMSE': np.sqrt(mean_squared_error(test_df['actual_rf'],test_df['lstm_GWO_pred'])),
    'R²': r2_score(test_df['actual_rf'],test_df['lstm_GWO_pred']),
    'MAPE (%)': mape(test_df['actual_rf'],test_df['lstm_GWO_pred'])
})

metrics_df = pd.DataFrame(metrics).round(3)

print("\nTest Metrics:")
print(metrics_df.to_string(index=False))

metrics_df.to_csv(
    '../data/final_test_metrics.csv',
    index=False
)
# ────────────────────────────────────────────────
# 3. Overlay time series (only overlap period)
# ────────────────────────────────────────────────

# Pick a week with interesting variation (adjust dates as needed)
start = '2025-12-15'
end   = '2025-12-22'

week = test_df.loc[start:end]

# plot actual vs. four models (Baseline, RF, XGBoost, LSTM) for the week
plt.figure(figsize=(14, 6))
plt.plot(week.index, week['actual_rf'], label='Actual', color='gray', lw=1.5, ls = '--')
plt.plot(week.index, week['b_pred'],   label='Baseline', color='darkred', lw=2)
plt.plot(week.index, week['rf_pred'],   label='RF', color='forestgreen', lw=2)
plt.plot(week.index, week['xgb_pred'],  label='XGBoost', color='darkorange', lw=2)
plt.plot(week.index, week['lstm_GWO_pred'], label='LSTM + GWO', color='dodgerblue', lw=2)

plt.title(f'Actual vs Predicted: Week ({start} to {end})')
plt.xlabel('Date & Time')
plt.ylabel('National Demand (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('Figures/overlay_comparison.png', dpi=180)


# ────────────────────────────────────────────────
# Clearer Plots
# ────────────────────────────────────────────────

palette= {
    'Baseline': '#8B0000',
    'Random Forest': '#228B22',
    'XGBoost': '#FF8C00',
    'LSTM + GWO': '#1E90FF'
}

model_map = {
    'Baseline': 'b_pred',
    'Random Forest': 'rf_pred',
    'XGBoost': 'xgb_pred',
    'LSTM + GWO': 'lstm_GWO_pred'
}

# Prepare data for facet plots
rows = []

for model_name, col in model_map.items():

    rows.append(pd.DataFrame({
        'time': week.index,
        'Actual': week['actual_rf'],
        'Prediction': week[col],
        'Model': model_name
    }))

plot_df = pd.concat(rows)

# Metric function
def mape(a, p):
    return 100 * np.mean(np.abs((a - p) / (a + 1e-10)))

week_metrics = {}

for model_name, col in model_map.items():

    y_true = week['actual_rf']
    y_pred = week[col]

    week_metrics[model_name] = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }

# Plots
sns.set_style("whitegrid")

g = sns.FacetGrid(
    plot_df,
    col="Model",
    col_wrap=2,
    height=5.2,        
    aspect=1.6,        
    sharey=True
)

def plot_lines(data, color):

    model = data['Model'].iloc[0]

    ax = plt.gca()

    ax.plot(
        data['time'],
        data['Actual'],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Actual"
    )

    ax.plot(
        data['time'],
        data['Prediction'],
        color=palette[model],
        linewidth=2.2,
        label=model
    )

    m = week_metrics[model]

    text = (
        f"MAE: {m['MAE']:.1f}\n"
        f"RMSE: {m['RMSE']:.1f}\n"
        f"R²: {m['R2']:.3f}\n"
        f"MAPE: {m['MAPE']:.2f}%"
    )

    ax.text(
        0.015,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=9,              
        linespacing=1.1,
        bbox=dict(
            boxstyle="round,pad=0.25",   
            facecolor="white",
            alpha=0.75
        )
    )

g.map_dataframe(plot_lines)

# Legend
g.add_legend(
    title="Series",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
    borderaxespad=0,
    frameon=False
)

# Formatting
g.set_axis_labels("Date & Time", "National Demand (MW)")
g.set_titles("{col_name}")

for ax in g.axes.flatten():

    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "Figures/comparison.png",
    dpi=180,
    bbox_inches="tight"
)

# ────────────────────────────────────────────────
# 4. Parity plots
# ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True, sharey=True)

minv, maxv = test_df['actual_rf'].min(), test_df['actual_rf'].max()

# plot four models (Baseline, RF, XGBoost, LSTM + GWO) on same parity plot (overlap period)
axes[0,0].scatter(test_df['actual_rf'], test_df['b_pred'], alpha=0.5, s=35, color='darkred')
axes[0,0].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[0,0].set_title('Baseline – Parity Plot')
axes[0,0].set_xlabel('Actual ND (MW)')
axes[0,0].set_ylabel('Predicted ND (MW)')
axes[0,0].grid(alpha=0.3)

axes[0,1].scatter(test_df['actual_rf'], test_df['rf_pred'], alpha=0.5, s=35, color='green')
axes[0,1].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[0,1].set_title('Random Forest – Parity Plot')
axes[0,1].set_xlabel('Actual ND (MW)')
axes[0,1].grid(alpha=0.3)

axes[1,0].scatter(test_df['actual_rf'], test_df['xgb_pred'], alpha=0.5, s=35, color='orange')   
axes[1,0].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[1,0].set_title('XGBoost – Parity Plot')
axes[1,0].set_xlabel('Actual ND (MW)')
axes[1,0].grid(alpha=0.3)

axes[1,1].scatter(test_df['actual_rf'], test_df['lstm_GWO_pred'], alpha=0.5, s=35, color='blue')   
axes[1,1].plot([minv, maxv], [minv, maxv], 'r--', lw=1.5)
axes[1,1].set_title('LSTM + GWO – Parity Plot')
axes[1,1].set_xlabel('Actual ND (MW)')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/parity.png', dpi=180)

# ────────────────────────────────────────────────
# 5. Residual plots (overlap)
# ────────────────────────────────────────────────

# Calculate residuals for each model
test_df['b_resid'] = test_df['actual_rf'] - test_df['b_pred']
test_df['rf_resid'] = test_df['actual_rf'] - test_df['rf_pred']
test_df['xgb_resid'] = test_df['actual_rf'] - test_df['xgb_pred']
test_df['lstm_GWO_resid'] = test_df['actual_rf'] - test_df['lstm_GWO_pred']

fig, axes = plt.subplots(4, 2, figsize=(15, 10))

# Baseline residual vs time
axes[0,0].plot(test_df.index, test_df['b_resid'], color='darkred', lw=1)
axes[0,0].axhline(0, color='red', ls='--', alpha=0.6)
axes[0,0].set_title('Baseline Residuals vs Time')
axes[0,0].grid(alpha=0.3)

# Baseline residual vs actual
axes[0,1].scatter(test_df['actual_rf'], test_df['b_resid'], alpha=0.5, color='darkred')
axes[0,1].axhline(0, color='red', ls='--', alpha=0.6)
axes[0,1].set_title('Baseline Residuals vs Actual')
axes[0,1].set_xlabel('Actual ND')
axes[0,1].grid(alpha=0.3)

# RF residual vs time
axes[1,0].plot(test_df.index, test_df['rf_resid'], color='green', lw=1)
axes[1,0].axhline(0, color='red', ls='--', alpha=0.6)
axes[1,0].set_title('Random Forest Residuals vs Time')
axes[1,0].grid(alpha=0.3)

# RF residual vs actual
axes[1,1].scatter(test_df['actual_rf'], test_df['rf_resid'], alpha=0.5, color='green')
axes[1,1].axhline(0, color='red', ls='--', alpha=0.6)
axes[1,1].set_title('Random Forest Residuals vs Actual')
axes[1,1].set_xlabel('Actual ND')
axes[1,1].grid(alpha=0.3)

# XGBoost residual vs time
axes[2,0].plot(test_df.index, test_df['xgb_resid'], color='orange', lw=1)
axes[2,0].axhline(0, color='red', ls='--', alpha=0.6)
axes[2,0].set_title('XGBoost Residuals vs Time')
axes[2,0].grid(alpha=0.3)

# XGBoost residual vs actual
axes[2,1].scatter(test_df['actual_rf'], test_df['xgb_resid'], alpha=0.5, color='orange')
axes[2,1].axhline(0, color='red', ls='--', alpha=0.6)
axes[2,1].set_title('XGBoost Residuals vs Actual')
axes[2,1].set_xlabel('Actual ND')
axes[2,1].grid(alpha=0.3)

# LSTM + GWO residual vs time
axes[3,0].plot(test_df.index, test_df['lstm_GWO_resid'], color='dodgerblue', lw=1)
axes[3,0].axhline(0, color='red', ls='--', alpha=0.6)
axes[3,0].set_title('LSTM + GWO Residuals vs Time')
axes[3,0].grid(alpha=0.3)

# LSTM + GWO residual vs actual
axes[3,1].scatter(test_df['actual_rf'], test_df['lstm_GWO_resid'], alpha=0.5, color='dodgerblue')
axes[3,1].axhline(0, color='red', ls='--', alpha=0.6)
axes[3,1].set_title('LSTM + GWO Residuals vs Actual')
axes[3,1].set_xlabel('Actual ND')
axes[3,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/residuals.png', dpi=180)

plt.show()

print("All plots and metrics saved. Check 'final_test_metrics.csv' for numbers.")