import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('../data/final_model_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['month'] = df.index.month
df['hour'] = df.index.hour
df['day_of_week'] = df.index.day_name()
df['season'] = df['month'].apply(get_season)

# For HDD (Heating Degree Days, base 15.5°C, common for UK energy)
temp_cols = [col for col in df.columns if col.startswith('Temp_')]  # All temperature columns
df['avg_temp'] = df[temp_cols].mean(axis=1)
df['HDD'] = np.maximum(15.5 - df['avg_temp'], 0)

# Plot 1: ND Time Series + Rolling Mean
plt.figure(figsize=(14, 6))
plt.plot(df['ND'], label='ND', alpha=0.5)
plt.plot(df['ND'].rolling(window=24*7).mean(), label='7-Day Rolling Mean', color='red')
plt.title('National Demand (ND) Time Series with Rolling Mean')
plt.xlabel('Date')
plt.ylabel('ND (MW)')
plt.legend()
plt.grid(True)
plt.savefig('Figures/nd_time_series.png')

# Plot 2: ND Average by Month
plt.figure(figsize=(8,4))
df.groupby('month')['ND'].mean().plot(marker='o')
plt.title('Average National Demand by Month (2020-2025)')
plt.xlabel('Month')
plt.ylabel('National Demand')
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/national_demand_vs_month.png', dpi=300)

# Plot 3: ND Average by Season + Hourly
plt.figure(figsize=(9,4))

seasonal_profile = (
    df
    .groupby(['season', 'hour'])['ND']
    .mean()
    .unstack('season')
)

seasonal_profile.plot(
    ax=plt.gca(),
    marker='o'
)

plt.title('Average Hourly National Demand by Season')
plt.xlabel('Hour of Day')
plt.ylabel('National Demand')
plt.grid(ls='--', alpha=0.6)
plt.legend(title='Season', fontsize=9)
plt.tight_layout()
plt.savefig('Figures/seasonal_nd_vs_hour.png', dpi=300)

# Plot 4: ND Average by Day of Week
plt.figure(figsize=(9, 4))

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

weekly_avg = (
    df
    .groupby('day_of_week')['ND']
    .mean()
    .reindex(day_order)                    
)

weekly_avg.plot(ax=plt.gca(), marker='o')
plt.title('Average National Demand by Day of Week', fontsize=13, pad=12)
plt.xlabel('Day of Week', fontsize=11)
plt.ylabel('National Demand (MW)', fontsize=11)
plt.grid(ls='--', alpha=0.6)
plt.xticks(rotation=0)                       
plt.legend().set_visible(False)              
plt.tight_layout()
plt.savefig('Figures/nd_avg_by_day_of_week.png', dpi=150, bbox_inches='tight')


# Plot 5: ND vs. Temp
# Option 1: Temp vs ND (binned, by season)
g = sns.FacetGrid(df, col='season', col_wrap=2, height=5)
g.map(sns.histplot, 'avg_temp', 'ND', bins=50, kde=False)  # Or use hexbin: g.map(plt.hexbin, 'avg_temp', 'ND', gridsize=30, cmap='Blues')
g.add_legend()
g.set_titles('Season: {col_name}')
g.set_axis_labels('Average Temperature (°C)', 'ND (MW)')
plt.savefig('Figures/temp_vs_nd_by_season.png')

# Option 2: HDD vs ND (simple scatter with regression)
plt.figure(figsize=(10, 6))
sns.regplot(x='HDD', y='ND', data=df, scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title('HDD vs ND with Regression Line')
plt.xlabel('HDD (Base 15.5°C)')
plt.ylabel('ND (MW)')
#plt.savefig('hdd_vs_nd.png')

# Plot 6: Corr Heatmap
weather_cols = [col for col in df.columns if 'Temp_' in col or 'Wind10m_' in col or 'Solar_' in col]
corr_df = df[weather_cols + ['ND']].copy()
corr_df['ND_lag24'] = corr_df['ND'].shift(-24)  # Lag +24 (weather predicting future ND)

plt.figure(figsize=(16, 12))
sns.heatmap(corr_df.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap: ND vs Regional Weather (with ND Lag +24)')
plt.savefig('Figures/correlation_heatmap.png')

# Show plots
plt.show()