import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

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

# Plot 1: Violin Plots of necessary features
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.violinplot(data=df, y='ND', ax=axes[0])
axes[0].set_title('Distribution of National Demand')
axes[0].set_ylabel('National Demand (MW)')

sns.violinplot(data=df, y='EMBEDDED_SOLAR_GENERATION', ax=axes[1])
axes[1].set_title('Distribution of Embedded Solar')
axes[1].set_ylabel('Embedded Solar (MW)')

sns.violinplot(data=df, y='EMBEDDED_WIND_GENERATION', ax=axes[2])
axes[2].set_title('Distribution of Embedded Wind')
axes[2].set_ylabel('Embedded Wind (MW)')

plt.tight_layout()
plt.savefig('Figures/distribution.png', dpi=300)

# Plot 2: ND Time Series + Rolling Mean
plt.figure(figsize=(14, 6))
plt.plot(df['ND'], label='ND', alpha=0.5)
plt.plot(df['ND'].rolling(window=24*7).mean(), label='7-Day Rolling Mean', color='red')
plt.title('National Demand (ND) Time Series with Rolling Mean')
plt.xlabel('Date')
plt.ylabel('ND (MW)')
plt.legend()
plt.grid(True)
plt.savefig('Figures/nd_time_series.png')

# Plot 3: Average National Demand by Month
plt.figure(figsize=(8,4))
df.groupby('month')['ND'].mean().plot(marker='o')
plt.title('Average National Demand by Month (2020-2025)')
plt.xlabel('Month')
plt.ylabel('National Demand')
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/national_demand_vs_month.png', dpi=300)

# Plot 4: ND Average by Season + Hourly
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

# Plot 5: ND Average by Day of Week
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

# Plot 6: Corr Heatmap
weather_cols = [col for col in df.columns if 'Temp_' in col or 'Wind10m_' in col or 'Solar_' in col]
corr_df = df[weather_cols + ['ND']].copy()
corr_df['ND_lag24'] = corr_df['ND'].shift(-24)  # Lag +24 (weather predicting future ND)

plt.figure(figsize=(16, 12))
sns.heatmap(corr_df.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap: ND vs Regional Weather (with ND Lag +24)')
plt.savefig('Figures/correlation_heatmap.png')

# Plot 7: Solar correlations (embedded_solar and solar regions)
solar_cols = [col for col in df.columns if 'Solar_' in col or 'embedded_solar' in col]
corr_df = df[solar_cols + ['EMBEDDED_SOLAR_GENERATION']].copy()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap='flare', vmin=-1, vmax=1)
plt.title('Correlation Heatmap: Solar vs Embedded Solar')
plt.tight_layout()
plt.savefig('Figures/solar_correlation_heatmap.png')

# Plot 8:Wind correlations (embedded_wind and wind regions)
wind_cols = [col for col in df.columns if 'Wind10m_' in col or 'embedded_wind' in col]
corr_df = df[wind_cols + ['EMBEDDED_WIND_GENERATION']].copy()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap='crest', vmin=-1, vmax=1)
plt.title('Correlation Heatmap: Wind vs Embedded Wind')
plt.tight_layout()
plt.savefig('Figures/wind_correlation_heatmap.png')

# Plot 9 & 10: Plots to show that combining MET and NESO data is valid.
# Plot 9: Embedded Solar vs Solar Radiation in East Midlands
plt.figure(figsize=(10, 6))
correlation_solar = df['Solar_Eng_East_Midlands'].corr(df['EMBEDDED_SOLAR_GENERATION'])

slope, intercept = np.polyfit(df['Solar_Eng_East_Midlands'], df['EMBEDDED_SOLAR_GENERATION'], 1)
trend_line = slope * df['Solar_Eng_East_Midlands'] + intercept

plt.scatter(df['Solar_Eng_East_Midlands'], df['EMBEDDED_SOLAR_GENERATION'], alpha=0.5, s=10, color='orange')
plt.plot(df['Solar_Eng_East_Midlands'], trend_line, color='black', label='Trend Line')
plt.xlabel('Solar Radiation - East Midlands (W/m²)')
plt.ylabel('Embedded Solar Generation (MW)')
plt.title('Embedded Solar Generation vs East Midlands Solar Radiation')
plt.text(0.05, 0.95, f'Correlation: {correlation_solar:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/solar_corr.png', dpi=300)

# Plot 10: Embedded Wind vs Wind Speed in North Cumbria
plt.figure(figsize=(10, 6))
correlation_wind = df['Wind10m_Eng_North_Cumbria'].corr(df['EMBEDDED_WIND_GENERATION'])

x = df['Wind10m_Eng_North_Cumbria']
y = df['EMBEDDED_WIND_GENERATION']

lowess_fit = lowess(y, x, frac=0.08)   # smooth curve

plt.scatter(x, y, alpha=0.5, s=10, color='green')
plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color='black', label='Trend Line')

plt.xlabel('Wind Speed - North Cumbria (m/s)')
plt.ylabel('Embedded Wind Generation (MW)')
plt.title('Embedded Wind Generation vs North Cumbria Wind Speed')
plt.text(0.05, 0.95, f'Correlation: {correlation_wind:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/wind_corr.png', dpi=300)

# Plot 11: ND vs. Temp
plt.figure(figsize=(10, 6))
correlation_temp = df['Temp_Eng_East_Norfolk'].corr(df['ND'])

x = df['Temp_Eng_East_Norfolk']
y = df['ND']

lowess_fit = lowess(y, x, frac=0.08)

plt.scatter(x, y, alpha=0.5, s=10, color='darkred')
plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color='black', label='Trend Line')

plt.xlabel('Temperature - East Norfolk (°C)')
plt.ylabel('National Demand (MW)')
plt.title('National Demand vs East Norfolk Temperature')

plt.text(
    0.05, 0.95,
    f'Correlation: {correlation_temp:.4f}',
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/temp_corr.png', dpi=300)

# Plot 12-14: Regional differences in solar/wind/temp (line plots of monthly averages for each region)
# Define the regions for each variable
wind_cols = ['Wind10m_Scot_Highlands', 'Wind10m_Wales_North_Gwynedd', 
             'Wind10m_Eng_East_Suffolk', 'Wind10m_Eng_East_Norfolk']

solar_cols = ['Solar_Eng_London', 'Solar_Eng_East_Norfolk', 
              'Solar_Eng_East_Suffolk', 'Solar_Eng_East_Midlands']

temp_cols = ['Temp_Scot_Highlands', 'Temp_Wales_North_Gwynedd', 
             'Temp_Eng_East_Suffolk', 'Temp_Eng_East_Norfolk']


# Function to create and save monthly plots
def create_regional_variation_plot(df, cols, variable_name, filename):
    
    plt.figure(figsize=(10,6))

    for col in cols:
        monthly_avg = df.groupby('month')[col].mean()
        plt.plot(monthly_avg.index, monthly_avg.values, label=col)

    plt.title(f'{variable_name} Monthly Regional Distribution')
    plt.xlabel('Month')
    plt.ylabel(variable_name)
    plt.xticks(range(1,13))
    plt.grid(ls='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(f"Figures/{filename}", dpi=300)

    plt.show()


# Wind plot
create_regional_variation_plot(
    df, wind_cols,
    "Wind Speed (m/s)",
    "wind_region_dist.png"
)

# Solar plot
create_regional_variation_plot(
    df, solar_cols,
    "Solar Radiation (W/m²)",
    "solar_region_dist.png"
)

# Temperature plot
create_regional_variation_plot(
    df, temp_cols,
    "Temperature (°C)",
    "temp_region_dist.png"
)

# Show plots
plt.show()