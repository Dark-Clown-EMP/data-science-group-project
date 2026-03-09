# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Script is running...") 

# load the dataset
df = pd.read_csv('../data/final_model_data.csv')

# Set up of data and aggregates

# Ensure datetime index
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Time features
df['hour'] = df.index.hour
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofyear'] = df.index.dayofyear

# Weather aggregates
temp_cols = [c for c in df.columns if c.startswith('Temp_')]
wind_cols = [c for c in df.columns if c.startswith('Wind10m_')]
solar_cols = [c for c in df.columns if c.startswith('Solar_')]

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

# Plot 2: Average National Demand by Month
plt.figure(figsize=(8,4))
df.groupby('month')['ND'].mean().plot(marker='o')
plt.title('Average National Demand by Month (2020-2025)')
plt.xlabel('Month')
plt.ylabel('National Demand')
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/national_demand_vs_month.png', dpi=300)

# Plot 3: Average National Demand by Hour in Different Seasons
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

# Plot 4 & 5: Plots to show that combining MET and NESO data is valid.
# Plot 4: Embedded Solar vs Solar Radiation in East Norfolk
plt.figure(figsize=(10, 6))
correlation_solar = df['Solar_Eng_East_Norfolk'].corr(df['EMBEDDED_SOLAR_GENERATION'])

slope, intercept = np.polyfit(df['Solar_Eng_East_Norfolk'], df['EMBEDDED_SOLAR_GENERATION'], 1)
trend_line = slope * df['Solar_Eng_East_Norfolk'] + intercept

plt.scatter(df['Solar_Eng_East_Norfolk'], df['EMBEDDED_SOLAR_GENERATION'], alpha=0.5, s=10, color='orange')
plt.plot(df['Solar_Eng_East_Norfolk'], trend_line, color='black', label='Trend Line')
plt.xlabel('Solar Radiation - East Norfolk (W/m²)')
plt.ylabel('Embedded Solar Generation (MW)')
plt.title('Embedded Solar Generation vs East Norfolk Solar Radiation')
plt.text(0.05, 0.95, f'Correlation: {correlation_solar:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/solar_corr.png', dpi=300)

# Plot 5: Embedded Wind vs Wind Speed in East Norfolk
plt.figure(figsize=(10, 6))
correlation_wind = df['Wind10m_Eng_East_Norfolk'].corr(df['EMBEDDED_WIND_GENERATION'])

slope, intercept = np.polyfit(df['Wind10m_Eng_East_Norfolk'], df['EMBEDDED_WIND_GENERATION'], 1)
trend_line = slope * df['Wind10m_Eng_East_Norfolk'] + intercept

plt.scatter(df['Wind10m_Eng_East_Norfolk'], df['EMBEDDED_WIND_GENERATION'], alpha=0.5, s=10, color='green')
plt.plot(df['Wind10m_Eng_East_Norfolk'], trend_line, color='black', label='Trend Line')
plt.xlabel('Wind Speed - East Norfolk (m/s)')
plt.ylabel('Embedded Wind Generation (MW)')
plt.title('Embedded Wind Generation vs East Norfolk Wind Speed')
plt.text(0.05, 0.95, f'Correlation: {correlation_wind:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/wind_corr.png', dpi=300)

# Plot 6: National Demand vs. Temperature
plt.figure(figsize=(10, 6))
correlation_temp = df['Temp_Eng_East_Norfolk'].corr(df['ND'])

slope, intercept = np.polyfit(df['Temp_Eng_East_Norfolk'], df['ND'], 1)
trend_line = slope * df['Temp_Eng_East_Norfolk'] + intercept

plt.scatter(df['Temp_Eng_East_Norfolk'], df['ND'], alpha=0.5, s=10, color='darkred')
plt.plot(df['Temp_Eng_East_Norfolk'], trend_line, color='black', label='Trend Line')
plt.xlabel('Temperature - East Norfolk (°C)')
plt.ylabel('National Demand (MW)')
plt.title('National Demand vs East Norfolk Temperature')
plt.text(0.05, 0.95, f'Correlation: {correlation_temp:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figures/temp_corr.png', dpi=300)

# insert plot 7 here
import os
import matplotlib.pyplot as plt

# Create folder if it doesn't exist
os.makedirs("Figures", exist_ok=True)

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
