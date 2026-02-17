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

# Plot 2: Average National Demand by Month
plt.figure(figsize=(8,4))
df.groupby('month')['ND'].mean().plot(marker='o')
plt.title('Average National Demand by Month (2020-2025)')
plt.xlabel('Month')
plt.ylabel('National Demand')
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()

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

# Plot 4 & 5: Plots to show that combining MET and NESO data is valid.
# Plot 4: Embedded Solar vs Solar Radiation in East Norfolk
plt.figure(figsize=(10, 6))
correlation_solar = df['Solar_Eng_East_Norfolk'].corr(df['EMBEDDED_SOLAR_GENERATION'])

plt.scatter(df['Solar_Eng_East_Norfolk'], df['EMBEDDED_SOLAR_GENERATION'], alpha=0.5, s=10, color='orange')
plt.xlabel('Solar Radiation - East Norfolk (W/m²)')
plt.ylabel('Embedded Solar Generation (MW)')
plt.title('Embedded Solar Generation vs East Norfolk Solar Radiation')
plt.text(0.05, 0.95, f'Correlation: {correlation_solar:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()

# Plot 5: Embedded Wind vs Wind Speed in East Norfolk
plt.figure(figsize=(10, 6))
correlation_wind = df['Wind10m_Eng_East_Norfolk'].corr(df['EMBEDDED_WIND_GENERATION'])

plt.scatter(df['Wind10m_Eng_East_Norfolk'], df['EMBEDDED_WIND_GENERATION'], alpha=0.5, s=10, color='green')
plt.xlabel('Wind Speed - East Norfolk (m/s)')
plt.ylabel('Embedded Wind Generation (MW)')
plt.title('Embedded Wind Generation vs East Norfolk Wind Speed')
plt.text(0.05, 0.95, f'Correlation: {correlation_wind:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
temp_by_hour = df.groupby('hour')[temp_cols].mean()
temp_by_month = df.groupby('month')[temp_cols].mean()
temp_by_year = df.groupby('year')[temp_cols].mean()
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()

# Plot 6a: Regional variations by Hour (solar, wind, temp)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# solar by hour
solar_by_hour = df.groupby('hour')[solar_cols].mean()
for col in solar_cols:
    axes[0].plot(solar_by_hour.index, solar_by_hour[col], label=col, alpha=0.7)
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Solar Radiation (W/m²)')
axes[0].set_title('Solar Radiation by Hour')
axes[0].legend(fontsize=7, loc='best')
axes[0].grid(ls='--', alpha=0.6)

# wind by hour
wind_by_hour = df.groupby('hour')[wind_cols].mean()
for col in wind_cols:
    axes[1].plot(wind_by_hour.index, wind_by_hour[col], label=col, alpha=0.7)
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Wind Speed (m/s)')
axes[1].set_title('Wind Speed by Hour')
axes[1].legend(fontsize=7, loc='best')
axes[1].grid(ls='--', alpha=0.6)

# temp by hour
temp_by_hour = df.groupby('hour')[temp_cols].mean()
for col in temp_cols:
    axes[2].plot(temp_by_hour.index, temp_by_hour[col], label=col, alpha=0.7)
axes[2].set_xlabel('Hour of Day')
axes[2].set_ylabel('Temperature (°C)')
axes[2].set_title('Temperature by Hour')
axes[2].legend(fontsize=7, loc='best')
axes[2].grid(ls='--', alpha=0.6)

plt.tight_layout()

# Plot 6b: Regional variations by Month (solar, wind, temp)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

solar_by_month = df.groupby('month')[solar_cols].mean()
for col in solar_cols:
    axes[0].plot(solar_by_month.index, solar_by_month[col], label=col, alpha=0.7)
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Solar Radiation (W/m²)')
axes[0].set_title('Solar Radiation by Month')
axes[0].legend(fontsize=7, loc='best')
axes[0].grid(ls='--', alpha=0.6)

wind_by_month = df.groupby('month')[wind_cols].mean()
for col in wind_cols:
    axes[1].plot(wind_by_month.index, wind_by_month[col], label=col, alpha=0.7)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Wind Speed (m/s)')
axes[1].set_title('Wind Speed by Month')
axes[1].legend(fontsize=7, loc='best')
axes[1].grid(ls='--', alpha=0.6)

temp_by_month = df.groupby('month')[temp_cols].mean()
for col in temp_cols:
    axes[2].plot(temp_by_month.index, temp_by_month[col], label=col, alpha=0.7)
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Temperature (°C)')
axes[2].set_title('Temperature by Month')
axes[2].legend(fontsize=7, loc='best')
axes[2].grid(ls='--', alpha=0.6)

plt.tight_layout()

# Plot 6c: Regional variations by Year (solar, wind, temp)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

solar_by_year = df.groupby('year')[solar_cols].mean()
for col in solar_cols:
    axes[0].plot(solar_by_year.index, solar_by_year[col], label=col, alpha=0.7, marker='o')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Solar Radiation (W/m²)')
axes[0].set_title('Solar Radiation by Year')
axes[0].legend(fontsize=7, loc='best')
axes[0].grid(ls='--', alpha=0.6)

wind_by_year = df.groupby('year')[wind_cols].mean()
for col in wind_cols:
    axes[1].plot(wind_by_year.index, wind_by_year[col], label=col, alpha=0.7, marker='o')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Wind Speed (m/s)')
axes[1].set_title('Wind Speed by Year')
axes[1].legend(fontsize=7, loc='best')
axes[1].grid(ls='--', alpha=0.6)

temp_by_year = df.groupby('year')[temp_cols].mean()
for col in temp_cols:
    axes[2].plot(temp_by_year.index, temp_by_year[col], label=col, alpha=0.7, marker='o')
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Temperature (°C)')
axes[2].set_title('Temperature by Year')
axes[2].legend(fontsize=7, loc='best')
axes[2].grid(ls='--', alpha=0.6)

plt.tight_layout()


# Show plots
plt.show()