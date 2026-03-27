import pandas as pd
import os

# Assume these imports work
from api_call import get_region_data
from demand_data_aggregate import get_year_list

# 1. SETUP
region_dict = get_region_data()
year_list = get_year_list()

interconnector_cols = [
    'IFA_FLOW', 'IFA2_FLOW', 'BRITNED_FLOW', 'MOYLE_FLOW',
    'EAST_WEST_FLOW', 'NEMO_FLOW', 'NSL_FLOW', 'ELECLINK_FLOW',
    'VIKING_FLOW', 'GREENLINK_FLOW'
]

wind_regions = [
    "Scot_Highlands", "Scot_Aberdeenshire", "Scot_Borders",
    "Wales_North_Gwynedd", "Eng_North_Cumbria", "Eng_Yorkshire",
    "Eng_East_Norfolk", "Eng_East_Suffolk"
]

solar_regions = [
    "Eng_South_Cornwall", "Eng_South_Hampshire", "Eng_South_Kent",
    "Eng_South_Bristol", "Eng_East_Suffolk", "Eng_East_Norfolk",
    "Eng_East_Midlands", "Eng_London"
]

# --- PART 1: LOAD GRID DATA ---
print("🚀 Loading Demand Data...")
power_demand_path = './data/uk_grid_hourly_cleaned_'
demand_dfs = []

for year in year_list:
    file_path = f"{power_demand_path}{year}.csv"
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping {year} (File not found)")
        continue
    
    df = pd.read_csv(file_path)
    
    # Merge Interconnectors
    available_cables = [col for col in interconnector_cols if col in df.columns]
    if available_cables:
        df['NET_IMPORTS'] = df[available_cables].sum(axis=1)
        df = df.drop(columns=available_cables)
    else:
        df['NET_IMPORTS'] = 0
        
    demand_dfs.append(df)

final_df = pd.concat(demand_dfs, ignore_index=True)

# FORCE DATETIME FORMAT (The "Nuclear" Fix)
final_df['datetime'] = pd.to_datetime(final_df['datetime'])
final_df = final_df.set_index('datetime').sort_index()
final_df.index = final_df.index.tz_localize(None) # Strip Timezones

print(f"✅ Grid Data: {len(final_df)} rows")

# --- PART 2: MERGE WEATHER ---
print("🔄 Merging Weather Data...")
weather_path = './data/weather_'

for key, loc in region_dict.items():
    file_path = f"{weather_path}{key}.csv"
    if not os.path.exists(file_path):
        continue

    # Load & Fix Index
    weather_df = pd.read_csv(file_path)
    time_col = 'time' if 'time' in weather_df.columns else 'datetime'
    weather_df['datetime'] = pd.to_datetime(weather_df[time_col])
    weather_df = weather_df.set_index('datetime')
    weather_df.index = weather_df.index.tz_localize(None) # Strip Timezones

    cols_to_keep = []
    rename_map = {}

    # 1. Temperature
    if 'temperature_2m' in weather_df.columns:
        cols_to_keep.append('temperature_2m')
        rename_map['temperature_2m'] = f'Temp_{key}'

    # 2. Wind (Prioritize 10m if 80m is missing)
    if key in wind_regions:
        for col in ['wind_speed_10m', 'wind_speed_80m']:
            if col in weather_df.columns:
                # ONLY add if data is not all null
                if weather_df[col].notna().sum() > 0:
                    cols_to_keep.append(col)
                    suffix = "10m" if "10m" in col else "80m"
                    rename_map[col] = f'Wind{suffix}_{key}'

    # 3. Solar
    if key in solar_regions and 'shortwave_radiation' in weather_df.columns:
        cols_to_keep.append('shortwave_radiation')
        rename_map['shortwave_radiation'] = f'Solar_{key}'

    if cols_to_keep:
        subset = weather_df[cols_to_keep].rename(columns=rename_map)
        final_df = final_df.join(subset, how='left')
        print(f"  + Merged {key}")

# --- PART 3: SMART CLEANING (THE FIX) ---
print("\n🧹 Cleaning Data...")
initial_shape = final_df.shape

# 1. Drop columns that are completely empty (Fixes the Wind80m issue)
final_df = final_df.dropna(axis=1, how='all')
print(f"   - Dropped {initial_shape[1] - final_df.shape[1]} empty columns")

# 2. Fill Grid Gaps (Fixes SCOTTISH_TRANSFER issue)
# If a flow value is missing, assume it's 0 or take the previous value
final_df = final_df.fillna(value={'SCOTTISH_TRANSFER': 0, 'NET_IMPORTS': 0})

# 3. Forward Fill tiny weather gaps (1-2 hours)
final_df = final_df.ffill(limit=2)

# 4. NOW drop rows (Only drops truly broken dates)
final_df = final_df.dropna()

print(f"✅ Final Shape: {final_df.shape}")
final_df.to_csv('./data/final_model_data.csv')
print("Saved to ./data/final_model_data.csv")