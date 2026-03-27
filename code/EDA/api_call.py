import requests
import pandas as pd
import json
from dotenv import load_dotenv
import os
import time
from pathlib import Path


# Load environment variables from .env file
load_dotenv()
# 1. SETUP YOUR CREDENTIALS
# This is the token you just provided
API_TOKEN = os.getenv("MET_OFFICE_API_KEY")

# 2. DEFINE THE REQUEST
# URL for Hourly Spot Data (Live Forecast)
url = "https://archive-api.open-meteo.com/v1/archive"

start_date = "2020-01-01"
end_date = "2025-12-31"

# Comprehensive UK Regional Coordinates for Weather API
# Includes England (spread), Scotland, Wales

uk_regions = {

    # --- SCOTLAND (Crucial for Wind) ---
    "Scot_Highlands":      [ 57.4778, -4.2247], # Inverness area
    "Scot_Aberdeenshire":  [ 57.1497, -2.0943], # Offshore wind hub
    "Scot_Glasgow_West":   [ 55.8642, -4.2518],
    "Scot_Edinburgh_East": [ 55.9533, -3.1883],
    "Scot_Borders":        [ 55.5486, -2.7861],

    # --- WALES ---
    "Wales_North_Gwynedd": [ 52.9370, -3.8960], # Mountainous/Windy
    "Wales_South_Cardiff": [ 51.4816, -3.1791],

    # --- ENGLAND NORTH ---
    "Eng_North_Tyne":      [ 55.0077, -1.6578], # Newcastle
    "Eng_North_Cumbria":   [ 54.8925, -2.9329], # West coast wind
    "Eng_Yorkshire":       [ 53.9590, -1.0873], # Drax area
    "Eng_Manchester":      [ 53.4808, -2.2426],

    # --- ENGLAND MIDLANDS ---
    "Eng_West_Midlands":   [ 52.4862, -1.8904], # Birmingham
    "Eng_East_Midlands":   [ 52.9548, -1.1581], # Nottingham

    # --- ENGLAND EAST (Wind & Solar) ---
    "Eng_East_Norfolk":    [ 52.6309, 1.2974],  # Bacton gas/wind
    "Eng_East_Suffolk":    [ 52.1732, 1.3513],  # Sizewell Nuclear/Wind

    # --- ENGLAND SOUTH ---
    "Eng_London":          [ 51.5074, -0.1278], # Main Load
    "Eng_South_Kent":      [ 51.2787, 0.5217],  # Interconnectors
    "Eng_South_Hampshire": [ 51.0577, -1.3187], # Southampton
    "Eng_South_Cornwall":  [ 50.2660, -5.0527], # Solar Hub
    "Eng_South_Bristol":   [ 51.4545, -2.5879]
}



# headers = {
#     "apikey": API_TOKEN,
#     "accept": "application/json"
# }

def get_region_data():
    return uk_regions

# 4. CHECK RESULTS
for key, location in uk_regions.items():
    print(f"📡 Fetching data for {key}...")

    params = {
        "latitude": location[0],
        "longitude": location[1],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m,wind_speed_80m,shortwave_radiation,direct_normal_irradiance,cloud_cover",
        "timezone": "GMT" # Crucial for alignment
    }
    file_name = f"weather_{key}.csv"
    if Path("./data/" + file_name).exists():
        print('✅ Data already exists')
        continue

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        
        # Open-Meteo returns data directly in the 'hourly' key
        hourly_data = data['hourly']
        
        # Create DataFrame
        df = pd.DataFrame(hourly_data)
        
        # Save
        df.to_csv(f"./data/{file_name}", index=False)
        print(f"✅ Saved {len(df)} rows to ./data/{file_name}")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    

    # Polite sleep to respect the free API
    time.sleep(1.5)

print("\n🚀 All downloads complete.")

