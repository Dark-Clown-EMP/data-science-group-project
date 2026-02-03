import requests
import pandas as pd
import json
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
# 1. SETUP YOUR CREDENTIALS
# This is the token you just provided
API_TOKEN = os.getenv("MET_OFFICE_API_KEY")

# 2. DEFINE THE REQUEST
# URL for Hourly Spot Data (Live Forecast)
url = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point/hourly"

start_date = "2020-01-01"
end_date = "2025-12-31"

# Comprehensive UK Regional Coordinates for Weather API
# Includes England (spread), Scotland, Wales, and Northern Ireland

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

# Coordinates for Grimsby (Near Offshore Wind Farms)


headers = {
    "apikey": API_TOKEN,
    "accept": "application/json"
}


# 4. CHECK RESULTS
for key, location in uk_regions.items():
    params = {
    "latitude": location[0],
    "longitude": location[1],
    "start_date": start_date,  # 5 Years of History
	"end_date": end_date,
    "excludeParameterMetadata": "true",
    "includeLocationName": "true"
    }
    # 3. FETCH THE DATA
    print("📡 Connecting to Met Office Live API...")
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        print("✅ Success! Data Received.")
        data = response.json()
    
        # Extract the time series data
        features = data['features'][0]['properties']['timeSeries']
    
        # Convert to DataFrame for viewing
        df = pd.DataFrame(features)
        print("\n--- LIVE FORECAST DATA (Next 24-48 Hours) ---")
        print(df[['time', 'screenTemperature', 'windSpeed10m']].head())
        file_loc = "live_weather_forecast" + "_lat_" + str(location[0]) + "_lon_" + str(location[1]) + key + ".csv"
        # Save for your project
        df.to_csv("./data/" + file_loc, index=False)
        print("\n💾 Saved to " + file_loc)
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        print("\nTIP: If you get a 401 Error, go back to 'My Applications' on the Met Office site")
        print("and copy the short 'Client ID' (it is shorter than the token you pasted).")