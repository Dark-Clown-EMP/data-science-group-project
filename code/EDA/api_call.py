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

longitiude = -0.08
latitude = 53.57
start_date = "2020-01-01"
end_date = "2025-12-31"

# Coordinates for Grimsby (Near Offshore Wind Farms)
params = {
    "latitude": latitude,
    "longitude": longitiude,
    "start_date": start_date,  # 5 Years of History
	"end_date": end_date,
    "excludeParameterMetadata": "true",
    "includeLocationName": "true"
}

headers = {
    "apikey": API_TOKEN,
    "accept": "application/json"
}

# 3. FETCH THE DATA
print("📡 Connecting to Met Office Live API...")
response = requests.get(url, headers=headers, params=params)

# 4. CHECK RESULTS
if response.status_code == 200:
    print("✅ Success! Data Received.")
    data = response.json()
    
    # Extract the time series data
    features = data['features'][0]['properties']['timeSeries']
    
    # Convert to DataFrame for viewing
    df = pd.DataFrame(features)
    print("\n--- LIVE FORECAST DATA (Next 24-48 Hours) ---")
    print(df[['time', 'screenTemperature', 'windSpeed10m']].head())
    file_loc = "live_weather_forecast" + "_lon_" + str(longitiude) + "_lat_" + str(latitude) + ".csv"
    # Save for your project
    df.to_csv("./data/" + file_loc, index=False)
    print("\n💾 Saved to " + file_loc)
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
    print("\nTIP: If you get a 401 Error, go back to 'My Applications' on the Met Office site")
    print("and copy the short 'Client ID' (it is shorter than the token you pasted).")