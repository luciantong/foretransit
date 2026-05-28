import requests
import zipfile
import io
import os
import json
from datetime import datetime

GTFS_RT_VEHICLES = "https://retro.umoiq.com/service/publicJSONFeed?command=vehicleLocations&a=ttc"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast?latitude=43.6532&longitude=-79.3832&hourly=rain,snowfall,windspeed_10m,visibility,temperature_2m&forecast_days=2&timezone=America%2FToronto"

RT_DIR = "data/raw/gtfs_realtime"
WEATHER_DIR = "data/raw/weather"

def fetch_gtfs_realtime():
    print("Fetching GTFS Realtime data...")
    response = requests.get(GTFS_RT_VEHICLES)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "data": response.json()
    }

    path = f"{RT_DIR}/vehicles_latest.json"
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    
def fetch_weather_data():
    print("Fetching weather data...")
    response = requests.get(WEATHER_URL)   

    path = f"{WEATHER_DIR}/weather_latest.json"
    with open(path, "w") as f:
        json.dump(response.json(), f, indent=2)

if __name__ == "__main__":
    fetch_gtfs_realtime()
    fetch_weather_data()


    


