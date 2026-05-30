from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from api.station_forecast import get_station_forecast, get_current_weather
import pandas as pd
import requests

app = FastAPI(title="ForéTransit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

GTFS_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data"
stops_df  = pd.read_csv(f"{GTFS_PATH}/stops.txt")

@app.get("/vehicles/live")
def live_vehicles():
    try:
        response = requests.get(
            "https://retro.umoiq.com/service/publicJSONFeed?command=vehicleLocations&a=ttc",
            timeout=10
        )
        return response.json()["vehicle"]
    except Exception as e:
        return {"error": str(e)}

# ─── Health Check ─────────────────────────────
@app.get("/")
def root():
    return {"status": "ForéTransit API is running"}

# ─── All Stops ────────────────────────────────
@app.get("/stops")
def get_stops(limit: int = 5000):
    df = stops_df.head(limit)
    stops = df.rename(columns={
        "stop_id":   "stop_id",
        "stop_name": "stop_name",
        "stop_lat":  "lat",
        "stop_lon":  "lon"
    })[["stop_id", "stop_name", "lat", "lon"]].copy()
    stops["mode"] = "bus"  # default; refine if you have route_type per stop
    return {"stops": stops.to_dict(orient="records")}

# ─── Stop Search ──────────────────────────────
@app.get("/stops/search")
def search_stops(q: str = "", limit: int = 8):
    if not q.strip():
        return {"stops": []}
    mask    = stops_df["stop_name"].str.contains(q, case=False, na=False)
    results = stops_df[mask].head(limit)
    stops   = results.rename(columns={
        "stop_lat": "lat",
        "stop_lon": "lon"
    })[["stop_id", "stop_name", "lat", "lon"]].copy()
    stops["mode"] = "bus"
    return {"stops": stops.to_dict(orient="records")}

# ─── Station Forecast ─────────────────────────
@app.get("/station/{stop_id}")
def station_forecast(stop_id: str):
    return get_station_forecast(stop_id)

# ─── Weather ──────────────────────────────────
@app.get("/weather/current")
def weather_current():
    w = get_current_weather()
    return {
        "temperature_c": w["temperature"],
        "rain_mm":        w["rain"],
        "wind_kmh":       w["wind"],
        "snow_mm":        w["snow"],
        "visibility_km":  w["visibility"] / 1000
    }

# ─── Bike Share ───────────────────────────────
@app.get("/bikeshare/stations")
def bikeshare_stations(limit: int = 5000):
    try:
        GBFS_INFO   = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information"
        GBFS_STATUS = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_status"
        info   = requests.get(GBFS_INFO,   timeout=5).json()
        status = requests.get(GBFS_STATUS, timeout=5).json()
        coords = {
            s["station_id"]: (s["lat"], s["lon"], s["name"])
            for s in info["data"]["stations"]
        }
        stations = []
        for s in status["data"]["stations"][:limit]:
            sid = s["station_id"]
            if sid not in coords:
                continue
            lat, lon, name = coords[sid]
            stations.append({
                "station_id":       sid,
                "name":             name,
                "lat":              lat,
                "lon":              lon,
                "bikes_available":  s["num_bikes_available"],
                "ebikes_available": s.get("num_ebikes_available", 0),
                "docks_available":  s["num_docks_available"],
                "is_renting":       bool(s["is_renting"])
            })
        return {"stations": stations}
    except Exception as e:
        return {"stations": [], "error": str(e)}