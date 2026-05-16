from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.station_forecast import get_station_forecast
import json

app = FastAPI(title="ForéTransit API")

# Allows frontend to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── Health Check ─────────────────────────────
@app.get("/")
def root():
    return {"status": "ForéTransit API is running"}

# ─── Station Forecast ─────────────────────────
# Called when user clicks a station on the map
@app.get("/station/{stop_id}")
def station_forecast(stop_id: str):
    return get_station_forecast(stop_id)

# ─── Live Vehicles ────────────────────────────
# Called to animate vehicles on the map
@app.get("/vehicles/live")
def live_vehicles():
    with open("data/raw/gtfs_realtime/vehicles_latest.json") as f:
        data = json.load(f)
    return data["vehicles"]["vehicle"]