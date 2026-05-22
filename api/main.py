from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.station_forecast import get_station_forecast
from api.cache import get as cache_get, set as cache_set, stats as cache_stats
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
# Results cached for 60s so repeated clicks don't re-run MAGI
@app.get("/station/{stop_id}")
def station_forecast(stop_id: str):
    cached = cache_get(stop_id)
    if cached:
        return {**cached, "cached": True}
    result = get_station_forecast(stop_id)
    cache_set(stop_id, result)
    return {**result, "cached": False}

# ─── Live Vehicles ────────────────────────────
# Called to animate vehicles on the map
@app.get("/vehicles/live")
def live_vehicles():
    with open("data/raw/gtfs_realtime/vehicles_latest.json") as f:
        data = json.load(f)
    return data["vehicles"]["vehicle"]

# ─── Cache Stats (debug) ──────────────────────
@app.get("/debug/cache")
def debug_cache():
    return cache_stats()