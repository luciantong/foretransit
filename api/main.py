from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.cache import get as cache_get, set as cache_set, stats as cache_stats
import json
import re
import pandas as pd
from datetime import datetime

app = FastAPI(title="ForéTransit API")
STOPS_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data/stops.txt"
STOP_TIMES_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data/stop_times.txt"
TRIPS_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data/trips.txt"
ROUTES_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data/routes.txt"
WEATHER_PATH = "data/raw/weather/weather_latest.json"
LIGHT_RAIL_KEYWORDS = {
    "streetcar",
    "lrt",
    "loop",
}
SUBWAY_STATION_KEYWORDS = {
    "vaughan metropolitan centre",
    "highway 407",
    "pioneer village",
    "york university",
    "finch west",
    "downsview park",
    "sheppard west",
    "wilson",
    "yorkdale",
    "lawrence west",
    "glencairn",
    "cedarvale",
    "st clair west",
    "dupont",
    "spadina",
    "st george",
    "museum",
    "queen's park",
    "st patrick",
    "osgoode",
    "st andrew",
    "union",
    "king",
    "queen",
    "dundas",
    "college",
    "wellesley",
    "bloor-yonge",
    "rosedale",
    "summerhill",
    "st clair",
    "davisville",
    "eglinton",
    "lawrence",
    "york mills",
    "sheppard-yonge",
    "north york centre",
    "finch",
    "kipling",
    "islington",
    "royal york",
    "old mill",
    "jane",
    "runnymede",
    "high park",
    "keele",
    "dundas west",
    "lansdowne",
    "dufferin",
    "ossington",
    "christie",
    "bathurst",
    "bay",
    "sherbourne",
    "castle frank",
    "broadview",
    "chester",
    "pape",
    "donlands",
    "greenwood",
    "coxwell",
    "woodbine",
    "main street",
    "victoria park",
    "warden",
    "kennedy",
    "bayview",
    "bessarion",
    "leslie",
    "don mills",
}
def classify_station_mode(station_name: str) -> str:
    name = str(station_name).lower()
    if re.search(r"\s-\s.*station\b", name) and "go station" not in name:
        return "railway"
    if any(keyword in name for keyword in LIGHT_RAIL_KEYWORDS):
        return "light_rail"
    if any(keyword in name for keyword in SUBWAY_STATION_KEYWORDS):
        return "railway"
    return "bus"


def is_ttc_train_station_name(stop_name: str) -> bool:
    name = str(stop_name).lower().strip()
    if "go station" in name:
        return False
    if re.search(r"\s-\s.*station\b", name):
        return True
    if re.search(r"\bstation\b", name) and any(keyword in name for keyword in SUBWAY_STATION_KEYWORDS):
        return True
    return False


def is_parent_ttc_station(stop_name: str) -> bool:
    name = str(stop_name).strip()
    return is_ttc_train_station_name(name) and (" - " not in name)


def build_stop_mode_map(stops_df: pd.DataFrame) -> dict:
    df = stops_df.copy()
    df["stop_id"] = df["stop_id"].astype(str)
    df["stop_name"] = df["stop_name"].astype(str)
    df["location_type"] = pd.to_numeric(df.get("location_type", 0), errors="coerce").fillna(0).astype(int)
    df["parent_station"] = df.get("parent_station", "").fillna("").astype(str)

    station_rows = df[df["location_type"] == 1]
    station_mode_by_id = {
        row.stop_id: classify_station_mode(row.stop_name)
        for row in station_rows.itertuples(index=False)
    }

    mode_by_stop_id = {}
    for row in df.itertuples(index=False):
        if row.location_type == 1:
            mode_by_stop_id[row.stop_id] = station_mode_by_id.get(row.stop_id, "bus")
            continue

        parent_id = str(row.parent_station).strip()
        if parent_id and parent_id in station_mode_by_id:
            mode_by_stop_id[row.stop_id] = station_mode_by_id[parent_id]
            continue

        name = str(row.stop_name).lower()
        if is_ttc_train_station_name(name):
            mode_by_stop_id[row.stop_id] = "railway"
            continue
        if re.search(r"\s-\s.*station\b", name) and "go station" not in name:
            mode_by_stop_id[row.stop_id] = "railway"
            continue
        if any(keyword in name for keyword in LIGHT_RAIL_KEYWORDS):
            mode_by_stop_id[row.stop_id] = "light_rail"
        else:
            mode_by_stop_id[row.stop_id] = "bus"

    return mode_by_stop_id


def build_parent_station_map(stops_df: pd.DataFrame) -> dict:
    df = stops_df.copy()
    df["stop_id"] = df["stop_id"].astype(str)
    df["stop_name"] = df["stop_name"].astype(str)
    df["location_type"] = pd.to_numeric(df.get("location_type", 0), errors="coerce").fillna(0).astype(int)
    df["parent_station"] = df.get("parent_station", "").fillna("").astype(str)

    parent_by_stop_id = {}
    for row in df.itertuples(index=False):
        name = str(row.stop_name)
        if not is_ttc_train_station_name(name):
            parent_by_stop_id[row.stop_id] = False
            continue

        has_parent_ref = bool(str(row.parent_station).strip())
        if has_parent_ref:
            parent_by_stop_id[row.stop_id] = False
            continue

        if " - " in name:
            parent_by_stop_id[row.stop_id] = False
            continue

        if row.location_type == 1:
            parent_by_stop_id[row.stop_id] = True
            continue

        # Some TTC station rows in raw GTFS are not flagged as location_type=1.
        parent_by_stop_id[row.stop_id] = is_parent_ttc_station(name)

    return parent_by_stop_id

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
    try:
        from api.station_forecast import get_station_forecast
    except Exception as exc:
        return {
            "error": "Station forecast unavailable. Check GTFS static files and Python dependencies.",
            "details": str(exc),
        }
    cached = cache_get(stop_id)
    if cached:
        return {**cached, "cached": True}
    try:
        result = get_station_forecast(stop_id)
    except Exception as exc:
        return {
            "error": "Station forecast failed while processing this stop.",
            "details": str(exc),
        }
    cache_set(stop_id, result)
    return {**result, "cached": False}

# ─── Live Vehicles ────────────────────────────
# Called to animate vehicles on the map
@app.get("/vehicles/live")
def live_vehicles():
    with open("data/raw/gtfs_realtime/vehicles_latest.json") as f:
        data = json.load(f)
    return data["data"]["vehicle"]


@app.get("/weather/current")
def current_weather():
    try:
        with open(WEATHER_PATH) as f:
            weather = json.load(f)

        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%dT%H:00")
        times = weather.get("hourly", {}).get("time", [])
        hourly = weather.get("hourly", {})

        if not times:
            return {
                "error": "Weather data unavailable",
                "details": "No hourly timestamps in weather file",
            }

        def at(values, idx):
            arr = values if isinstance(values, list) else []
            return arr[idx] if 0 <= idx < len(arr) else None

        warning = None

        if current_hour in times:
            idx = times.index(current_hour)
            observed_hour = current_hour
        else:
            earlier = [t for t in times if t <= current_hour]
            observed_hour = earlier[-1] if earlier else times[-1]
            idx = times.index(observed_hour)
            warning = "Using nearest available hour from weather file"

        response = {
            "temperature_c": at(hourly.get("temperature_2m"), idx),
            "rain_mm": at(hourly.get("rain"), idx),
            "snow_mm": at(hourly.get("snowfall"), idx),
            "wind_kmh": at(hourly.get("windspeed_10m"), idx),
            "visibility_m": at(hourly.get("visibility"), idx),
            "observed_hour": observed_hour,
            "source": "weather_latest.json",
        }
        if warning:
            response["warning"] = warning
        return response
    except Exception as exc:
        return {
            "error": "Weather data unavailable",
            "details": str(exc),
        }


@app.get("/routes/geojson")
def routes_geojson():
    """
    Returns TTC route shapes as GeoJSON FeatureCollection.
    Each feature is a LineString representing one route path.
    """
    try:
        shapes_path = "data/raw/gtfs_static/TTC Routes and Schedules Data/shapes.txt"
        shapes_df = pd.read_csv(shapes_path)
        
        # Group by shape_id and build LineStrings
        features = []
        for shape_id, group in shapes_df.groupby("shape_id"):
            coords = group.sort_values("shape_pt_sequence")[["shape_pt_lon", "shape_pt_lat"]].values.tolist()
            if len(coords) < 2:
                continue
            
            feature = {
                "type": "Feature",
                "properties": {"shape_id": str(shape_id)},
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    except Exception as exc:
        return {
            "type": "FeatureCollection",
            "features": [],
            "error": str(exc)
        }


@app.get("/stops/search")
def search_stops(q: str = "", limit: int = 30):
    try:
        stops_df = pd.read_csv(
            STOPS_PATH,
            usecols=["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
        )
    except Exception as exc:
        return {"error": "Stops data unavailable", "details": str(exc), "stops": []}

    mode_by_stop_id = build_stop_mode_map(stops_df)
    parent_by_stop_id = build_parent_station_map(stops_df)

    results = stops_df
    query = q.strip()
    if query:
        results = results[results["stop_name"].str.contains(query, case=False, na=False)]

    capped = results.head(max(1, min(limit, 100)))
    return {
        "stops": [
            {
                "stop_id": str(row.stop_id),
                "stop_name": row.stop_name,
                "mode": mode_by_stop_id.get(str(row.stop_id), "bus"),
                "is_parent_station": parent_by_stop_id.get(str(row.stop_id), False),
                "lat": float(row.stop_lat),
                "lon": float(row.stop_lon),
            }
            for row in capped.itertuples(index=False)
        ]
    }


@app.get("/stops")
def get_stops(q: str = "", limit: int = 1500):
    try:
        stops_df = pd.read_csv(
            STOPS_PATH,
            usecols=["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
        )
    except Exception as exc:
        return {"error": "Stops data unavailable", "details": str(exc), "stops": []}

    mode_by_stop_id = build_stop_mode_map(stops_df)
    parent_by_stop_id = build_parent_station_map(stops_df)

    results = stops_df
    query = q.strip()
    if query:
        results = results[results["stop_name"].str.contains(query, case=False, na=False)]

    capped = results.head(max(1, min(limit, 5000)))
    return {
        "stops": [
            {
                "stop_id": str(row.stop_id),
                "stop_name": row.stop_name,
                "lat": float(row.stop_lat),
                "lon": float(row.stop_lon),
                "mode": mode_by_stop_id.get(str(row.stop_id), "bus"),
                "is_parent_station": parent_by_stop_id.get(str(row.stop_id), False),
            }
            for row in capped.itertuples(index=False)
        ]
    }


@app.get("/stops/subway")
def get_subway_stops(q: str = "", limit: int = 1200):
    try:
        stops_df = pd.read_csv(
            STOPS_PATH,
            usecols=["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
        ).dropna()
    except Exception as exc:
        return {"error": "Stops data unavailable", "details": str(exc), "stops": []}

    mode_by_stop_id = build_stop_mode_map(stops_df)
    parent_by_stop_id = build_parent_station_map(stops_df)

    try:
        routes_df = pd.read_csv(ROUTES_PATH, usecols=["route_id", "route_type"]).dropna()
        trips_df = pd.read_csv(TRIPS_PATH, usecols=["trip_id", "route_id"]).dropna()
        stop_times_df = pd.read_csv(STOP_TIMES_PATH, usecols=["trip_id", "stop_id"]).dropna()

        route_types = pd.to_numeric(routes_df["route_type"], errors="coerce")
        subway_route_ids = set(routes_df.loc[route_types == 1, "route_id"].astype(str))
        subway_trip_ids = set(trips_df[trips_df["route_id"].astype(str).isin(subway_route_ids)]["trip_id"].astype(str))
        subway_stop_ids = set(
            stop_times_df[stop_times_df["trip_id"].astype(str).isin(subway_trip_ids)]["stop_id"].astype(str)
        )

        results = stops_df[stops_df["stop_id"].astype(str).isin(subway_stop_ids)]
        if results.empty:
            raise ValueError("No subway stops matched GTFS relations")
    except Exception:
        # Fallback when stop_times/trips are missing: TTC subway station-name heuristic.
        name_lower = stops_df["stop_name"].astype(str).str.lower()
        has_station = name_lower.str.contains("station", na=False)
        is_ttc_subway_station = name_lower.apply(
            lambda name: any(keyword in name for keyword in SUBWAY_STATION_KEYWORDS)
        )
        not_go = ~name_lower.str.contains("go station", na=False)
        results = stops_df[has_station & is_ttc_subway_station & not_go]

    query = q.strip()
    if query:
        results = results[results["stop_name"].str.contains(query, case=False, na=False)]

    results = results.drop_duplicates(subset=["stop_id"])
    capped = results.head(max(1, min(limit, 5000)))

    return {
        "stops": [
            {
                "stop_id": str(row.stop_id),
                "stop_name": row.stop_name,
                "lat": float(row.stop_lat),
                "lon": float(row.stop_lon),
                "mode": mode_by_stop_id.get(str(row.stop_id), "bus"),
                "is_parent_station": parent_by_stop_id.get(str(row.stop_id), False),
            }
            for row in capped.itertuples(index=False)
        ]
    }

# ─── Cache Stats (debug) ──────────────────────
@app.get("/debug/cache")
def debug_cache():
    return cache_stats()