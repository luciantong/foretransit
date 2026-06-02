from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import APIRouter
from api.station_forecast import get_station_forecast, get_current_weather
import pandas as pd
import requests
import os

app = FastAPI(title="ForéTransit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

api = APIRouter(prefix="/api")

GTFS_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data"
try:
    stops_df = pd.read_csv(f"{GTFS_PATH}/stops.txt")
except Exception as e:
    import logging
    logging.exception("Failed to load GTFS stops file; continuing with empty stops dataframe")
    stops_df = pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])

try:
    stop_times_df = pd.read_csv(
        f"{GTFS_PATH}/stop_times.txt",
        usecols=["trip_id", "stop_id", "arrival_time"],
    )
except Exception:
    stop_times_df = pd.DataFrame(columns=["trip_id", "stop_id", "arrival_time"])

METRO_STATIONS_CSV_PATH = "data/processed/toronto_subway_parent_stations_matched.csv"
STATION_BUS_ETA_CSV_PATH = "data/processed/station_bus_eta_confidence.csv"
BUS_STATION_SUMMARY_CSV_PATH = "data/processed/bus_station_routes_summary.csv"


def load_metro_station_catalog() -> pd.DataFrame:
    if not os.path.exists(METRO_STATIONS_CSV_PATH):
        return pd.DataFrame(columns=["stop_id", "station_name", "lat", "lon", "line_numbers", "station_type"])

    df = pd.read_csv(METRO_STATIONS_CSV_PATH, dtype=str).fillna("")
    required = {"stop_id", "station_name", "lat", "lon"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["stop_id", "station_name", "lat", "lon", "line_numbers", "station_type"])

    df["stop_id"] = df["stop_id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df.drop_duplicates(subset=["stop_id"]).copy()


metro_catalog_df = load_metro_station_catalog()
METRO_STOP_IDS = set(metro_catalog_df["stop_id"].astype(str).tolist())
METRO_LINE_BY_STOP_ID = {
    str(row["stop_id"]): str(row.get("line_numbers", ""))
    for _, row in metro_catalog_df.iterrows()
}


def load_station_bus_eta_catalog() -> pd.DataFrame:
    if not os.path.exists(STATION_BUS_ETA_CSV_PATH):
        return pd.DataFrame(columns=[
            "station_name",
            "stop_name",
            "stop_id",
            "bus_route_number",
            "bus_route_name",
            "direction",
            "estimated_arrival",
            "estimated_arrival_in_min",
            "prediction_score_100",
            "prediction_confidence",
            "forecast_generated_at",
        ])

    df = pd.read_csv(STATION_BUS_ETA_CSV_PATH, dtype=str).fillna("")
    if "stop_id" not in df.columns:
        return pd.DataFrame(columns=["stop_id"])

    df["stop_id"] = df["stop_id"].astype(str)
    return df


station_bus_eta_df = load_station_bus_eta_catalog()


def load_bus_station_summary_catalog() -> pd.DataFrame:
    if not os.path.exists(BUS_STATION_SUMMARY_CSV_PATH):
        return pd.DataFrame(columns=[
            "stop_id",
            "stop_name",
            "bus_route_numbers",
            "bus_route_names",
            "directions",
            "route_direction_pairs",
            "eastbound_southbound_routes",
            "westbound_northbound_routes",
            "route_count",
        ])

    df = pd.read_csv(BUS_STATION_SUMMARY_CSV_PATH, dtype=str).fillna("")
    if "stop_id" not in df.columns:
        return pd.DataFrame(columns=["stop_id"])

    df["stop_id"] = df["stop_id"].astype(str)
    return df


bus_station_summary_df = load_bus_station_summary_catalog()


def infer_stop_mode(stop_name: str) -> str:
    name = str(stop_name or "").strip().lower()
    # GTFS stop metadata in this project does not include reliable parent_station/location_type,
    # so use station naming patterns to surface rail stops on the map.
    rail_tokens = [" station", "station ", " station -", " subway", "rt station"]
    return "railway" if any(token in name for token in rail_tokens) else "bus"


def _split_summary_field(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]

@api.get("/vehicles/live")
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
@api.get("/health")
def root():
    return {"status": "ForéTransit API is running"}

# ─── All Stops ────────────────────────────────
@api.get("/stops")
def get_stops(limit: int = 12000):
    safe_limit = max(1, min(int(limit), len(stops_df)))
    df = stops_df.head(safe_limit)
    stops = df.rename(columns={
        "stop_id":   "stop_id",
        "stop_name": "stop_name",
        "stop_lat":  "lat",
        "stop_lon":  "lon"
    })[["stop_id", "stop_name", "lat", "lon"]].copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    stops["mode"] = stops["stop_id"].apply(lambda stop_id: "railway" if stop_id in METRO_STOP_IDS else "bus")
    stops["is_parent_station"] = stops["stop_id"].isin(METRO_STOP_IDS)
    stops["line_numbers"] = stops["stop_id"].map(METRO_LINE_BY_STOP_ID).fillna("")
    # Sort: railway first, then bus
    stops = stops.sort_values(by=["mode", "stop_name"], ascending=[False, True]).reset_index(drop=True)
    return {"stops": stops.to_dict(orient="records")}

# ─── Stop Search ──────────────────────────────
@api.get("/stops/search")
def search_stops(q: str = "", limit: int = 8):
    if not q.strip():
        return {"stops": []}
    mask    = stops_df["stop_name"].str.contains(q, case=False, na=False)
    results = stops_df[mask].head(limit)
    stops   = results.rename(columns={
        "stop_lat": "lat",
        "stop_lon": "lon"
    })[["stop_id", "stop_name", "lat", "lon"]].copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    stops["mode"] = stops["stop_id"].apply(lambda stop_id: "railway" if stop_id in METRO_STOP_IDS else "bus")
    stops["is_parent_station"] = stops["stop_id"].isin(METRO_STOP_IDS)
    stops["line_numbers"] = stops["stop_id"].map(METRO_LINE_BY_STOP_ID).fillna("")
    return {"stops": stops.to_dict(orient="records")}


@api.get("/metro/stations")
def get_metro_stations():
    if metro_catalog_df.empty:
        return {"stops": []}

    stops = metro_catalog_df.rename(columns={
        "station_name": "stop_name",
    })[["stop_id", "stop_name", "lat", "lon", "line_numbers", "station_type"]].copy()
    stops["mode"] = "railway"
    stops["is_parent_station"] = True
    return {"stops": stops.to_dict(orient="records")}


@api.get("/station/{stop_id}/next-arrivals")
def station_next_arrivals(stop_id: str, limit: int = Query(default=6, ge=1, le=25)):
    if station_bus_eta_df.empty:
        return {"rows": []}

    rows = station_bus_eta_df[station_bus_eta_df["stop_id"] == str(stop_id)].copy()
    if rows.empty:
        return {"rows": []}

    preferred_cols = [
        "station_name",
        "stop_name",
        "stop_id",
        "bus_route_number",
        "bus_route_name",
        "direction",
        "estimated_arrival",
        "estimated_arrival_in_min",
        "prediction_score_100",
        "prediction_confidence",
        "forecast_generated_at",
    ]

    existing = [col for col in preferred_cols if col in rows.columns]
    rows = rows[existing].head(limit)

    forecast_data = get_station_forecast(stop_id)
    base_eta = forecast_data.get("current", {}).get("estimated_arrival_in_min")
    if not isinstance(base_eta, (int, float)):
        base_eta = None

    bus_gap = forecast_data.get("by_mode", {}).get("bus", {}).get("gap_min")
    if not isinstance(bus_gap, (int, float)) or bus_gap <= 0:
        bus_gap = 4

    resolved_rows = []
    for idx, (_, row) in enumerate(rows.iterrows()):
        record = row.to_dict()
        eta_raw = str(record.get("estimated_arrival_in_min", "")).strip()
        if not eta_raw:
            fallback_base = base_eta if base_eta is not None else 5
            record["estimated_arrival_in_min"] = max(0, int(round(fallback_base + idx * bus_gap)))
            record["estimated_arrival"] = f"{record['estimated_arrival_in_min']} min"

        resolved_rows.append(record)

    return {"rows": resolved_rows}


@api.get("/station/{stop_id}/bus-summary")
def station_bus_summary(stop_id: str):
    if bus_station_summary_df.empty:
        return {"summary": None}

    rows = bus_station_summary_df[bus_station_summary_df["stop_id"] == str(stop_id)].copy()
    if rows.empty:
        return {"summary": None}

    row = rows.iloc[0].to_dict()
    return {
        "summary": {
            "stop_id": str(row.get("stop_id", stop_id)),
            "stop_name": row.get("stop_name", ""),
            "bus_route_numbers": _split_summary_field(row.get("bus_route_numbers", "")),
            "bus_route_names": _split_summary_field(row.get("bus_route_names", "")),
            "directions": _split_summary_field(row.get("directions", "")),
            "route_direction_pairs": _split_summary_field(row.get("route_direction_pairs", "")),
            "eastbound_southbound_routes": _split_summary_field(row.get("eastbound_southbound_routes", "")),
            "westbound_northbound_routes": _split_summary_field(row.get("westbound_northbound_routes", "")),
            "route_count": int(float(row.get("route_count", 0) or 0)),
        }
    }

# ─── Station Forecast ─────────────────────────
@api.get("/station/{stop_id}")
def station_forecast(stop_id: str):
    return get_station_forecast(stop_id)

# ─── Weather ──────────────────────────────────
@api.get("/weather/current")
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
@api.get("/bikeshare/stations")
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
    
@api.get("/debug/station/{stop_id}")
def debug_station(stop_id: str):
    import requests, math
    if stops_df.empty:
        return {"error": "stops data not available on server"}
    
    def haversine(lat1,lon1,lat2,lon2):
        R=6371; dl=math.radians(lat2-lat1); dlo=math.radians(lon2-lon1)
        a=math.sin(dl/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlo/2)**2
        return R*2*math.asin(math.sqrt(a))
    
    stop = stops_df[stops_df["stop_id"] == int(stop_id)].iloc[0]
    stop_lat, stop_lon = stop["stop_lat"], stop["stop_lon"]
    
    resp = requests.get(
        "https://retro.umoiq.com/service/publicJSONFeed?command=vehicleLocations&a=ttc",
        timeout=10
    )
    vehicles = resp.json()["vehicle"]
    
    nearby_025 = [v for v in vehicles if haversine(stop_lat, stop_lon, float(v["lat"]), float(v["lon"])) <= 0.25]
    nearby_05  = [v for v in vehicles if haversine(stop_lat, stop_lon, float(v["lat"]), float(v["lon"])) <= 0.5]
    nearby_10  = [v for v in vehicles if haversine(stop_lat, stop_lon, float(v["lat"]), float(v["lon"])) <= 1.0]
    
    return {
        "stop": stop["stop_name"],
        "lat": stop_lat,
        "lon": stop_lon,
        "total_vehicles": len(vehicles),
        "within_025km": len(nearby_025),
        "within_05km":  len(nearby_05),
        "within_10km":  len(nearby_10),
        "nearest_vehicles": sorted([
            {"id": v["id"], "route": v["routeTag"], 
             "dist": round(haversine(stop_lat, stop_lon, float(v["lat"]), float(v["lon"])), 3)}
            for v in vehicles
        ], key=lambda x: x["dist"])[:5]
    }

# ─── Register API router ───────────────────────────────────────────────────
app.include_router(api)

# ─── Serve built frontend (production) ────────────────────────────────────
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
_frontend_assets = os.path.join(_frontend_dist, "assets")
_frontend_index = os.path.join(_frontend_dist, "index.html")
if os.path.isdir(_frontend_dist) and os.path.isdir(_frontend_assets) and os.path.isfile(_frontend_index):
    app.mount("/assets", StaticFiles(directory=_frontend_assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        return FileResponse(_frontend_index)