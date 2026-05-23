import pandas as pd
import json
from collections import Counter
from datetime import datetime
from pipeline.utils import haversine, parse_gtfs_time
from models.MAGI import run_magi

# ─── Load Static Data Once ───────────────────
GTFS_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data"

stops_df      = pd.read_csv(f"{GTFS_PATH}/stops.txt")
stop_times_df = pd.read_csv(f"{GTFS_PATH}/stop_times.txt")
trips_df      = pd.read_csv(f"{GTFS_PATH}/trips.txt")
routes_df     = pd.read_csv(f"{GTFS_PATH}/routes.txt")

# ─── Mode Classification ─────────────────────
# GTFS route_type spec:
#   0 = tram/streetcar, 1 = subway, 3 = bus (default)
def classify_mode(route_type: int) -> str:
    if route_type == 0: return "streetcar"
    if route_type == 1: return "subway"
    return "bus"

# ─── Get Vehicles Near A Stop ────────────────
def get_vehicles_near_stop(stop_lat, stop_lon, radius_km=0.1):
    with open("data/raw/gtfs_realtime/vehicles_latest.json") as f:
        rt_data = json.load(f)

    nearby = []
    for v in rt_data["data"]["vehicle"]:
        try:
            dist = haversine(
                stop_lat, stop_lon,
                float(v["lat"]), float(v["lon"]))
            if dist <= radius_km:
                v["dist_km"] = round(dist, 4)
                nearby.append(v)
        except:
            continue
    return nearby

# ─── Get Current Weather ─────────────────────
def get_current_weather():
    try:
        with open("data/raw/weather/weather_latest.json") as f:
            weather = json.load(f)
        now          = datetime.now()
        current_hour = now.strftime("%Y-%m-%dT%H:00")
        times        = weather["hourly"]["time"]
        if current_hour in times:
            idx = times.index(current_hour)
            return {
                "rain":        weather["hourly"]["rain"][idx],
                "snow":        weather["hourly"]["snowfall"][idx],
                "wind":        weather["hourly"]["windspeed_10m"][idx],
                "visibility":  weather["hourly"]["visibility"][idx],
                "temperature": weather["hourly"]["temperature_2m"][idx]
            }
    except:
        pass
    return {
        "rain": 0, "snow": 0,
        "wind": 0,
        "visibility": 10000,
        "temperature": 10
    }

# ─── Get Top Delay Factors ───────────────────
def get_top_factors(delay_seconds, weather, speed):
    factors = []
    if delay_seconds > 180:
        factors.append({"factor": "High dwell time",                    "impact": "high"})
    if weather["snow"] > 0:
        factors.append({"factor": f"Snowfall {weather['snow']}cm",      "impact": "high"})
    if weather["rain"] > 1:
        factors.append({"factor": f"Rain {weather['rain']}mm",          "impact": "medium"})
    if weather["wind"] > 30:
        factors.append({"factor": f"Wind {weather['wind']}km/h",        "impact": "medium"})
    if speed < 5:
        factors.append({"factor": "Very slow traffic",                   "impact": "high"})
    dow = datetime.now().strftime("%A")
    if dow == "Monday":
        factors.append({"factor": "Monday peak",                         "impact": "medium"})
    if dow == "Sunday":
        factors.append({"factor": "Sunday — historically low delay",     "impact": "low"})
    return factors[:3]

# ─── Build MAGI features for one mode group ──
def _build_features(mode_delays, weather, now):
    if not mode_delays:
        return None

    avg_delay = sum(d["delay_seconds"] for d in mode_delays) / len(mode_delays)
    avg_speed = sum(d["speed_kmh"]     for d in mode_delays) / len(mode_delays)

    # Inter-vehicle gap from sorted scheduled arrival times
    arrival_times = [d["scheduled_time"] for d in mode_delays
                     if d.get("scheduled_time")]
    if len(arrival_times) >= 2:
        arrival_times.sort()
        gaps    = [(arrival_times[i+1] - arrival_times[i]).total_seconds()
                   for i in range(len(arrival_times) - 1)]
        avg_gap = sum(gaps) / len(gaps)
    else:
        avg_gap = 0

    return {
        "delay_seconds":         avg_delay,
        "gap_seconds":           avg_gap,
        "cumulative_dwell_time": avg_delay / 60,
        "cumulative_leg_time":   avg_delay / 60,
        "cumulative_stops":      len(mode_delays),
        "rain_mm":               weather["rain"],
        "wind_speed":            weather["wind"],
        "visibility_km":         weather["visibility"] / 1000,
        "snow_mm":               weather["snow"],
        "temperature_c":         weather["temperature"],
        "commercial_speed":      avg_speed,
        "hour_of_day":           now.hour,
        "is_peak_hour":          int(7 <= now.hour <= 9 or 16 <= now.hour <= 18),
        "is_sunday":             int(now.weekday() == 6),
        "day_of_week":           now.weekday(),
        "mode":                  mode_delays[0]["mode"]
    }

# ─── Bike Share Toronto ───────────────────────
def get_nearby_bikeshare(stop_lat, stop_lon, radius_km=0.3):
    """
    Pulls live Bike Share Toronto availability.
    Returns list of nearby stations sorted by distance.
    No delay model — availability only.
    """
    import requests
    GBFS_INFO   = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information"
    GBFS_STATUS = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_status"

    try:
        info   = requests.get(GBFS_INFO,   timeout=3).json()
        status = requests.get(GBFS_STATUS, timeout=3).json()

        coords = {
            s["station_id"]: (s["lat"], s["lon"], s["name"])
            for s in info["data"]["stations"]
        }

        nearby = []
        for s in status["data"]["stations"]:
            sid = s["station_id"]
            if sid not in coords:
                continue
            lat, lon, name = coords[sid]
            dist = haversine(stop_lat, stop_lon, lat, lon)
            if dist <= radius_km:
                nearby.append({
                    "station_id":      sid,
                    "name":            name,
                    "dist_km":         round(dist, 3),
                    "bikes_available": s["num_bikes_available"],
                    "ebikes_available": s.get("num_ebikes_available", 0),
                    "docks_available": s["num_docks_available"],
                    "is_renting":      bool(s["is_renting"])
                })

        return sorted(nearby, key=lambda x: x["dist_km"])

    except Exception:
        return []

# ─── Main: Get Station Forecast ──────────────
# Runs when user clicks a station on the map
def get_station_forecast(stop_id):

    # 1. Get stop info
    stop = stops_df[stops_df["stop_id"] == int(stop_id)]
    if stop.empty:
        return {"error": f"Stop {stop_id} not found"}

    stop      = stop.iloc[0]
    stop_lat  = stop["stop_lat"]
    stop_lon  = stop["stop_lon"]
    stop_name = stop["stop_name"]

    # 2. Get vehicles near this stop
    nearby_vehicles = get_vehicles_near_stop(stop_lat, stop_lon)

    # 3. Get weather + time
    weather = get_current_weather()
    now     = datetime.now()

    # 4. Match each vehicle to a scheduled arrival and classify its mode
    #    delays_by_mode buckets every matched vehicle by transit type
    delays_by_mode = {"bus": [], "streetcar": [], "subway": []}

    for v in nearby_vehicles:
        try:
            route_tag = v["routeTag"]
            speed     = float(v["speedKmHr"])

            # ── Detect mode from GTFS route_type ──
            route_info = routes_df[
                routes_df["route_id"].astype(str) == str(route_tag)]
            route_type = int(route_info["route_type"].iloc[0]) \
                         if not route_info.empty else 3
            mode = classify_mode(route_type)

            # ── Find scheduled arrival at this stop ──
            route_trips = trips_df[
                trips_df["route_id"].astype(str) == str(route_tag)
            ]["trip_id"].tolist()

            scheduled = stop_times_df[
                (stop_times_df["stop_id"] == int(stop_id)) &
                (stop_times_df["trip_id"].isin(route_trips))
            ].copy()

            if scheduled.empty:
                continue

            scheduled["parsed_time"] = scheduled["arrival_time"].apply(parse_gtfs_time)
            scheduled["time_diff"]   = abs(
                scheduled["parsed_time"] - now).dt.total_seconds()

            closest        = scheduled.loc[scheduled["time_diff"].idxmin()]
            scheduled_time = closest["parsed_time"]
            delay_seconds  = (now - scheduled_time).total_seconds()

            delays_by_mode[mode].append({
                "vehicle_id":    v["id"],
                "route_id":      route_tag,
                "delay_seconds": round(delay_seconds),
                "speed_kmh":     speed,
                "mode":          mode,
                "scheduled_time": scheduled_time   # kept for gap calc, not serialised
            })

        except:
            continue

    # 5. Run MAGI separately for each mode that has vehicles
    magi_by_mode = {}

    for mode, mode_delays in delays_by_mode.items():
        if not mode_delays:
            continue

        features    = _build_features(mode_delays, weather, now)
        magi_result = run_magi(features)

        avg_delay = features["delay_seconds"]
        avg_speed = features["commercial_speed"]
        avg_gap   = features["gap_seconds"]

        # Strip non-serialisable scheduled_time before returning
        safe_delays = [
            {k: v for k, v in d.items() if k != "scheduled_time"}
            for d in mode_delays
        ]

        magi_by_mode[mode] = {
            "label":        magi_result["label"],
            "color":        magi_result["color"],
            "predicted":    magi_result["predicted"],
            "delay_min":    round(avg_delay / 60, 1),
            "gap_min":      round(avg_gap   / 60, 1),
            "num_vehicles": len(mode_delays),
            "top_factors":  get_top_factors(avg_delay, weather, avg_speed),
            "vehicles":     safe_delays,
            "magi":         magi_result
        }

    # 6. Overall summary — weighted by vehicle count across all modes
    all_delays = [d for md in delays_by_mode.values() for d in md]

    if all_delays:
        overall_avg_delay = sum(d["delay_seconds"] for d in all_delays) / len(all_delays)
        overall_avg_speed = sum(d["speed_kmh"]     for d in all_delays) / len(all_delays)
        dominant_mode     = Counter(d["mode"] for d in all_delays).most_common(1)[0][0]
    else:
        overall_avg_delay = 0
        overall_avg_speed = 0
        dominant_mode     = "bus"

    # Overall MAGI uses dominant mode's features (or a fresh aggregate)
    overall_features = _build_features(all_delays, weather, now) or {
        "delay_seconds": 0, "gap_seconds": 0,
        "cumulative_dwell_time": 0, "cumulative_leg_time": 0,
        "cumulative_stops": 0, "rain_mm": weather["rain"],
        "wind_speed": weather["wind"], "visibility_km": weather["visibility"] / 1000,
        "snow_mm": weather["snow"], "temperature_c": weather["temperature"],
        "commercial_speed": 0, "hour_of_day": now.hour,
        "is_peak_hour": int(7 <= now.hour <= 9 or 16 <= now.hour <= 18),
        "is_sunday": int(now.weekday() == 6), "day_of_week": now.weekday(),
        "mode": dominant_mode
    }
    overall_magi = run_magi(overall_features)

    # 7. Bike Share (separate — no delay model, just availability)
    bikeshare = get_nearby_bikeshare(stop_lat, stop_lon)

    # 8. Return everything
    return {
        "stop_id":      stop_id,
        "stop_name":    stop_name,
        "lat":          stop_lat,
        "lon":          stop_lon,
        "generated_at": now.isoformat(),

        # Top-level summary (overall, all modes combined)
        "current": {
            "label":        overall_magi["label"],
            "color":        overall_magi["color"],
            "predicted":    overall_magi["predicted"],
            "delay_min":    round(overall_avg_delay / 60, 1),
            "num_vehicles": len(all_delays),
            "dominant_mode": dominant_mode
        },

        # Per-mode breakdown — each key present only if vehicles exist
        "by_mode":   magi_by_mode,

        # Bike Share nearby (independent of TTC MAGI)
        "bikeshare": bikeshare,

        "top_factors": get_top_factors(
            overall_avg_delay, weather, overall_avg_speed),
        "weather":     weather,
        "magi":        overall_magi
    }