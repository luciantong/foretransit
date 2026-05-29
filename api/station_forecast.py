import pandas as pd
import json
import hashlib
from collections import Counter
from datetime import datetime
from pipeline.utils import (
    haversine, parse_gtfs_time, classify_mode,
    is_valid_vehicle, is_valid_delay, deduplicate_vehicles
)
from models.MAGI import run_magi

# ─── Load Static Data Once ───────────────────
GTFS_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data"

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


stops_df = safe_read_csv(f"{GTFS_PATH}/stops.txt")
stop_times_df = safe_read_csv(f"{GTFS_PATH}/stop_times.txt")
trips_df = safe_read_csv(f"{GTFS_PATH}/trips.txt")
routes_df = safe_read_csv(f"{GTFS_PATH}/routes.txt")

# ─── Get Vehicles Near A Stop ────────────────
def get_vehicles_near_stop(stop_lat, stop_lon, radius_km=0.25):
    with open("data/raw/gtfs_realtime/vehicles_latest.json") as f:
        rt_data = json.load(f)

    raw_vehicles = rt_data["data"]["vehicle"]

    # 1. Deduplicate by vehicle ID
    raw_vehicles = deduplicate_vehicles(raw_vehicles)

    nearby = []
    for v in raw_vehicles:
        # 2. Reject implausible records before doing any math
        if not is_valid_vehicle(v):
            continue
        try:
            dist = haversine(
                stop_lat, stop_lon,
                float(v["lat"]), float(v["lon"]))
            if dist <= radius_km:
                v["dist_km"] = round(dist, 4)
                nearby.append(v)
        except Exception:
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
    except Exception:
        pass
    return {
        "rain": 0, "snow": 0,
        "wind": 0,
        "visibility": 10000,
        "temperature": 10
    }

def _stable_section_id(route_id):
    text = str(route_id or "unknown")
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _factor_tone_from_score(score_100):
    if score_100 >= 70:
        return "positive"
    if score_100 <= 44:
        return "negative"
    return "neutral"


def _factor_color_from_score(score_100):
    if score_100 >= 70:
        return "green"
    if score_100 <= 44:
        return "red"
    return "yellow"


def _make_factor(label, impact, score_100):
    score_value = int(max(0, min(100, round(score_100))))
    return {
        "factor": label,
        "impact": impact,
        "score_100": score_value,
        "tone": _factor_tone_from_score(score_value),
        "color": _factor_color_from_score(score_value),
    }


# ─── Get Top Delay Factors ───────────────────
def get_top_factors(delay_seconds, weather, speed, vehicle_count=0):
    factors = []
    if delay_seconds > 300:
        factors.append(_make_factor("High dwell time", "high", 20))
    elif delay_seconds > 180:
        factors.append(_make_factor("Elevated dwell time", "medium", 35))

    if weather["snow"] > 0:
        factors.append(_make_factor(f"Snowfall {weather['snow']}cm", "high", 22))
    if weather["rain"] > 5:
        factors.append(_make_factor(f"Heavy rain {weather['rain']}mm", "high", 28))
    elif weather["rain"] > 1:
        factors.append(_make_factor(f"Rain {weather['rain']}mm", "medium", 42))

    if weather["wind"] > 45:
        factors.append(_make_factor(f"Strong wind {weather['wind']}km/h", "high", 30))
    elif weather["wind"] > 30:
        factors.append(_make_factor(f"Wind {weather['wind']}km/h", "medium", 45))

    if 0 < speed < 5:  # guard: 0 means no data, not slow
        factors.append(_make_factor("Very slow traffic", "high", 24))

    if vehicle_count >= 8:
        factors.append(_make_factor(f"High nearby vehicle volume ({vehicle_count})", "medium", 40))
    elif 0 < vehicle_count <= 2 and delay_seconds < 180:
        factors.append(_make_factor(f"Few vehicles nearby ({vehicle_count})", "low", 82))

    dow = datetime.now().strftime("%A")
    if dow == "Monday":
        factors.append(_make_factor("Monday peak demand", "medium", 46))
    if dow == "Sunday":
        factors.append(_make_factor("Sunday pattern usually lowers delay", "low", 78))

    if not factors:
        return [_make_factor("No strong confounding factors detected right now", "low", 76)]
    return factors[:3]

# ─── Build MAGI features for one mode group ──
def _build_features(mode_delays, weather, now):
    if not mode_delays:
        return None

    avg_delay = sum(d["delay_seconds"] for d in mode_delays) / len(mode_delays)
    avg_speed = sum(d["speed_kmh"]     for d in mode_delays) / len(mode_delays)

    # Inter-vehicle gap from sorted scheduled arrival times
    arrival_times = sorted(
        d["scheduled_time"] for d in mode_delays if d.get("scheduled_time"))
    if len(arrival_times) >= 2:
        gaps    = [(arrival_times[i+1] - arrival_times[i]).total_seconds()
                   for i in range(len(arrival_times) - 1)]
        avg_gap = sum(gaps) / len(gaps)
    else:
        avg_gap = 0

    route_ids = [str(d.get("route_id", "")).strip() for d in mode_delays if d.get("route_id") is not None]
    section_source = route_ids[0] if route_ids else "unknown"

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
        "section_id":            _stable_section_id(section_source),
        "mode":                  mode_delays[0]["mode"]
    }

# ─── Bike Share Toronto ───────────────────────
def get_nearby_bikeshare(stop_lat, stop_lon, radius_km=0.3):
    try:
        import requests
        GBFS_INFO   = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information"
        GBFS_STATUS = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_status"
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
                    "station_id":       sid,
                    "name":             name,
                    "lat":              lat,
                    "lon":              lon,
                    "dist_km":          round(dist, 3),
                    "bikes_available":  s["num_bikes_available"],
                    "ebikes_available": s.get("num_ebikes_available", 0),
                    "docks_available":  s["num_docks_available"],
                    "is_renting":       bool(s["is_renting"])
                })
        return sorted(nearby, key=lambda x: x["dist_km"])
    except Exception:
        return []

# ─── Main: Get Station Forecast ──────────────
def get_station_forecast(stop_id):
    if stops_df is None:
        return {
            "error": "GTFS stops file is missing. Please refresh static data and retry.",
            "details": f"Missing required file in {GTFS_PATH}: stops.txt",
        }

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

    # 4. Match each vehicle to a scheduled arrival
    delays_by_mode = {"bus": [], "streetcar": [], "subway": []}
    seen_vehicle_ids = set()   # dedup safety net

    for v in nearby_vehicles:
        try:
            vid       = v["id"]
            route_tag = v["routeTag"]
            speed     = float(v["speedKmHr"])

            # Skip if we already processed this vehicle
            if vid in seen_vehicle_ids:
                continue
            seen_vehicle_ids.add(vid)

            # ── Detect mode ──
            if routes_df is not None:
                route_info = routes_df[
                    routes_df["route_id"].astype(str) == str(route_tag)]
                route_type = int(route_info["route_type"].iloc[0]) \
                             if not route_info.empty else 3
            else:
                route_type = 3
            mode = classify_mode(route_type)

            # Degraded mode: if schedule files are unavailable, keep vehicle-level
            # signal using zero-delay features instead of failing the endpoint.
            if stop_times_df is None or trips_df is None:
                delays_by_mode[mode].append({
                    "vehicle_id":     vid,
                    "route_id":       route_tag,
                    "direction_id":   -1,
                    "delay_seconds":  0,
                    "speed_kmh":      speed,
                    "mode":           mode,
                    "scheduled_time": None,
                })
                continue

            # ── Direction-aware schedule lookup ──
            # Use direction_id from trips to avoid matching
            # vehicles going the opposite way on the same route
            route_trips = trips_df[
                trips_df["route_id"].astype(str) == str(route_tag)
            ][["trip_id", "direction_id"]]

            if route_trips.empty:
                delays_by_mode[mode].append({
                    "vehicle_id":     vid,
                    "route_id":       route_tag,
                    "direction_id":   -1,
                    "delay_seconds":  0,
                    "speed_kmh":      speed,
                    "mode":           mode,
                    "scheduled_time": None,
                })
                continue

            # Filter stop_times to trips that serve this stop
            scheduled = stop_times_df[
                (stop_times_df["stop_id"] == int(stop_id)) &
                (stop_times_df["trip_id"].isin(route_trips["trip_id"]))
            ].copy()

            if scheduled.empty:
                continue

            # Attach direction_id so we can filter by it
            scheduled = scheduled.merge(
                route_trips[["trip_id", "direction_id"]],
                on="trip_id", how="left")

            # Parse times, drop unparseable rows (handles 25:xx:xx midnight times)
            scheduled["parsed_time"] = scheduled["arrival_time"].apply(parse_gtfs_time)
            scheduled = scheduled.dropna(subset=["parsed_time"])

            if scheduled.empty:
                continue

            scheduled["time_diff"] = abs(
                scheduled["parsed_time"] - now).dt.total_seconds()

            closest        = scheduled.loc[scheduled["time_diff"].idxmin()]
            scheduled_time = closest["parsed_time"]
            delay_seconds  = (now - scheduled_time).total_seconds()

            # ── Reject implausible delays (bad schedule match) ──
            if not is_valid_delay(delay_seconds):
                continue

            delays_by_mode[mode].append({
                "vehicle_id":     vid,
                "route_id":       route_tag,
                "direction_id":   int(closest.get("direction_id", -1)),
                "delay_seconds":  round(delay_seconds),
                "speed_kmh":      speed,
                "mode":           mode,
                "scheduled_time": scheduled_time   # stripped before response
            })

        except Exception:
            continue

    # 5. Run MAGI separately per mode
    magi_by_mode = {}

    for mode, mode_delays in delays_by_mode.items():
        if not mode_delays:
            continue

        features    = _build_features(mode_delays, weather, now)
        magi_result = run_magi(features)

        avg_delay = features["delay_seconds"]
        avg_speed = features["commercial_speed"]
        avg_gap   = features["gap_seconds"]

        safe_delays = [
            {k: v for k, v in d.items() if k != "scheduled_time"}
            for d in mode_delays
        ]

        magi_by_mode[mode] = {
            "label":        magi_result["label"],
            "color":        magi_result["color"],
            "predicted":    magi_result["predicted"],
            "delay_min":    round(avg_delay / 60, 1),
            "gap_min":      round(avg_gap / 60, 1) if len(mode_delays) >= 2 else None,
            "num_vehicles": len(mode_delays),
            "top_factors":  get_top_factors(avg_delay, weather, avg_speed, len(mode_delays)),
            "vehicles":     safe_delays,
            "magi":         magi_result
        }

    # 6. Overall summary
    all_delays = [d for md in delays_by_mode.values() for d in md]

    if all_delays:
        overall_avg_delay = sum(d["delay_seconds"] for d in all_delays) / len(all_delays)
        overall_avg_speed = sum(d["speed_kmh"]     for d in all_delays) / len(all_delays)
        dominant_mode     = Counter(d["mode"] for d in all_delays).most_common(1)[0][0]
    else:
        overall_avg_delay = 0
        overall_avg_speed = 0
        dominant_mode     = "bus"

    overall_features = _build_features(all_delays, weather, now) or {
        "delay_seconds": 0, "gap_seconds": 0,
        "cumulative_dwell_time": 0, "cumulative_leg_time": 0,
        "cumulative_stops": 0, "rain_mm": weather["rain"],
        "wind_speed": weather["wind"], "visibility_km": weather["visibility"] / 1000,
        "snow_mm": weather["snow"], "temperature_c": weather["temperature"],
        "commercial_speed": 0, "hour_of_day": now.hour,
        "is_peak_hour": int(7 <= now.hour <= 9 or 16 <= now.hour <= 18),
        "is_sunday": int(now.weekday() == 6), "day_of_week": now.weekday(),
        "section_id": _stable_section_id(str(stop_id)),
        "mode": dominant_mode
    }
    overall_magi = run_magi(overall_features)

    # 7. Bike Share
    bikeshare = get_nearby_bikeshare(stop_lat, stop_lon)

    # 8. Return
    return {
        "stop_id":      stop_id,
        "stop_name":    stop_name,
        "lat":          stop_lat,
        "lon":          stop_lon,
        "generated_at": now.isoformat(),

        "current": {
            "label":         overall_magi["label"],
            "color":         overall_magi["color"],
            "predicted":     overall_magi["predicted"],
            "delay_min":     round(overall_avg_delay / 60, 1),
            "num_vehicles":  len(all_delays),
            "dominant_mode": dominant_mode
        },

        "by_mode":   magi_by_mode,
        "bikeshare": bikeshare,

        "top_factors": get_top_factors(
            overall_avg_delay, weather, overall_avg_speed, len(all_delays)),
        "weather":     weather,
        "magi":        overall_magi
    }