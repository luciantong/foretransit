import pandas as pd
import json
import os
import math
import requests
import zipfile
import io
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
GTFS_ZIP_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/7795b45e-e65a-4465-81fc-c36b9dfff169/resource/cfb6b2b8-6191-41e3-bda1-b175c51148cb/download/TTC%20Routes%20and%20Schedules%20Data.zip"

def _ensure_gtfs_static():
    if not os.path.exists(f"{GTFS_PATH}/stop_times.txt"):
        print("stop_times.txt not found — downloading GTFS static data...")
        os.makedirs("data/raw/gtfs_static", exist_ok=True)
        r = requests.get(GTFS_ZIP_URL, timeout=120)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            print("Zip contents:", z.namelist()[:10])
            z.extractall("data/raw/gtfs_static/TTC Routes and Schedules Data")
        print("GTFS static data downloaded and extracted.")

_ensure_gtfs_static()

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

stops_df  = safe_read_csv(f"{GTFS_PATH}/stops.txt")
trips_df  = safe_read_csv(f"{GTFS_PATH}/trips.txt")
routes_df = safe_read_csv(f"{GTFS_PATH}/routes.txt")

# Load stop_times with only needed columns to save memory
try:
    stop_times_df = pd.read_csv(
        f"{GTFS_PATH}/stop_times.txt",
        usecols=["trip_id", "stop_id", "arrival_time", "stop_sequence"]
    )
except Exception:
    stop_times_df = None
    print("WARNING: stop_times.txt could not be loaded")

# ─── Get Vehicles Near A Stop ────────────────
def get_vehicles_near_stop(stop_lat, stop_lon, radius_km=0.25):
    try:
        resp = requests.get(
            "https://retro.umoiq.com/service/publicJSONFeed?command=vehicleLocations&a=ttc",
            timeout=10
        )
        raw_vehicles = resp.json()["vehicle"]
    except Exception:
        raw_vehicles = []

    raw_vehicles = deduplicate_vehicles(raw_vehicles)

    nearby = []
    for v in raw_vehicles:
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
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast?latitude=43.6532&longitude=-79.3832&hourly=rain,snowfall,windspeed_10m,visibility,temperature_2m&forecast_days=1",
            timeout=5
        ).json()
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%dT%H:00")
        times = resp["hourly"]["time"]
        if current_hour in times:
            idx = times.index(current_hour)
            return {
                "rain":        resp["hourly"]["rain"][idx],
                "snow":        resp["hourly"]["snowfall"][idx],
                "wind":        resp["hourly"]["windspeed_10m"][idx],
                "visibility":  resp["hourly"]["visibility"][idx],
                "temperature": resp["hourly"]["temperature_2m"][idx]
            }
    except Exception:
        pass
    return {"rain": 0, "snow": 0, "wind": 0, "visibility": 10000, "temperature": 10}

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
    if 0 < speed < 5:
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
        "delay_seconds":         max(avg_delay, 0),
        "gap_seconds":           avg_gap,
        "cumulative_dwell_time": max(avg_delay, 0),
        "cumulative_leg_time":   max(avg_delay * 4, 0),
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
def get_nearby_bikeshare(stop_lat, stop_lon, radius_km=1.25):
    try:
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

# ─── Dynamic Blending Helper ──────────────────
def blend_prediction_score(raw_predicted, num_vehicles, mode="bus"):
    baselines = {"subway": 85.0, "streetcar": 74.0, "bus": 76.0}
    fallback_score = baselines.get(mode.lower(), 76.0)
    if num_vehicles == 0:
        return int(fallback_score)
    weight = 1.0 - math.exp(-1.6 * num_vehicles)
    weight = max(0.0, min(1.0, weight))
    blended = (raw_predicted * weight) + (fallback_score * (1.0 - weight))
    return int(max(0, min(100, round(blended))))

# ─── Main: Get Station Forecast ──────────────
def get_station_forecast(stop_id):
    if stops_df is None:
        return {
            "error": "GTFS stops file is missing.",
            "details": f"Missing required file in {GTFS_PATH}: stops.txt",
        }

    stop = stops_df[stops_df["stop_id"] == int(stop_id)]
    if stop.empty:
        return {"error": f"Stop {stop_id} not found"}

    stop      = stop.iloc[0]
    stop_lat  = stop["stop_lat"]
    stop_lon  = stop["stop_lon"]
    stop_name = stop["stop_name"]

    nearby_vehicles = get_vehicles_near_stop(stop_lat, stop_lon)
    weather = get_current_weather()
    now     = datetime.now()

    delays_by_mode   = {"bus": [], "streetcar": [], "subway": []}
    seen_vehicle_ids = set()

    for v in nearby_vehicles:
        try:
            vid       = v["id"]
            route_tag = v["routeTag"]
            speed     = float(v["speedKmHr"])

            if vid in seen_vehicle_ids:
                continue
            seen_vehicle_ids.add(vid)

            if routes_df is not None:
                route_info = routes_df[routes_df["route_id"].astype(str) == str(route_tag)]
                route_type = int(route_info["route_type"].iloc[0]) if not route_info.empty else 3
            else:
                route_type = 3
            mode = classify_mode(route_type)

            # If schedule data unavailable, record zero-delay placeholder
            if stop_times_df is None or trips_df is None:
                delays_by_mode[mode].append({
                    "vehicle_id": vid, "route_id": route_tag,
                    "direction_id": -1, "delay_seconds": 0,
                    "speed_kmh": speed, "mode": mode, "scheduled_time": None,
                })
                continue

            route_trips = trips_df[
                trips_df["route_id"].astype(str) == str(route_tag)
            ][["trip_id", "direction_id"]]

            if route_trips.empty:
                delays_by_mode[mode].append({
                    "vehicle_id": vid, "route_id": route_tag,
                    "direction_id": -1, "delay_seconds": 0,
                    "speed_kmh": speed, "mode": mode, "scheduled_time": None,
                })
                continue

            scheduled = stop_times_df[
                (stop_times_df["stop_id"] == int(stop_id)) &
                (stop_times_df["trip_id"].isin(route_trips["trip_id"]))
            ].copy()

            # Fallback: find nearest stop on this route if this stop not served
            if scheduled.empty:
                try:
                    candidates = stop_times_df[
                        stop_times_df["trip_id"].isin(route_trips["trip_id"])
                    ].copy()
                    if not candidates.empty and stops_df is not None:
                        candidates = candidates.merge(
                            stops_df[["stop_id", "stop_lat", "stop_lon"]],
                            on="stop_id", how="left"
                        ).dropna(subset=["stop_lat", "stop_lon"])
                        vlat, vlon = float(v.get("lat")), float(v.get("lon"))
                        candidates["_dist_km"] = candidates.apply(
                            lambda r: haversine(vlat, vlon, float(r["stop_lat"]), float(r["stop_lon"])),
                            axis=1
                        )
                        nearby_cands = candidates[candidates["_dist_km"] <= 1.0]
                        if not nearby_cands.empty:
                            closest_stop = nearby_cands.loc[nearby_cands["_dist_km"].idxmin()]
                            scheduled = stop_times_df[
                                (stop_times_df["stop_id"] == closest_stop["stop_id"]) &
                                (stop_times_df["trip_id"].isin(route_trips["trip_id"]))
                            ].copy()
                except Exception:
                    pass

            if scheduled.empty:
                delays_by_mode[mode].append({
                    "vehicle_id": vid, "route_id": route_tag,
                    "direction_id": -1, "delay_seconds": 0,
                    "speed_kmh": speed, "mode": mode, "scheduled_time": None,
                })
                continue

            scheduled = scheduled.merge(
                route_trips[["trip_id", "direction_id"]], on="trip_id", how="left")
            scheduled["parsed_time"] = scheduled["arrival_time"].apply(parse_gtfs_time)
            scheduled = scheduled.dropna(subset=["parsed_time"])

            if scheduled.empty:
                continue

            scheduled["time_diff"] = (scheduled["parsed_time"] - now).dt.total_seconds()
            upcoming = scheduled[scheduled["time_diff"] >= -60]
            closest  = upcoming.loc[upcoming["time_diff"].idxmin()] \
                       if not upcoming.empty \
                       else scheduled.loc[scheduled["time_diff"].idxmin()]

            scheduled_time = closest.get("parsed_time")
            if scheduled_time is None:
                continue

            delay_seconds = (now - scheduled_time).total_seconds()

            if not is_valid_delay(delay_seconds):
                delays_by_mode[mode].append({
                    "vehicle_id": vid, "route_id": route_tag,
                    "direction_id": int(closest.get("direction_id", -1)),
                    "delay_seconds": 0, "speed_kmh": speed,
                    "mode": mode, "scheduled_time": scheduled_time
                })
                continue

            delays_by_mode[mode].append({
                "vehicle_id":   vid,
                "route_id":     route_tag,
                "direction_id": int(closest.get("direction_id", -1)),
                "delay_seconds": round(delay_seconds),
                "speed_kmh":    speed,
                "mode":         mode,
                "scheduled_time": scheduled_time
            })
        except Exception:
            continue

    # 5. Run MAGI per mode
    magi_by_mode = {}
    for mode, mode_delays in delays_by_mode.items():
        if not mode_delays:
            continue

        features    = _build_features(mode_delays, weather, now)
        magi_result = run_magi(features)

        standardized_score = blend_prediction_score(
            magi_result["selection_context"]["winner_score_100"],
            len(mode_delays), mode=mode)

        avg_delay = features["delay_seconds"]
        avg_speed = features["commercial_speed"]
        avg_gap   = features["gap_seconds"]

        safe_delays = [{k: v for k, v in d.items() if k != "scheduled_time"}
                       for d in mode_delays]

        magi_by_mode[mode] = {
            "label":        magi_result["label"],
            "color":        magi_result["color"],
            "predicted":    standardized_score,
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

    global_standardized_score = blend_prediction_score(
        overall_magi["selection_context"]["winner_score_100"],
        len(all_delays), mode=dominant_mode)

    bikeshare = get_nearby_bikeshare(stop_lat, stop_lon)

    return {
        "stop_id":      stop_id,
        "stop_name":    stop_name,
        "lat":          stop_lat,
        "lon":          stop_lon,
        "generated_at": now.isoformat(),

        "current": {
            "label":         overall_magi["label"],
            "color":         overall_magi["color"],
            "predicted":     global_standardized_score,
            "delay_min":     round(overall_avg_delay / 60, 1),
            "num_vehicles":  len(all_delays),
            "dominant_mode": dominant_mode
        },

        "by_mode":   magi_by_mode,
        "bikeshare": bikeshare,

        "top_factors": (
            [_make_factor("No real-time vehicle data; using schedule defaults", "low", 76)]
            if len(all_delays) == 0
            else get_top_factors(overall_avg_delay, weather, overall_avg_speed, len(all_delays))
        ),
        "weather":     weather,
        "magi":        overall_magi,
        "no_realtime": len(all_delays) == 0
    }