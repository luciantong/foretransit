import pandas as pd
import json
import math
from datetime import datetime, timedelta
from models.MAGI import run_magi

# ─── Load Static Data Once ───────────────────
GTFS_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data"

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return radius_km * 2 * math.asin(math.sqrt(a))


def parse_gtfs_time(time_str):
    h, m, s = map(int, time_str.split(":"))
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return base + timedelta(hours=h, minutes=m, seconds=s)


stops_df = safe_read_csv(f"{GTFS_PATH}/stops.txt")
stop_times_df = safe_read_csv(f"{GTFS_PATH}/stop_times.txt")
trips_df = safe_read_csv(f"{GTFS_PATH}/trips.txt")
routes_df = safe_read_csv(f"{GTFS_PATH}/routes.txt")

# ─── Get Vehicles Near A Stop ────────────────
def get_vehicles_near_stop(stop_lat, stop_lon,
                            radius_km=0.1):
    with open("data/raw/gtfs_realtime/vehicles_latest.json") as f:
        rt_data = json.load(f)

    nearby = []
    for v in rt_data["vehicles"]["vehicle"]:
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
        with open("data/weather/weather_latest.json") as f:
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
        factors.append({
            "factor": "High dwell time",
            "impact": "high"
        })
    if weather["snow"] > 0:
        factors.append({
            "factor": f"Snowfall {weather['snow']}cm",
            "impact": "high"
        })
    if weather["rain"] > 1:
        factors.append({
            "factor": f"Rain {weather['rain']}mm",
            "impact": "medium"
        })
    if weather["wind"] > 30:
        factors.append({
            "factor": f"Wind {weather['wind']}km/h",
            "impact": "medium"
        })
    if speed < 5:
        factors.append({
            "factor": "Very slow traffic",
            "impact": "high"
        })
    dow = datetime.now().strftime("%A")
    if dow == "Monday":
        factors.append({
            "factor": "Monday peak",
            "impact": "medium"
        })
    if dow == "Sunday":
        factors.append({
            "factor": "Sunday — historically low delay",
            "impact": "low"
        })
    return factors[:3]

# ─── Main: Get Station Forecast ──────────────
# Runs when user clicks a station on the map
def get_station_forecast(stop_id):
    warnings = []

    if stops_df is None or trips_df is None:
        return {
            "error": "GTFS static files are missing. stops.txt and trips.txt are required.",
            "details": "Verify files exist under data/raw/gtfs_static/TTC Routes and Schedules Data/",
        }
    if stop_times_df is None:
        warnings.append("stop_times.txt missing: using fallback delay estimate based on nearby vehicles.")

    # 1. Get stop info
    stop = stops_df[stops_df["stop_id"] == int(stop_id)]
    if stop.empty:
        return {"error": f"Stop {stop_id} not found"}

    stop      = stop.iloc[0]
    stop_lat  = stop["stop_lat"]
    stop_lon  = stop["stop_lon"]
    stop_name = stop["stop_name"]

    # 2. Get vehicles near this stop only
    nearby_vehicles = get_vehicles_near_stop(
                            stop_lat, stop_lon)

    # 3. Get weather
    weather = get_current_weather()
    now     = datetime.now()

    # 4. Calculate delay per vehicle
    delays = []
    speeds = []

    for v in nearby_vehicles:
        try:
            route_tag = v["routeTag"]
            speed     = float(v["speedKmHr"])
            speeds.append(speed)

            if stop_times_df is None:
                delays.append({
                    "vehicle_id":    v["id"],
                    "route_id":      route_tag,
                    "delay_seconds": 0,
                    "speed_kmh":     speed
                })
                continue

            route_trips = trips_df[
                trips_df["route_id"].astype(str) ==
                str(route_tag)
            ]["trip_id"].tolist()

            scheduled = stop_times_df[
                (stop_times_df["stop_id"] == int(stop_id)) &
                (stop_times_df["trip_id"].isin(route_trips))
            ]

            if scheduled.empty:
                continue

            scheduled = scheduled.copy()
            scheduled["parsed_time"] = scheduled[
                "arrival_time"].apply(parse_gtfs_time)
            scheduled["time_diff"] = abs(
                scheduled["parsed_time"] - now
            ).dt.total_seconds()

            closest        = scheduled.loc[
                                scheduled["time_diff"].idxmin()]
            scheduled_time = closest["parsed_time"]
            delay_seconds  = (
                now - scheduled_time).total_seconds()

            delays.append({
                "vehicle_id":    v["id"],
                "route_id":      route_tag,
                "delay_seconds": round(delay_seconds),
                "speed_kmh":     speed
            })
        except:
            continue

    # 5. Aggregate
    if delays:
        avg_delay = sum(d["delay_seconds"]
                       for d in delays) / len(delays)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
    else:
        avg_delay = 0
        avg_speed = 0

    # 6. Build features for MAGI
    features = {
        "delay_seconds":         avg_delay,
        "gap_seconds":           avg_speed,
        "cumulative_dwell_time": avg_delay / 60,
        "cumulative_leg_time":   avg_delay / 60,
        "cumulative_stops":      len(delays),
        "rain_mm":               weather["rain"],
        "wind_speed":            weather["wind"],
        "visibility_km":         weather["visibility"] / 1000,
        "snow_mm":               weather["snow"],
        "temperature_c":         weather["temperature"],
        "commercial_speed":      avg_speed,
        "hour_of_day":           now.hour,
        "is_peak_hour":          int(7 <= now.hour <= 9 or
                                     16 <= now.hour <= 18),
        "is_sunday":             int(now.weekday() == 6),
        "day_of_week":           now.weekday(),
        "mode":                  "bus"
    }

    # 7. Run MAGI — picks best model
    magi_result = run_magi(features)

    # 8. Return to frontend
    return {
        "stop_id":      stop_id,
        "stop_name":    stop_name,
        "lat":          stop_lat,
        "lon":          stop_lon,
        "generated_at": now.isoformat(),

        "current": {
            "score":        magi_result["predicted"],
            "label":        magi_result["label"],
            "color":        magi_result["color"],
            "delay_min":    round(avg_delay / 60, 1),
            "num_vehicles": len(delays)
        },

        "top_factors":     get_top_factors(
                                avg_delay,
                                weather,
                                avg_speed),
        "warnings":        warnings,
        "weather":         weather,
        "vehicles_nearby": delays,
        "magi":            magi_result
    }