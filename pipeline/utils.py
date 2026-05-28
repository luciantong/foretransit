import pandas as pd
import json
import math
from datetime import datetime, timedelta

GTFS_STATIC_PATH = "data/raw/gtfs_static/TTC Routes and Schedules Data"


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


stops_df = safe_read_csv(f"{GTFS_STATIC_PATH}/stops.txt")
stop_times_df = safe_read_csv(f"{GTFS_STATIC_PATH}/stop_times.txt")
trips_df = safe_read_csv(f"{GTFS_STATIC_PATH}/trips.txt")
routes_df = safe_read_csv(f"{GTFS_STATIC_PATH}/routes.txt")

try:
    with open("data/raw/gtfs_realtime/vehicles_latest.json", "r") as f:
        gtfs_rt_data = json.load(f)
except Exception:
    gtfs_rt_data = {"data": {"vehicle": []}, "timestamp": ""}

vehicles = gtfs_rt_data.get("data", {}).get("vehicle", [])
capture_time = gtfs_rt_data.get("timestamp", "")     # fixed key

# ─── Haversine ───────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

# ─── Nearest stop ────────────────────────────
def find_nearest_stop(lat, lon):
    if stops_df is None or stops_df.empty:
        raise ValueError("stops.txt is missing or empty")
    stops_df["dist"] = stops_df.apply(
        lambda r: haversine(lat, lon,
                            r["stop_lat"],
                            r["stop_lon"]), axis=1)
    nearest = stops_df.loc[stops_df["dist"].idxmin()]
    return nearest["stop_id"], nearest["dist"]

# ─── GTFS time parser ────────────────────────
# Handles times past midnight e.g. 25:30:00
def parse_gtfs_time(time_str):
    try:
        h, m, s = map(int, str(time_str).split(":"))
        base = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0)
        return base + timedelta(hours=h, minutes=m, seconds=s)
    except Exception:
        return None   # caller must dropna

# ─── Mode classifier ─────────────────────────
def classify_mode(route_type: int) -> str:
    if route_type == 0: return "streetcar"
    if route_type == 1: return "subway"
    return "bus"

# ─── Delay classifier (Chen et al. 2020) ─────
def classify_delay(seconds):
    if seconds < 60:  return 0   # on time  (includes early)
    if seconds < 180: return 1   # minor
    if seconds < 300: return 2   # moderate
    return 3                     # severe

# ─── Severity helpers ────────────────────────
def severity_label(severity):
    return {0: "on_time", 1: "minor",
            2: "moderate", 3: "severe"}[severity]

def severity_color(severity):
    return {0: "green", 1: "yellow",
            2: "orange", 3: "red"}[severity]

# ─── Data cleansing helpers ───────────────────

# Plausible speed range for TTC vehicles (km/h)
MIN_SPEED_KMH = 0
MAX_SPEED_KMH = 120

# Max believable delay window in either direction (seconds)
# Beyond ±2 hours almost certainly a bad schedule match
MAX_DELAY_ABS = 7200

def is_valid_vehicle(v: dict) -> bool:
    """
    Rejects vehicles with implausible or missing data.
    Returns False if the record should be skipped.
    """
    try:
        lat   = float(v["lat"])
        lon   = float(v["lon"])
        speed = float(v["speedKmHr"])
    except (KeyError, ValueError, TypeError):
        return False

    # Coordinate sanity — Toronto bounding box
    if not (43.4 <= lat <= 43.9):  return False
    if not (-79.7 <= lon <= -79.1): return False

    # Speed sanity
    if speed < MIN_SPEED_KMH or speed > MAX_SPEED_KMH:
        return False

    # Must have an ID and routeTag
    if not v.get("id") or not v.get("routeTag"):
        return False

    return True

def is_valid_delay(delay_seconds: float) -> bool:
    """
    Rejects delays that are implausibly large (bad schedule match).
    Note: negative = early, which is valid but capped.
    """
    return abs(delay_seconds) <= MAX_DELAY_ABS

def deduplicate_vehicles(vehicles: list) -> list:
    """
    Keeps only the most recent record per vehicle ID.
    If the feed has duplicates, last one wins.
    """
    seen = {}
    for v in vehicles:
        vid = v.get("id")
        if vid:
            seen[vid] = v
    return list(seen.values())

### TRASH BELOW - FOR TESTING PURPOSES ONLY - NOT PRODUCTION CODE

"""""""""""
def calculate_delays():
    records = []

    try:
        for v in vehicles: #Iterate through each vehicle in the GTFS Realtime feed
            lat = float(v["lat"])
            lon = float(v["lon"])
            route_tag = v["routeTag"]
            vehicle_id = v["id"]    
            speed = float(v["speedKmHr"])

            stop_id, dist_km = find_nearest_stop(lat, lon)

            if dist_km > 0.1: #skip vehicles who are 100m or more away from a stop
                continue

            route_trips = trips_df[trips_df["route_id"].astype(str) == str(route_tag)]["trip_id"].tolist() #use trip_id to look up stop times

            now = datetime.now()

            scheduled = stop_times_df[
                (stop_times_df["stop_id"] == stop_id) &
                (stop_times_df["trip_id"].isin(route_trips))
            ]
            
            if scheduled.empty:
                continue
        
        scheduled["parsed_time"] = scheduled[
            "arrival_time"].apply(parse_gtfs_time)
        scheduled["time_diff"] = abs(
            scheduled["parsed_time"] - now
        ).dt.total_seconds()

        closest = scheduled.loc[
            scheduled["time_diff"].idxmin()]
        
        actual_time = now
        scheduled_time = closest["parsed_time"]

        delay_seconds = (actual_time - scheduled_time).total_seconds()

        #Compile all relevant data into a dataframe
        records.append({
            "vehicle_id":      vehicle_id,
            "route_id":        route_tag,
            "stop_id":         stop_id,
            "stop_sequence":   closest["stop_sequence"],
            "lat":             lat,
            "lon":             lon,
            "speed_kmh":       speed,
            "scheduled_time":  scheduled_time.isoformat(),
            "actual_time":     actual_time.isoformat(),
            "delay_seconds":   round(delay_seconds),
            "delay_severity":  classify_delay(delay_seconds),
            "dist_to_stop_km": round(dist_km, 4),
            "captured_at":     capture_time
        })
    except Exception as e:
        pass

    print(f"✓ Processed {len(records)} vehicle-stop matches")
    return records

if __name__ == "__main__":
    records = calculate_delays()

    if records:
        df = pd.DataFrame(records)
        df.to_csv("data/processed/delays.csv", index=False)
        print(f"✓ Saved to data/processed/delays.csv")
        print(df[["route_id", "stop_id",
                  "delay_seconds",
                  "delay_severity"]].head(10))
    else:
        print("⚠️ No records matched")
"""""""""""