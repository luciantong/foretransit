import pandas as pd
import json
import math
from datetime import datetime, timedelta

stops_df = pd.read_csv("data/raw/gtfs_static/TTC Routes and Schedules Data/stops.txt")
stop_times_df = pd.read_csv("data/raw/gtfs_static/TTC Routes and Schedules Data/stop_times.txt")
trips_df = pd.read_csv("data/raw/gtfs_static/TTC Routes and Schedules Data/trips.txt")
routes_df = pd.read_csv("data/raw/gtfs_static/TTC Routes and Schedules Data/routes.txt")

with open("data/raw/gtfs_realtime/vehicles_latest.json", "r") as f:
    gtfs_rt_data = json.load(f)

vehicles = gtfs_rt_data["vehicles"]["vehicle"]
capture_time = gtfs_rt_data["captured_at"]
#----- Haverstine distance function - curved distance between two points on a sphere -----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

#Calculate Nearest stop for each vehicle
def find_nearest_stop(lat, lon):
    stops_df["dist"] = stops_df.apply(
        lambda r: haversine(lat, lon,
                            r["stop_lat"],
                            r["stop_lon"]), axis=1)
    nearest = stops_df.loc[stops_df["dist"].idxmin()]
    return nearest["stop_id"], nearest["dist"]

# ---- Transform GTFS Time into Actual human time ------
def parse_gtfs_time(time_str):
    h, m, s  = map(int, time_str.split(":"))
    base     = datetime.now().replace(
                    hour=0, minute=0,
                    second=0, microsecond=0)
    return base + timedelta(hours=h, minutes=m, seconds=s)

# ---- Classification delay based on (Chen Et. Al 2020) - https://arxiv.org/pdf/2006.16180.pdf ----
def classify_delay(seconds):
    if seconds < 60:   return 0  # on time
    if seconds < 180:  return 1  # minor
    if seconds < 300:  return 2  # moderate
    return 3                     # severe
# ---- Severity helpers ----
def severity_label(severity):
    return {
        0: "on_time",
        1: "minor",
        2: "moderate",
        3: "severe"
    }[severity]

def severity_color(severity):
    return {
        0: "green",
        1: "yellow",
        2: "orange",
        3: "red"
    }[severity]

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