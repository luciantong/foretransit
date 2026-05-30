"""
pipeline/fetch_historical_delays.py
Downloads TTC historical delay data from Toronto Open Data
and converts it into XGBoost-compatible training features.

Package IDs (from Toronto Open Data):
  Bus:       e271cdae-8788-4980-96ce-6a5c95bc6618
  Streetcar: b68cb708-561a-4de2-a2cb-f56b4ec77213
  Subway:    996cfe8d-fb35-40ce-b569-698d51fc683b

Usage:
  PYTHONPATH=. python3 pipeline/fetch_historical_delays.py
"""

import os
import io
import requests
import pandas as pd
import numpy as np

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action"

PACKAGES = {
    "bus":       "e271cdae-8788-4980-96ce-6a5c95bc6618",
    "streetcar": "b68cb708-561a-4de2-a2cb-f56b4ec77213",
    "subway":    "996cfe8d-fb35-40ce-b569-698d51fc683b",
}

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_resources(package_id):
    """List all downloadable resources for a package."""
    url = f"{CKAN_API}/package_show?id={package_id}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()["result"]["resources"]


def download_resource(resource):
    """Download a resource and return as a DataFrame."""
    url    = resource["url"]
    fmt    = resource.get("format", "").upper()
    name   = resource.get("name", url)
    print(f"    Downloading: {name} ({fmt})")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    if fmt in ("XLSX", "XLS"):
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name=None)
        # Some files have multiple sheets — concatenate all
        frames = []
        for sheet_name, sheet_df in df.items():
            sheet_df["_sheet"] = sheet_name
            frames.append(sheet_df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if fmt == "CSV":
        return pd.read_csv(io.StringIO(resp.text))

    print(f"    Skipping unsupported format: {fmt}")
    return pd.DataFrame()


def normalize_bus_streetcar(df, mode):
    """
    Normalize bus/streetcar delay columns to XGBoost feature format.

    Raw columns vary by year but typically include:
      Report Date, Route, Time, Day, Location, Incident, Min Delay, Min Gap, Direction
    """
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Find delay column (varies: 'min_delay', 'delay', 'min delay')
    delay_col = next((c for c in df.columns if "delay" in c and "min" in c), None)
    gap_col   = next((c for c in df.columns if "gap" in c), None)
    date_col  = next((c for c in df.columns if "date" in c), None)
    time_col  = next((c for c in df.columns if "time" in c), None)
    route_col = next((c for c in df.columns if "route" in c), None)

    if delay_col is None:
        print(f"    Could not find delay column in {list(df.columns)[:8]}")
        return pd.DataFrame()

    df = df.dropna(subset=[delay_col]).copy()
    df[delay_col] = pd.to_numeric(df[delay_col], errors="coerce").fillna(0)

    # Parse datetime
    if date_col and time_col:
        try:
            df["_datetime"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce"
            )
        except Exception:
            df["_datetime"] = pd.NaT
    elif date_col:
        df["_datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["_datetime"] = pd.NaT

    df["_datetime"] = df["_datetime"].fillna(pd.Timestamp.now())

    # Delay in seconds
    delay_seconds = df[delay_col] * 60  # column is in minutes

    # Classify severity
    def classify(s):
        if s < 60:   return 0
        if s < 180:  return 1
        if s < 300:  return 2
        return 3

    records = pd.DataFrame({
        "cumulative_dwell_time": (df["_datetime"].dt.minute + 1) * 30,
        "cumulative_leg_time":   (df["_datetime"].dt.minute + 1) * 120,
        "cumulative_stops":      np.random.randint(3, 25, size=len(df)),
        "day_of_week":           df["_datetime"].dt.dayofweek,
        "section_id":            pd.to_numeric(
                                     df[route_col].astype(str)
                                     .str.extract(r"(\d+)")[0]
                                     .fillna(0), errors="coerce"
                                 ).fillna(0).astype(int) % 100
                                 if route_col else 0,
        "hour_of_day":           df["_datetime"].dt.hour,
        "is_sunday":             (df["_datetime"].dt.dayofweek == 6).astype(int),
        "route_type":            0 if mode == "streetcar" else 3,
        "delay_seconds":         delay_seconds,
        "gap_seconds":           (df[gap_col] * 60) if gap_col else 0,
        "delay_severity":        delay_seconds.apply(classify),
        "source":                f"toronto_open_data_{mode}",
    })

    return records


def normalize_subway(df):
    """
    Normalize subway delay columns.
    Raw columns: Date, Time, Day, Station, Code, Min Delay, Min Gap, Bound, Line, Vehicle
    """
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    delay_col = next((c for c in df.columns if "delay" in c), None)
    gap_col   = next((c for c in df.columns if "gap" in c), None)
    date_col  = next((c for c in df.columns if "date" in c), None)
    time_col  = next((c for c in df.columns if "time" in c), None)

    if delay_col is None:
        return pd.DataFrame()

    df = df.dropna(subset=[delay_col]).copy()
    df[delay_col] = pd.to_numeric(df[delay_col], errors="coerce").fillna(0)

    if date_col and time_col:
        try:
            df["_datetime"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce"
            )
        except Exception:
            df["_datetime"] = pd.NaT
    elif date_col:
        df["_datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["_datetime"] = pd.NaT

    df["_datetime"] = df["_datetime"].fillna(pd.Timestamp.now())
    delay_seconds   = df[delay_col] * 60

    def classify(s):
        if s < 60:  return 0
        if s < 180: return 1
        if s < 300: return 2
        return 3

    records = pd.DataFrame({
        "cumulative_dwell_time": (df["_datetime"].dt.minute + 1) * 30,
        "cumulative_leg_time":   (df["_datetime"].dt.minute + 1) * 120,
        "cumulative_stops":      np.random.randint(2, 15, size=len(df)),
        "day_of_week":           df["_datetime"].dt.dayofweek,
        "section_id":            np.random.randint(0, 100, size=len(df)),
        "hour_of_day":           df["_datetime"].dt.hour,
        "is_sunday":             (df["_datetime"].dt.dayofweek == 6).astype(int),
        "route_type":            1,  # subway
        "delay_seconds":         delay_seconds,
        "gap_seconds":           (df[gap_col] * 60) if gap_col else 0,
        "delay_severity":        delay_seconds.apply(classify),
        "source":                "toronto_open_data_subway",
    })

    return records


def fetch_mode(mode, package_id):
    print(f"\n{'='*50}")
    print(f"Fetching {mode.upper()} delay data...")
    print(f"{'='*50}")

    try:
        resources = get_resources(package_id)
    except Exception as e:
        print(f"  ERROR fetching resource list: {e}")
        return pd.DataFrame()

    # Only download recent years to keep it manageable
    recent = [r for r in resources if any(
        str(yr) in r.get("name", "") for yr in ["2022", "2023", "2024", "2025"]
    )]
    if not recent:
        recent = resources[-4:]  # fallback: last 4 resources

    print(f"  Found {len(resources)} total resources, downloading {len(recent)} recent ones")

    all_frames = []
    for resource in recent:
        try:
            df = download_resource(resource)
            if df.empty:
                continue

            if mode == "subway":
                normed = normalize_subway(df)
            else:
                normed = normalize_bus_streetcar(df, mode)

            if not normed.empty:
                all_frames.append(normed)
                print(f"    -> {len(normed)} records normalized")
        except Exception as e:
            print(f"    ERROR processing resource: {e}")
            continue

    if not all_frames:
        print(f"  No data fetched for {mode}")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"  Total {mode} records: {len(combined)}")
    print(f"  Severity distribution:\n{combined['delay_severity'].value_counts().sort_index()}")
    return combined


def main():
    print("TTC Historical Delay Data Fetcher")
    print("Sourcing from Toronto Open Data Portal")

    all_modes = []

    for mode, package_id in PACKAGES.items():
        df = fetch_mode(mode, package_id)
        if not df.empty:
            # Save per-mode file
            out_path = f"{OUTPUT_DIR}/{mode}_historical_delays.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved to {out_path}")
            all_modes.append(df)

    if not all_modes:
        print("\nERROR: No data fetched. Check your internet connection.")
        return

    # Combine all modes
    combined = pd.concat(all_modes, ignore_index=True)
    combined_path = f"{OUTPUT_DIR}/all_historical_delays.csv"
    combined.to_csv(combined_path, index=False)

    print(f"\n{'='*50}")
    print(f"DONE — {len(combined)} total historical delay records")
    print(f"Combined file: {combined_path}")
    print(f"\nOverall severity distribution:")
    print(combined["delay_severity"].value_counts().sort_index())
    print(f"\nNext step: retrain XGBoost with:")
    print(f"  PYTHONPATH=. python3 models/train_xgboost.py")


if __name__ == "__main__":
    main()