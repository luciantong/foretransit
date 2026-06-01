# ForéTransit

Transit delay forecasting API for the Toronto Transit Commission (TTC). Predicts delay severity at any stop in real time using an ensemble of three models — Probit (Chen et al. 2007), XGBoost, and the Jayden Method — selected automatically by the MAGI system.

---

## How it works

When a user clicks a stop on the map, the frontend calls the API which:

1. Finds all TTC vehicles currently near that stop
2. Compares their positions against the scheduled timetable to calculate delay
3. Pulls the current Toronto weather
4. Runs **MAGI** — three models compete, the one with the lowest log loss wins
5. Returns a JSON forecast with severity label, delay in minutes, weather, and which model won

### MAGI — Multi-Answer Geographical Informations

| Model | Method | Based on |
|---|---|---|
| Probit | Ordered probit with fixed beta coefficients | Chen et al. (2007) |
| XGBoost | Gradient boosted classifier, trained per transit mode | Toronto GTFS data |
| Jayden Method | Weighted formula: delay × 0.7 + congestion × 0.3 | Custom |

Models are scored by log loss. The winner is used as the final prediction.

---

## Data sources

| Source | What | How |
|---|---|---|
| [TTC Open Data](https://www.ttc.ca/open-data) | Static schedule (stops, routes, timetables) | Manual download |
| [Umo IQ](https://retro.umoiq.com/service/publicJSONFeed?command=vehicleLocations&a=ttc) | Live vehicle positions and speed | Auto-fetched |
| [Open-Meteo](https://open-meteo.com) | Hourly weather for Toronto | Auto-fetched |

---

## Performance & Trends

### TTC Bus Network Average Commercial Speed (2019–2026)

![TTC Bus Average Speed 2019-2026](https://raw.githubusercontent.com/luciantong/foretransit/main/images/ttc_bus_average_speed.png)

The TTC bus network has experienced a steady decline in average commercial speed from 18.9 km/h in 2019 to 17.6 km/h in 2023. Projections indicate stabilization at 17.2 km/h through 2026 pending infrastructure improvements.

### Historical Decline & Recovery Target

![TTC Speed Trend](https://raw.githubusercontent.com/luciantong/foretransit/main/images/ttc_speed_trend.png)

This chart shows the actual speed decline (red) and the projected recovery trajectory (orange dashed line) based on TTC Corporate Plan targets. The stabilization at 17.2 km/h reflects infrastructure initiatives aimed at addressing congestion.

---

## Project structure

```
foretransit/
├── api/
│   ├── main.py               # FastAPI routes
│   ├── station_forecast.py   # Core forecast logic
│   └── cache.py              # In-memory TTL cache (60s)
│
├── models/
│   ├── MAGI.py               # Ensemble — runs all three, picks winner
│   ├── probit_model.py       # Chen et al. probit model
│   ├── xgboost_model.py      # XGBoost predictor
│   ├── jayden_model.py       # Custom delay + congestion scoring
│   ├── train_xgboost.py      # Training script (run once)
│   └── saved/                # Trained .pkl files (git ignored)
│
├── pipeline/
│   ├── fetch_gtfs.py         # Fetches realtime vehicles + weather
│   └── utils.py              # Shared helpers
│
└── data/
    └── raw/
        ├── gtfs_static/      # TTC schedule — manual download
        ├── gtfs_realtime/    # Live vehicles — auto fetched
        └── weather/          # Hourly weather — auto fetched
```

---

## Try ForéTransit

### 🚀 Live Demo

**Visit the app now:** [ForéTransit on Vercel](https://foretransit.vercel.app)

Click any subway or bus stop on the map to see real-time delay predictions powered by MAGI ensemble modeling.

---

## Local Development Setup (Optional)

To run locally for development:

### 1. Clone the repo

```bash
git clone https://github.com/your-repo/foretransit.git
cd foretransit
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn xgboost scikit-learn pandas scipy
```

### 3. Download static GTFS data (one time only)

1. Go to https://open.toronto.ca/dataset/merged-gtfs-ttc-routes-and-schedules/
2. Download **TTC Routes and Schedules (GTFS)**
3. Unzip and place the contents at:

```
data/raw/gtfs_static/TTC Routes and Schedules Data/
```

These four files must be present:

```
stops.txt
stop_times.txt
trips.txt
routes.txt
```

### 4. Create required folders

```bash
mkdir -p data/raw/gtfs_realtime
mkdir -p data/raw/weather
mkdir -p models/saved
```

### 5. Start backend

```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### 6. Start frontend (in another terminal)

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173

### 5. Train XGBoost models (optional, one time only)

```bash
python -m models.train_xgboost
```

This saves trained models to `models/saved/`. Only needed if retraining on new data.

---
Drop `--reload` in production — it's for development only and restarts the server on every file change.

To keep data fresh in production, set a cron job to fetch every 60 seconds:

```bash
# crontab -e
* * * * * cd /path/to/foretransit && python pipeline/fetch_gtfs.py
```

---

## API endpoints

Base URL: `http://localhost:8000`

### `GET /`
Health check.

```json
{ "status": "ForéTransit API is running" }
```

---

### `GET /station/{stop_id}`
Main endpoint. Returns a full delay forecast for a stop.

`stop_id` — any TTC stop ID from `stops.txt`

**Example:** `GET /station/14200`

```json
{
  "stop_id": "14200",
  "stop_name": "Bloor St East At Church St",
  "location_type": 0,
  "parent_station": null,
  "lat": 43.6702,
  "lon": -79.3757,
  "generated_at": "2026-05-22T14:32:00",

  "current": {
    "score": 2,
    "label": "moderate",
    "color": "orange",
    "delay_min": 4.2,
    "num_vehicles": 3,
    "mode": "bus",
    "gap_min": 6.1
  },

  "top_factors": [
    { "factor": "High dwell time", "impact": "high" },
    { "factor": "Snowfall 0.5cm",  "impact": "high" },
    { "factor": "Monday peak",     "impact": "medium" }
  ],

  "weather": {
    "rain": 0,
    "snow": 0.5,
    "wind": 22,
    "visibility": 8000,
    "temperature": -3
  },

  "vehicles_nearby": [
    {
      "vehicle_id": "4201",
      "route_id": "25",
      "delay_seconds": 252,
      "speed_kmh": 14.3,
      "mode": "bus"
    }
  ],

  "magi": {
    "model_used": "xgboost",
    "predicted": 2,
    "label": "moderate",
    "color": "orange",
    "all_models": {
      "probit":  { "predicted": 1, "probabilities": { "on_time": 0.52, "minor": 0.28, "moderate": 0.14, "severe": 0.06 } },
      "xgboost": { "predicted": 2, "probabilities": { "on_time": 0.18, "minor": 0.31, "moderate": 0.38, "severe": 0.13 } },
      "jayden":  { "predicted": 2, "score": 6.4,   "probabilities": { "on_time": 0.1,  "minor": 0.1,  "moderate": 0.7,  "severe": 0.1  } }
    },
    "model_info": {
      "winner": "xgboost",
      "method": "MAGI - Multi-Answer Geographical Informations",
      "reference": "Chen et al. 2007"
    }
  },

  "cached": false
}
```

**Severity scale:**

| score | label | color |
|---|---|---|
| 0 | on_time | green |
| 1 | minor | yellow |
| 2 | moderate | orange |
| 3 | severe | red |

---

### `GET /vehicles/live`
Returns all active TTC vehicles with their current positions, speed, and route.

```json
[
  {
    "id": "4201",
    "routeTag": "25",
    "lat": "43.6712",
    "lon": "-79.3891",
    "speedKmHr": "18.4"
  }
]
```

---

### `GET /debug/cache`
Returns how many stops are currently cached and their keys. Useful during testing.

```json
{
  "count": 3,
  "keys": ["14200", "9021", "7734"]
}
```

---

## How to find a stop ID

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/gtfs_static/TTC Routes and Schedules Data/stops.txt')
print(df[['stop_id', 'stop_name']].head(30))
"
```

---

## Common errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pipeline'` | Run from project root using `python -m`, not `python file.py` |
| `FileNotFoundError: weather_latest.json` | Run `mkdir -p data/raw/weather` then `python pipeline/fetch_gtfs.py` |
| `KeyError: 'vehicles'` | Re-run `python pipeline/fetch_gtfs.py` to refresh stale data |
| `Stop XXXX not found` | Use a real `stop_id` from `stops.txt` — see above |
| XGBoost missing from `all_models` | Run `python -m models.train_xgboost` and confirm `models/saved/` has the `.pkl` files |
| `permission denied` on first run | Make sure you `cd` into the project folder before running any commands |

---

## Report Visualization

Below is the Python code used to generate the system overview analysis, proving why Toronto's scale and multimodal complexity require locally-calibrated models.

## TTC system chart showing the importance of Toronto's local context

``` 
"""
TTC System Overview 2024 - Horizontal Bar Chart
Supports the argument that Toronto's transit delay context requires locally-calibrated models
due to scale, multimodal complexity, and surface-subway interdependence.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

# Data Preparation
categories = [
    'Total TTC trips (2024)', 'Bus trips', 'Subway rides', 'Streetcar trips',
    'Conventional bus/streetcar routes', 'Surface routes connecting\nto subway (A.M. rush)',
    'Subway connections\n(A.M. rush)', 'Streetcar network length'
]
values = [423, 204, 181, 35, 173, 167, 272, 308]
units = ['million', 'million', 'million', 'million', 'routes', 'routes', 'connections', 'km']

colors = [
    '#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', # Ridership
    '#1B7536', '#1B7536', '#1B7536',            # Network structure
    '#D35400'                                    # Infrastructure
]

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

y_positions = range(len(categories))
bars = ax.barh(y_positions, values, color=colors, edgecolor='black', linewidth=0.5, height=0.65)

for bar, value, unit in zip(bars, values, units):
    label_x = bar.get_width() + 5
    ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{value} {unit}', 
            va='center', ha='left', fontsize=10, fontweight='medium')

ax.set_yticks(y_positions)
ax.set_yticklabels(categories, fontsize=10)
ax.set_xlabel('Value (see units in labels)', fontsize=11, labelpad=10)
ax.set_ylabel('TTC System Metric', fontsize=11, labelpad=10)
ax.set_xlim(0, 520)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.invert_yaxis()

fig.suptitle("Why Toronto's Transit Delay Context Is Locally Specific", fontsize=14, fontweight='bold', y=0.96)
subtitle = ('Note: Values represent different measurement units. Colors indicate metric type:\n'
            'ridership (blue), network structure (green), infrastructure (orange).')
fig.text(0.5, 0.91, subtitle, ha='center', va='top', fontsize=9, style='italic', color='#555555', transform=fig.transFigure)

legend_elements = [
    mpatches.Patch(facecolor='#2E86AB', edgecolor='black', linewidth=0.5, label='Ridership (millions of trips)'),
    mpatches.Patch(facecolor='#1B7536', edgecolor='black', linewidth=0.5, label='Network structure (routes/connections)'),
    mpatches.Patch(facecolor='#D35400', edgecolor='black', linewidth=0.5, label='Infrastructure (km)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

source_text = (
    "Source: Author's chart based on data from Toronto Transit Commission, Annual Report 2024; "
    "Toronto Transit Commission, Operating Statistics — 2024; and Toronto Transit Commission, "
    "'TTC Welcomes 60th Streetcar, Expanding Fleet to 264.' "
    "Note: trip figures are in millions; routes, connections and kilometres are shown as raw counts."
)
wrapped_source = textwrap.fill(source_text, width=115)
fig.text(0.5, 0.02, wrapped_source, ha='center', va='bottom', fontsize=8, style='italic', color='#333333', transform=fig.transFigure, linespacing=1.3)

plt.tight_layout()
plt.subplots_adjust(top=0.86, bottom=0.14)

plt.savefig('images/ttc_system_overview.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
```

### Generated Chart

![TTC System Overview](https://raw.githubusercontent.com/luciantong/foretransit/main/images/ttc_system_overview.png)

This chart demonstrates why Toronto's transit system is locally context-specific:

- **423M annual trips** across bus, subway, and streetcar
- **173 conventional routes + 167 surface routes** connecting to subway, especially during A.M. rush
- **272 critical subway connections** during peak hours
- **Multimodal interdependence**: surface transit feeds into subway, making local performance patterns essential

The scale and complexity shown here explain why generic delay models fail for Toronto—locally-calibrated MAGI models are required to capture the unique characteristics of bus-subway integration.

---

## TTC Bus Average Speed

```
# Methodology Figures — Tier 2 Indicators from TTC Sources (No APC data)
# Updated commercial speed series based on TTC Corporate Plan figures provided:
# 2019 Actual: 18.9 km/h
# 2022 Actual: 18.6 km/h
# 2023 Actual: 17.6 km/h
# 2024 Projection: 17.1 km/h
# 2025 Target: 17.2 km/h
# 2026 Target: 17.2 km/h

import matplotlib.pyplot as plt

# Clean academic style
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# =============================================================================
# GRAPH 1 — TTC Bus Average Speed, 2019–2026 (Updated series)
# =============================================================================

years = ["2019 (Actual)", "2022 (Actual)", "2023 (Actual)", "2024 (Projection)", "2025 (Target)", "2026 (Target)"]
avg_speed_kmh = [18.9, 18.6, 17.6, 17.1, 17.2, 17.2]

fig1, ax1 = plt.subplots(figsize=(9.2, 5.4))
fig1.patch.set_facecolor("white")
ax1.set_facecolor("white")

ax1.plot(years, avg_speed_kmh, color="#2E86AB", marker="o", linewidth=2, label="Average speed / target")
ax1.set_title("TTC Bus Average Speed, 2019–2026", pad=12)
ax1.set_xlabel("Year (status)")
ax1.set_ylabel("Bus average speed (km/h)")
ax1.grid(axis="y", linestyle="--", alpha=0.3)

# Value labels
for x, y in zip(years, avg_speed_kmh):
    ax1.text(x, y + 0.18, f"{y:.1f}", ha="center", va="bottom", fontsize=9)

# Figure note
note1 = (
    "Figure 1. Bus Average Kilometres/Hour from the TTC Corporate Plan 2024 Year in Review Progress Report: "
    "2019 Actual 18.9; 2022 Actual 18.6; 2023 Actual 17.6; 2024 Projection 17.1; 2025 Target 17.2; 2026 Target 17.2. "
    "This commercial-speed indicator (Tier 2, AVL-related) supports adapting an ordered probit delay-severity "
    "model to Toronto’s local operating context."
)
fig1.text(0.5, 0.02, note1, ha="center", va="bottom", fontsize=9, style="italic", color="#333333")

plt.tight_layout()
plt.subplots_adjust(bottom=0.26)
plt.savefig("images/ttc_bus_average_speed.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig1)
```

### Generated Chart

![TTC Bus Average Speed 2019-2026](https://raw.githubusercontent.com/luciantong/foretransit/main/images/ttc_bus_average_speed.png)

---

## Monthly Toronto Subway Disruption in 2025

```
# TTC Subway Delay Data Analysis - Monthly Disruptions in 2025
# This script analyzes subway delay incidents from the City of Toronto Open Data

import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Load the TTC Subway Delay Data CSV file
# =============================================================================
file_path = "TTC Subway Delay Data since 2025.csv"  # your filename
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# =============================================================================
# STEP 2: Identify and convert the date column to datetime (no deprecated args)
# =============================================================================
if "Date" in df.columns:
    date_column = "Date"
elif "Report Date" in df.columns:
    date_column = "Report Date"
else:
    date_candidates = [c for c in df.columns if "date" in c.lower()]
    if date_candidates:
        date_column = date_candidates[0]
    else:
        raise ValueError("No date-like column found. Please verify the CSV headers.")

print(f"Using date column: '{date_column}'")

# First pass parse
df[date_column] = pd.to_datetime(df[date_column], errors="coerce", dayfirst=False)

# Optional second pass for potential day-first formats if many failed
if df[date_column].isna().mean() > 0.5:
    mask_nat = df[date_column].isna()
    df.loc[mask_nat, date_column] = pd.to_datetime(df.loc[mask_nat, date_column], errors="coerce", dayfirst=True)

# Drop rows where parsing failed
initial_count = len(df)
df = df.dropna(subset=[date_column])
print(f"Records after removing invalid dates: {len(df)} (removed {initial_count - len(df)})\n")

# =============================================================================
# STEP 3: Filter the data to include only records from 2025
# =============================================================================
df_2025 = df[df[date_column].dt.year == 2025].copy()
print(f"Records from 2025: {len(df_2025)}\n")

if df_2025.empty:
    print("WARNING: No records found for 2025. Available years in the dataset:")
    print(df[date_column].dt.year.value_counts().sort_index())
    raise ValueError("No 2025 data available for analysis.")

# =============================================================================
# STEP 4: Group the data by month
# =============================================================================
df_2025["YearMonth"] = df_2025[date_column].dt.to_period("M")

# =============================================================================
# STEP 5: Count the number of subway delay incidents per month
# =============================================================================
monthly_counts = df_2025.groupby("YearMonth").size().reset_index(name="Disruptions")
monthly_counts = monthly_counts.sort_values("YearMonth")
monthly_counts["MonthLabel"] = monthly_counts["YearMonth"].apply(lambda x: x.strftime("%B %Y"))
monthly_counts["MonthShort"] = monthly_counts["YearMonth"].apply(lambda x: x.strftime("%b"))

# Optional: ensure all months Jan–Dec appear (fill missing with 0)
# full_months = pd.period_range("2025-01", "2025-12", freq="M")
# monthly_counts = (
#     monthly_counts.set_index("YearMonth")
#     .reindex(full_months, fill_value=0)
#     .rename_axis("YearMonth")
#     .reset_index()
# )
# monthly_counts["MonthLabel"] = monthly_counts["YearMonth"].apply(lambda x: x.strftime("%B %Y"))
# monthly_counts["MonthShort"] = monthly_counts["YearMonth"].apply(lambda x: x.strftime("%b"))

# =============================================================================
# Print the monthly disruption table
# =============================================================================
print("=" * 50)
print("MONTHLY TTC SUBWAY DISRUPTIONS IN 2025")
print("=" * 50)

table_display = monthly_counts[["MonthLabel", "Disruptions"]].copy()
table_display.columns = ["Month", "Number of Disruptions"]
table_display = table_display.reset_index(drop=True)

print(table_display.to_string(index=False))
print("-" * 50)
print(f"Total disruptions in 2025: {monthly_counts['Disruptions'].sum()}")
print(f"Average per month: {monthly_counts['Disruptions'].mean():.1f}")
print(f"Highest: {monthly_counts.loc[monthly_counts['Disruptions'].idxmax(), 'MonthLabel']} "
      f"({monthly_counts['Disruptions'].max()} incidents)")
print(f"Lowest: {monthly_counts.loc[monthly_counts['Disruptions'].idxmin(), 'MonthLabel']} "
      f"({monthly_counts['Disruptions'].min()} incidents)")
print("=" * 50 + "\n")

# =============================================================================
# STEP 6: Create a bar chart suitable for academic reports
# =============================================================================
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(
    monthly_counts["MonthShort"],
    monthly_counts["Disruptions"],
    color="#1f77b4",
    edgecolor="black",
    linewidth=0.7,
    alpha=0.9
)

for bar, value in zip(bars, monthly_counts["Disruptions"]):
    ax.annotate(
        f"{value:,}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.set_title("Monthly TTC Subway Disruptions in 2025", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Month", fontsize=12, labelpad=10)
ax.set_ylabel("Number of Subway Disruptions", fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
ax.set_ylim(0, max(5, monthly_counts["Disruptions"].max()) * 1.15)

fig.text(
    0.5, 0.02,
    "Data Source: City of Toronto Open Data - TTC Subway Delay Data",
    ha="center",
    fontsize=9,
    style="italic",
    color="gray"
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# plt.savefig("ttc_subway_disruptions_2025.png", dpi=300, bbox_inches="tight")
plt.show()

```

### Generated Chart

![Monthly TTC Subway Disruptions in 2025](https://raw.githubusercontent.com/luciantong/foretransit/main/images/ttc_subway_disruptions_2025.png)

---

## Top 10 TTC Streetcar Delay Routes since 2025

```

# =============================================================================
# TTC Streetcar Delay Data — Top 10 Delay Categories Since 2025 (No 'Incident' column)
# Uses pandas and matplotlib only
# - Reads local CSV: "TTC Streetcar Delay Data since 2025.csv"
# - Cleans columns; handles missing Date, Time, Min Delay, Min Gap
# - Filters to 2025 onward
# - Aggregates by a chosen key (default: route) to show Top 10 categories
# - Saves 300 dpi PNG suitable for academic reports
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "TTC Streetcar Delay Data since 2025.csv"

if not os.path.exists(csv_path):
    print("ERROR: Could not find 'TTC Streetcar Delay Data since 2025.csv'.")
    print("Please place the CSV file in the same folder as this notebook and rerun the cell.")
else:
    # ----------------------------
    # 1) Load and normalize column names
    # ----------------------------
    df = pd.read_csv(csv_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]", "", regex=True)
        .str.lower()
    )

    # Candidate lists for key fields (no 'incident' required)
    date_candidates = ["date", "report_date", "reportdate", "occurrence_date", "start_date"]
    time_candidates = ["time", "report_time", "occurrence_time", "start_time"]
    route_candidates = ["route", "route_id", "route_name", "line"]
    location_candidates = ["location", "stop", "intersection", "area"]
    mindelay_candidates = ["min_delay", "mindelay", "delay_minutes", "minutes_delayed", "delay_min"]
    mingap_candidates = ["min_gap", "mingap", "gap_minutes", "minutes_gap", "headway_gap"]

    def pick(colnames, candidates):
        for c in candidates:
            if c in colnames:
                return c
        return None

    date_col = pick(df.columns, date_candidates)
    time_col = pick(df.columns, time_candidates)
    route_col = pick(df.columns, route_candidates)
    location_col = pick(df.columns, location_candidates)
    mindelay_col = pick(df.columns, mindelay_candidates)
    mingap_col = pick(df.columns, mingap_candidates)

    print("Detected columns (after cleaning):")
    print(f"  Date -> {date_col}")
    print(f"  Time -> {time_col}")
    print(f"  Route -> {route_col}")
    print(f"  Location -> {location_col}")
    print(f"  Min Delay -> {mindelay_col}")
    print(f"  Min Gap -> {mingap_col}\n")

    if date_col is None:
        raise ValueError("No date-like column found. Look for 'Date' or 'Report Date' in your CSV headers.")

    # ----------------------------
    # 2) Parse and clean
    # ----------------------------
    # Parse dates (try both orders if necessary)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    if df[date_col].isna().mean() > 0.5:
        mask = df[date_col].isna()
        df.loc[mask, date_col] = pd.to_datetime(df.loc[mask, date_col], errors="coerce", dayfirst=True)

    # Optional time parsing
    if time_col is not None:
        try:
            parsed = pd.to_datetime(df[time_col], errors="coerce", format="%H:%M")
        except Exception:
            parsed = pd.to_datetime(df[time_col], errors="coerce")
        df[time_col] = parsed.dt.time

    # Numeric delay/gap
    if mindelay_col is not None:
        df[mindelay_col] = pd.to_numeric(df[mindelay_col], errors="coerce")
    if mingap_col is not None:
        df[mingap_col] = pd.to_numeric(df[mingap_col], errors="coerce")

    # Drop rows missing the date
    before_drop = len(df)
    df = df.dropna(subset=[date_col])
    after_drop = len(df)

    # ----------------------------
    # 3) Filter to 2025 onward
    # ----------------------------
    df_2025p = df[df[date_col].dt.year >= 2025].copy()
    if df_2025p.empty:
        print("No streetcar delay records found from 2025 onward after cleaning.")
    else:
        # ----------------------------
        # 4) Choose grouping key (no Incident column)
        #    Default to 'route'; fallback to 'location' if route is missing.
        # ----------------------------
        group_key = route_col if route_col is not None else location_col
        if group_key is None:
            # If neither route nor location exist, fall back to counting all records (single bar)
            df_2025p["_group"] = "All Records"
            group_key = "_group"

        # Prepare a clean label series for the group key
        df_2025p[group_key] = df_2025p[group_key].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
        # For missing group labels, fill with 'Unknown'
        df_2025p[group_key] = df_2025p[group_key].fillna("Unknown")

        # ----------------------------
        # 5) Aggregate: count of records + total delay minutes (if available)
        # ----------------------------
        agg_dict = {"count": ("__dummy__", "size")}
        # Create a dummy column for counting via agg
        df_2025p["__dummy__"] = 1

        if mindelay_col is not None:
            agg_dict["total_delay_min"] = (mindelay_col, "sum")

        grouped = df_2025p.groupby(group_key).agg(**agg_dict).reset_index()

        # Rank by count of delay records
        grouped = grouped.sort_values("count", ascending=False)

        # Select Top 10
        top10 = grouped.head(10).sort_values("count", ascending=True)

        # ----------------------------
        # 6) Plot — Horizontal bar chart (Top 10 by count)
        # ----------------------------
        plt.rcParams.update({
            "figure.figsize": (12, 7),
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        })

        fig, ax = plt.subplots()

        bars = ax.barh(
            top10[group_key].astype(str),
            top10["count"].astype(int),
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9
        )

        title_dim = "Route" if group_key == route_col else ("Location" if group_key == location_col else "Category")
        ax.set_title(f"Top 10 TTC Streetcar Delay {title_dim}s Since 2025")
        ax.set_xlabel("Number of delay incidents")
        ax.set_ylabel(f"{title_dim}")

        # Value labels at bar ends
        for bar, val in zip(bars, top10["count"].astype(int).values):
            ax.annotate(
                f"{val:,}",
                xy=(val, bar.get_y() + bar.get_height() / 2),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=10
            )

        # Grid and headroom
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        xmax = top10["count"].max() if len(top10) else 0
        ax.set_xlim(0, xmax * 1.12 if xmax > 0 else 1)

        # Caption/source note
        caption = (
            "Note. Data from City of Toronto Open Data, TTC Streetcar Delay Data. "
            "Frequent and uneven delay patterns support a forecast app tailored to Toronto operations."
        )
        fig.subplots_adjust(bottom=0.20)
        fig.text(0.5, 0.05, caption, ha="center", va="center", fontsize=9, color="gray", style="italic")

        plt.tight_layout(rect=[0, 0.08, 1, 1])

        # Save at 300 dpi
        out_file = "ttc_streetcar_top_10_delay_causes_since_2025.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.show()

        # Confirmation
        print("Chart saved as:", out_file)
        print(f"Records in original file: {before_drop:,}")
        print(f"Records after dropping missing dates: {after_drop:,}")
        print(f"Records from 2025 onward used: {len(df_2025p):,}")
```

### Generated Chart

![Top 10 TTC Streetcar Delay Routes Since 2025](https://raw.githubusercontent.com/luciantong/foretransit/main/images/ttc_streetcar_top_10_delay_causes_since_2025.png)

---
## Severe XGBoost Overfitting - Urban Bus Delay Prediction (For illustrative purpose)

```
# ============================================================
# Demonstrating Severe XGBoost Overfitting
# Domain: Urban Public Transit — Bus Severe Delay Prediction
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# 1. SYNTHETIC TRANSIT DATASET
#
#    Each row represents one historical bus trip.
#    Class 1 (~5%)  = Severe Delay
#    Class 0 (~95%) = On Time / Minor Delay
#
#    Informative features (4):
#      - Passenger_Load_Factor
#      - Historical_Congestion_Index
#      - Weather_Severity_Score
#      - Scheduled_Headway_Variance
#
#    Noise / shortcut features (21 remaining):
#      - Driver_ID_Hash, random sensor artifacts, etc.
#      These have zero true predictive power but give an
#      overfit model spurious "shortcuts" to memorise.
# ============================================================

N_SAMPLES     = 5_000
N_FEATURES    = 25
N_INFORMATIVE = 4
N_REDUNDANT   = 2   # linear combos of informative features
                     # remaining 19 are pure noise

X, y = make_classification(
    n_samples            = N_SAMPLES,
    n_features           = N_FEATURES,
    n_informative        = N_INFORMATIVE,
    n_redundant          = N_REDUNDANT,
    n_repeated           = 0,
    n_clusters_per_class = 1,
    weights              = [0.95, 0.05],   # 95% on-time, 5% severe delay
    flip_y               = 0.01,           # small label noise for realism
    random_state         = 42,
)

# Assign human-readable feature names for context
informative_names = [
    "Passenger_Load_Factor",
    "Historical_Congestion_Index",
    "Weather_Severity_Score",
    "Scheduled_Headway_Variance",
]
redundant_names = ["Redundant_Congestion_Combo", "Redundant_Load_Combo"]
noise_names     = [f"Noise_Sensor_{i:02d}" for i in range(1, N_FEATURES - N_INFORMATIVE - N_REDUNDANT + 1)]
# noise_names includes Driver_ID_Hash as the first entry to match domain story
noise_names[0]  = "Driver_ID_Hash"
feature_names   = informative_names + redundant_names + noise_names

print("=" * 60)
print("  TRANSIT DATASET SUMMARY")
print("=" * 60)
print(f"  Total trips      : {N_SAMPLES:,}")
print(f"  Features         : {N_FEATURES}  "
      f"({N_INFORMATIVE} informative, {N_REDUNDANT} redundant, "
      f"{len(noise_names)} noise)")
print(f"  On-Time trips    : {np.sum(y == 0):,}  "
      f"({np.sum(y == 0)/N_SAMPLES*100:.1f} %)")
print(f"  Severe Delays    : {np.sum(y == 1):,}  "
      f"({np.sum(y == 1)/N_SAMPLES*100:.1f} %)")
print("=" * 60)

# ============================================================
# 2. STRATIFIED TRAIN / TEST SPLIT
#    Stratify preserves the 95/5 imbalance in both partitions.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.25,   # 3,750 train | 1,250 test
    stratify     = y,
    random_state = 42,
)

# Standard-scale for Logistic Regression (XGBoost is scale-invariant)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n  Training trips   : {len(y_train):,}")
print(f"  Test trips       : {len(y_test):,}")

# ============================================================
# 3. INTENTIONALLY OVERFITTED XGBClassifier
#
#    Why each hyperparameter causes overfitting:
#      max_depth=18      → trees deep enough to memorise every
#                          training trip, including Driver_ID_Hash
#      n_estimators=2500 → far too many boosting rounds with no
#                          stopping criterion
#      learning_rate=0.3 → large step size; model overcommits
#                          aggressively at each round
#      subsample=1.0     → sees ALL training rows per tree,
#                          amplifying memorisation
#      colsample_bytree=1.0 → uses ALL 25 features per split,
#                          giving noise features full access
#      reg_alpha=0       → no L1 regularisation penalty
#      reg_lambda=0      → no L2 regularisation penalty
#      min_child_weight=1 → allows leaf nodes with a single trip
#      gamma=0           → no minimum loss-reduction threshold
#                          for making a split
#    NO early_stopping_rounds → all 2,500 rounds always execute
# ============================================================

xgb_overfit = XGBClassifier(
    n_estimators     = 2_500,
    max_depth        = 18,
    learning_rate    = 0.3,
    subsample        = 1.0,
    colsample_bytree = 1.0,
    reg_alpha        = 0,
    reg_lambda       = 0,
    min_child_weight = 1,
    gamma            = 0,
    scale_pos_weight = 1,        # deliberately ignores class imbalance
    eval_metric      = "logloss",
    random_state     = 42,
    verbosity        = 0,
)

print("\n  Training overfit XGBoost (2,500 rounds, depth 18) ...")
xgb_overfit.fit(X_train, y_train)   # NO early_stopping_rounds
print("  Done.")

# ============================================================
# 4. BASELINE: Default LogisticRegression
# ============================================================

print("  Training baseline Logistic Regression ...")
log_reg = LogisticRegression(max_iter=1_000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
print("  Done.\n")

# ============================================================
# 5. METRICS
# ============================================================

xgb_train_acc = accuracy_score(y_train, xgb_overfit.predict(X_train))
xgb_test_acc  = accuracy_score(y_test,  xgb_overfit.predict(X_test))
lr_train_acc  = accuracy_score(y_train, log_reg.predict(X_train_scaled))
lr_test_acc   = accuracy_score(y_test,  log_reg.predict(X_test_scaled))

xgb_gap = xgb_train_acc - xgb_test_acc
lr_gap  = lr_train_acc  - lr_test_acc

print("=" * 60)
print("  ACCURACY SUMMARY")
print("=" * 60)
print(f"  {'Model':<22} {'Train':>8} {'Test':>8} {'Gap':>8}")
print(f"  {'-'*50}")
print(f"  {'XGBoost (Overfit)':<22} {xgb_train_acc:>8.4f} "
      f"{xgb_test_acc:>8.4f} {xgb_gap:>8.4f}  ← memorised noise")
print(f"  {'Logistic Regression':<22} {lr_train_acc:>8.4f} "
      f"{lr_test_acc:>8.4f} {lr_gap:>8.4f}")
print("=" * 60)

# Predicted probabilities for calibration curves
xgb_prob_test = xgb_overfit.predict_proba(X_test)[:, 1]
lr_prob_test  = log_reg.predict_proba(X_test_scaled)[:, 1]

frac_pos_xgb, mean_pred_xgb = calibration_curve(
    y_test, xgb_prob_test, n_bins=10, strategy="uniform"
)
frac_pos_lr, mean_pred_lr = calibration_curve(
    y_test, lr_prob_test, n_bins=10, strategy="uniform"
)

# ============================================================
# 6. TWO-PANEL VISUALISATION
# ============================================================

DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"
GRID_COL = "#2e3246"
TEXT_COL = "#e8eaf6"
XGB_COL  = "#ef5350"   # red  — overfit model
LR_COL   = "#42a5f5"   # blue — logistic regression baseline
GAP_COL  = "#ffca28"   # amber — gap annotation
PERF_COL = "#66bb6a"   # green — perfect calibration line

fig, axes = plt.subplots(
    1, 2,
    figsize     = (17, 7.5),
    facecolor   = DARK_BG,
    gridspec_kw = {"wspace": 0.38},
)

fig.suptitle(
    "Severe XGBoost Overfitting — Urban Bus Delay Prediction\n"
    "5,000 Historical Transit Trips  ·  95% On-Time / 5% Severe Delay  "
    "·  25 Features (4 Informative + 21 Noise incl. Driver ID Hash)",
    fontsize   = 12.5,
    fontweight = "bold",
    color      = TEXT_COL,
    y          = 1.03,
)

# ----------------------------------------------------------
# LEFT PANEL — Generalisation Gap Bar Chart
# ----------------------------------------------------------
ax1 = axes[0]
ax1.set_facecolor(PANEL_BG)

x     = np.array([0, 1])
width = 0.28

bars_train = ax1.bar(
    x - width / 2,
    [lr_train_acc, xgb_train_acc],
    width,
    color   = [LR_COL, XGB_COL],
    alpha   = 0.88,
    zorder  = 3,
    label   = "Training Accuracy",
)
bars_test = ax1.bar(
    x + width / 2,
    [lr_test_acc, xgb_test_acc],
    width,
    color     = [LR_COL, XGB_COL],
    alpha     = 0.38,
    hatch     = "//",
    edgecolor = "white",
    linewidth = 0.6,
    zorder    = 3,
    label     = "Test Accuracy (Unseen Trips)",
)

# Value labels on every bar
for bar in list(bars_train) + list(bars_test):
    h = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.003,
        f"{h:.3f}",
        ha         = "center",
        va         = "bottom",
        fontsize   = 9,
        color      = TEXT_COL,
        fontweight = "bold",
    )

# Double-headed arrow annotating the XGBoost gap
bracket_x = (x[1] + width / 2) + 0.18
ax1.annotate(
    "",
    xy         = (bracket_x, xgb_test_acc),
    xytext     = (bracket_x, xgb_train_acc),
    arrowprops = dict(arrowstyle="<->", color=GAP_COL, lw=2.4),
)
ax1.text(
    bracket_x + 0.04,
    (xgb_train_acc + xgb_test_acc) / 2,
    f"Overfit\nGAP\n{xgb_gap:.3f}",
    color      = GAP_COL,
    fontsize   = 9,
    fontweight = "bold",
    va         = "center",
)

ax1.set_xticks(x)
ax1.set_xticklabels(
    ["Logistic Regression\n(Baseline)", "XGBoost\n(Overfit — No Reg.)"],
    color    = TEXT_COL,
    fontsize = 10.5,
)
ax1.set_ylabel("Bus Delay Prediction Accuracy", color=TEXT_COL, fontsize=11)
ax1.set_ylim(0.82, 1.06)
ax1.set_title(
    "Generalisation Gap\nTraining Trips vs. Unseen Test Trips",
    color      = TEXT_COL,
    fontsize   = 12,
    fontweight = "bold",
    pad        = 12,
)
ax1.tick_params(colors=TEXT_COL)
ax1.spines[:].set_color(GRID_COL)
ax1.yaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.45, zorder=0)
ax1.set_axisbelow(True)

solid_patch   = mpatches.Patch(color="grey", alpha=0.88,
                                label="Training Accuracy")
hatched_patch = mpatches.Patch(facecolor="grey", alpha=0.38,
                                hatch="//", edgecolor="white",
                                label="Test Accuracy (Unseen Trips)")
ax1.legend(
    handles   = [solid_patch, hatched_patch],
    loc       = "lower right",
    fontsize  = 9,
    facecolor = PANEL_BG,
    edgecolor = GRID_COL,
    labelcolor= TEXT_COL,
)

# Explanatory footnote
ax1.text(
    0.5, -0.13,
    "XGBoost memorised Driver_ID_Hash & sensor noise on training trips.\n"
    "Performance collapses on unseen routes it has never encountered.",
    transform  = ax1.transAxes,
    ha         = "center",
    fontsize   = 8.2,
    color      = "#9e9e9e",
    style      = "italic",
)

# ----------------------------------------------------------
# RIGHT PANEL — Transit Propensity / Calibration Curve
# ----------------------------------------------------------
ax2 = axes[1]
ax2.set_facecolor(PANEL_BG)

# Perfect calibration reference diagonal
ax2.plot(
    [0, 1], [0, 1],
    linestyle = "--",
    color     = PERF_COL,
    linewidth = 2,
    label     = "Perfect Calibration (Ideal)",
    zorder    = 2,
)

# Logistic Regression calibration
ax2.plot(
    mean_pred_lr, frac_pos_lr,
    marker    = "s",
    color     = LR_COL,
    linewidth = 2,
    markersize= 7,
    label     = "Logistic Regression (Baseline)",
    zorder    = 3,
)

# Overfit XGBoost calibration
ax2.plot(
    mean_pred_xgb, frac_pos_xgb,
    marker    = "o",
    linestyle = "-.",
    color     = XGB_COL,
    linewidth = 2.5,
    markersize= 8,
    label     = "XGBoost (Overfit) — Distorted Propensity",
    zorder    = 4,
)

# Shade miscalibration region for XGBoost
ax2.fill_between(
    mean_pred_xgb,
    mean_pred_xgb,
    frac_pos_xgb,
    alpha = 0.15,
    color = XGB_COL,
    label = "XGBoost Calibration Error Region",
)

ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel(
    "Predicted Severe Delay Probability (Model Output)",
    color    = TEXT_COL,
    fontsize = 11,
)
ax2.set_ylabel(
    "Observed Severe Delay Rate (Ground Truth)",
    color    = TEXT_COL,
    fontsize = 11,
)
ax2.set_title(
    "Transit Propensity Distortion\n"
    "Reliability Diagram — Severe Bus Delay Probability Estimates",
    color      = TEXT_COL,
    fontsize   = 12,
    fontweight = "bold",
    pad        = 12,
)
ax2.tick_params(colors=TEXT_COL)
ax2.spines[:].set_color(GRID_COL)
ax2.xaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.45, zorder=0)
ax2.yaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.45, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(
    fontsize  = 8.8,
    loc       = "upper left",
    facecolor = PANEL_BG,
    edgecolor = GRID_COL,
    labelcolor= TEXT_COL,
)

# Callout annotation on worst distortion point
worst_idx = int(np.argmax(np.abs(frac_pos_xgb - mean_pred_xgb)))
ax2.annotate(
    "XGBoost over-promises\nsevere delay risk here.\nUnsafe for transit ops.",
    xy         = (mean_pred_xgb[worst_idx], frac_pos_xgb[worst_idx]),
    xytext     = (0.42, 0.08),
    fontsize   = 8.5,
    color      = GAP_COL,
    fontweight = "bold",
    arrowprops = dict(
        arrowstyle      = "->",
        color           = GAP_COL,
        lw              = 1.8,
        connectionstyle = "arc3,rad=0.35",
    ),
)

# Explanatory footnote
ax2.text(
    0.5, -0.13,
    "A perfectly calibrated model predicts 40% severe delay → 40% of those trips\n"
    "are truly delayed. XGBoost's estimates cluster near 0 or 1 — dangerously overconfident.",
    transform  = ax2.transAxes,
    ha         = "center",
    fontsize   = 8.2,
    color      = "#9e9e9e",
    style      = "italic",
)

plt.tight_layout()
plt.savefig(
    "images/transit_xgboost_overfitting.png",
    dpi         = 150,
    bbox_inches = "tight",
    facecolor   = DARK_BG,
)
plt.show()
print("\nFigure saved → images/transit_xgboost_overfitting.png")
```

### Generated Chart

![Severe XGBoost Overfitting — Urban Bus Delay Prediction](https://raw.githubusercontent.com/luciantong/foretransit/main/images/transit_xgboost_overfitting.png)

---

## Academic references

- Chen, A. et al. (2007). *Ordered probit model for transit delay severity prediction.*
- Trépanier, M. et al. *Commercial speed and occupancy rate as delay predictors.*
