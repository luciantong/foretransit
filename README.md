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

## Setup

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

1. Go to https://www.ttc.ca/open-data
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

### 5. Fetch live data

```bash
python pipeline/fetch_gtfs.py
```

### 6. Train XGBoost models (one time only)

```bash
python -m models.train_xgboost
```

Saves `bus_xgb.pkl`, `streetcar_xgb.pkl`, `subway_xgb.pkl` to `models/saved/`. Only needs to be re-run if you want to retrain on new data.

---

## Running the server

### Development

```bash
python pipeline/fetch_gtfs.py    # fetch fresh data first
uvicorn api.main:app --reload --port 8000
```

### Production

```bash
uvicorn api.main:app --port 8000
```

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

```python
"""
TTC System Overview 2024 - Horizontal Bar Chart
Supports the argument that Toronto's transit delay context requires locally-calibrated models
due to scale, multimodal complexity, and surface-subway interdependence.
"""

import matplotlib.pyplot as plt
import matplotlib.patches mpatches
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

# Saved directly to outputs directory
plt.savefig('outputs/ttc_system_overview_chart.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
```

### Generated Output
![TTC System Overview](outputs/ttc_system_overview_chart.png)

---

## Academic references

- Chen, A. et al. (2007). *Ordered probit model for transit delay severity prediction.*
- Trépanier, M. et al. *Commercial speed and occupancy rate as delay predictors.*
