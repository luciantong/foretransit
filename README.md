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

Here are Python codes used for generating graphs in the report




## Academic references

- Chen, A. et al. (2007). *Ordered probit model for transit delay severity prediction.*
- Trépanier, M. et al. *Commercial speed and occupancy rate as delay predictors.*
