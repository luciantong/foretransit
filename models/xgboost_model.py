import pandas as pd
import numpy as np
import os

# ─── XGBoost Model ───────────────────────────
# Trained on Toronto GTFS data
# Uses all three feature tiers
# Separate models per mode (bus, streetcar, subway)

try:
    from xgboost import XGBClassifier
    import pickle
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

# ─── Feature Tiers ───────────────────────────

# TIER 1 — Chen et al. proven variables
TIER_1 = [
    "cumulative_dwell_time",   # β=0.237 most important
    "cumulative_leg_time",     # β=0.186
    "cumulative_stops",        # β=0.099
    "day_of_week",
    "section_id",
    "rain_mm",
    "wind_speed",
    "visibility_km"
]

# TIER 2 — Trépanier et al. additions
TIER_2 = [
    "commercial_speed",
    "occupancy_rate",
    "schedule_adherence",
    "hour_of_day",
    "is_peak_hour"
]

# TIER 3 — Toronto specific
TIER_3 = [
    "temperature_c",
    "snow_mm",
    "is_holiday",
    "is_special_event",
    "delay_trend",
    "route_historical_avg"
]

ALL_FEATURES = TIER_1 + TIER_2 + TIER_3

# ─── Load or Create Model ────────────────────
def load_model(mode="bus"):
    """
    Load pre-trained model if exists
    Otherwise return untrained model
    Separate model per mode: bus, streetcar, subway
    """
    path = f"models/saved/{mode}_xgb.pkl"

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Return fresh untrained model
    return XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        objective        = "multi:softprob",
        num_class        = 4,
        eval_metric      = "mlogloss"
    )

def xgboost_predict(features, mode="bus"):
    """
    Predicts delay severity using XGBoost
    Returns probabilities for each severity level
    """
    if not XGBOOST_AVAILABLE:
        return None

    model = load_model(mode)

    # ── Must match FEATURES list in train_xgboost.py exactly ──
    try:
        X = pd.DataFrame([{
            "cumulative_dwell_time": features.get("cumulative_dwell_time", 0),
            "cumulative_leg_time":   features.get("cumulative_leg_time", 0),
            "cumulative_stops":      features.get("cumulative_stops", 0),
            "day_of_week":           features.get("day_of_week", 0),
            "section_id":            features.get("section_id", 0),
            "hour_of_day":           features.get("hour_of_day", 12),
            "is_sunday":             features.get("is_sunday", 0),
            "route_type":            {"bus": 3, "streetcar": 0, "subway": 1}.get(
                                         features.get("mode", "bus"), 3)
        }])

        probs     = model.predict_proba(X)[0]
        predicted = int(np.argmax(probs))

        return {
            "model":     "xgboost",
            "predicted": predicted,
            "probabilities": {
                "on_time":  round(float(probs[0]), 4),
                "minor":    round(float(probs[1]), 4),
                "moderate": round(float(probs[2]), 4),
                "severe":   round(float(probs[3]), 4)
            }
        }
    except Exception as e:
        return None

def calc_log_loss(predicted_probs, actual):
    import math
    prob = predicted_probs[actual]
    return round(-math.log(prob + 1e-10), 4)