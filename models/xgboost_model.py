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
# Replace ALL_FEATURES = TIER_1 + TIER_2 + TIER_3 with:
ALL_FEATURES = [
    "cumulative_dwell_time",
    "cumulative_leg_time", 
    "cumulative_stops",
    "day_of_week",
    "section_id",
    "hour_of_day",
    "is_sunday",
    "route_type"
]

# ─── Load or Create Model ────────────────────
def load_model(mode="bus"):
    path = f"models/saved/{mode}_xgb.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
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
    Predicts delay severity using XGBoost.
    Ensures the inference DataFrame structure aligns perfectly with ALL_FEATURES.
    """
    if not XGBOOST_AVAILABLE:
        return None

    try:
        model = load_model(mode)

        # Build full-tier dictionary with smart fallback values
        aligned_features = {}
        for key in ALL_FEATURES:
            if "mm" in key or "speed" in key or "time" in key or "rate" in key or "trend" in key or "avg" in key:
                default_val = 0.0
            elif "is_" in key:
                default_val = 0
            elif "hour" in key:
                default_val = 12
            elif "visibility" in key:
                default_val = 10.0
            elif "temperature" in key:
                default_val = 15.0
            else:
                default_val = 0

            aligned_features[key] = features.get(key, default_val)

        # Build single-row DataFrame matching column order exactly
        X = pd.DataFrame([aligned_features], columns=ALL_FEATURES)

        # Run inference
        try:
            probs        = model.predict_proba(X)[0]
            predicted    = int(np.argmax(probs))
            inference_ok = True
        except Exception:
            # Uniform distribution — no false confidence when model fails
            probs        = [0.25, 0.25, 0.25, 0.25]
            predicted    = 0
            inference_ok = False

        return {
            "model":         "xgboost",
            "predicted":     predicted,
            "inference_ok":  inference_ok,
            "probabilities": {
                "on_time":  round(float(probs[0]), 4),
                "minor":    round(float(probs[1]), 4),
                "moderate": round(float(probs[2]), 4),
                "severe":   round(float(probs[3]), 4)
            }
        }
    except Exception as e:
        print(f"XGBoost Prediction Failure: {str(e)}")
        return None

def calc_log_loss(predicted_probs, actual):
    import math
    prob = predicted_probs[actual]
    return round(-math.log(prob + 1e-10), 4)