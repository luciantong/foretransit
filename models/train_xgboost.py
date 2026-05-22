import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report

# ─── Import from your utils ──────────────────
from pipeline.utils import (
    haversine,
    parse_gtfs_time,
    classify_delay,
    stops_df,
    stop_times_df,
    trips_df,
    routes_df,
    vehicles,
    capture_time
)

# ─── Step 1: Build Real Delay Records ────────
print("→ Building real delay records...")

real_records = []
now = datetime.now()

for v in vehicles:
    try:
        v_lat     = float(v["lat"])
        v_lon     = float(v["lon"])
        route_tag = v["routeTag"]
        speed     = float(v["speedKmHr"])

        # Bounding box first for speed
        nearby_stops = stops_df[
            (stops_df["stop_lat"].between(v_lat - 0.01, v_lat + 0.01)) &
            (stops_df["stop_lon"].between(v_lon - 0.01, v_lon + 0.01))
        ].copy()

        if nearby_stops.empty:
            continue

        nearby_stops["dist"] = nearby_stops.apply(
            lambda r: haversine(v_lat, v_lon,
                                r["stop_lat"],
                                r["stop_lon"]), axis=1)

        nearest = nearby_stops.loc[nearby_stops["dist"].idxmin()]
        stop_id = nearest["stop_id"]
        dist_km = nearest["dist"]

        if dist_km > 0.1:
            continue

        route_trips = trips_df[
            trips_df["route_id"].astype(str) == str(route_tag)
        ]["trip_id"].tolist()

        scheduled = stop_times_df[
            (stop_times_df["stop_id"] == stop_id) &
            (stop_times_df["trip_id"].isin(route_trips))
        ].copy()

        if scheduled.empty:
            continue

        scheduled["parsed_time"] = scheduled[
            "arrival_time"].apply(parse_gtfs_time)
        scheduled = scheduled.dropna(subset=["parsed_time"])
        scheduled["time_diff"] = abs(
            scheduled["parsed_time"] - now
        ).dt.total_seconds()

        closest        = scheduled.loc[
                            scheduled["time_diff"].idxmin()]
        scheduled_time = closest["parsed_time"]
        delay_seconds  = (now - scheduled_time).total_seconds()

        route_info = routes_df[
            routes_df["route_id"].astype(str) == str(route_tag)]
        route_type = int(route_info["route_type"].iloc[0]) \
                     if not route_info.empty else 3

        real_records.append({
            "cumulative_dwell_time": closest["stop_sequence"] * 0.3,
            "cumulative_leg_time":   closest.get("shape_dist_traveled", 0),
            "cumulative_stops":      closest["stop_sequence"],
            "day_of_week":           now.weekday(),
            "section_id":            hash(str(route_tag)) % 100,
            "hour_of_day":           now.hour,
            "is_sunday":             int(now.weekday() == 6),
            "route_type":            route_type,
            "delay_severity":        classify_delay(delay_seconds),
            "source":                "realtime"
        })

    except:
        continue

print(f"  ✓ {len(real_records)} real records built")

# ─── Step 2: Build Simulated Records ─────────
print("\n→ Building simulated records...")

trips_routes = trips_df.merge(routes_df, on="route_id", how="left")
df_static    = stop_times_df.merge(
                    trips_routes, on="trip_id", how="left")

def parse_hour(time_str):
    try:
        return int(str(time_str).split(":")[0]) % 24
    except:
        return 12

df_static["hour_of_day"] = df_static["arrival_time"].apply(parse_hour)
df_static = df_static.sort_values(["trip_id", "stop_sequence"])
df_static["cumulative_stops"] = df_static.groupby(
    "trip_id").cumcount() + 1
df_static["cumulative_leg_time"]   = df_static[
    "shape_dist_traveled"].fillna(0)
df_static["cumulative_dwell_time"] = df_static["cumulative_stops"] * 0.3
df_static["day_of_week"] = df_static["service_id"].apply(
    lambda x: 6 if "SUN" in str(x).upper()
    else 5 if "SAT" in str(x).upper()
    else np.random.randint(0, 5)
)
df_static["is_sunday"]  = (df_static["day_of_week"] == 6).astype(int)
df_static["section_id"] = df_static[
    "route_id"].astype("category").cat.codes

route_type_col = "route_type_x" \
    if "route_type_x" in df_static.columns \
    else "route_type"
df_static["route_type"] = df_static[route_type_col].fillna(3)

def simulate_delay(row):
    score = 0
    if 7  <= row["hour_of_day"] <= 9:    score += 2
    if 16 <= row["hour_of_day"] <= 18:   score += 1
    if row["cumulative_stops"] > 10:     score += 1
    if row["cumulative_stops"] > 20:     score += 1
    if row["cumulative_dwell_time"] > 8: score += 2
    if row["is_sunday"]:                 score -= 2
    score += np.random.normal(0, 0.5)
    if score <= 0:  return 0
    if score <= 2:  return 1
    if score <= 4:  return 2
    return 3

np.random.seed(42)
df_static["delay_severity"] = df_static.apply(simulate_delay, axis=1)
df_static["source"] = "simulated"

print(f"  ✓ {len(df_static)} simulated records built")

# ─── Step 3: Combine ─────────────────────────
print("\n→ Combining real + simulated...")

FEATURES = [
    "cumulative_dwell_time",  # Chen β=0.237
    "cumulative_leg_time",    # Chen β=0.186
    "cumulative_stops",       # Chen β=0.099
    "day_of_week",
    "section_id",
    "hour_of_day",
    "is_sunday",
    "route_type"
]

df_sim = df_static[FEATURES + ["delay_severity", "source"]].copy()

if len(real_records) > 0:
    df_real     = pd.DataFrame(real_records)
    df_combined = pd.concat(
                    [df_sim, df_real[FEATURES + ["delay_severity", "source"]]],
                    ignore_index=True)
    print(f"  ✓ Real:      {len(real_records)} rows")
    print(f"  ✓ Simulated: {len(df_sim)} rows")
    print(f"  ✓ Total:     {len(df_combined)} rows")
else:
    df_combined = df_sim
    print("  ⚠️ No real records — using simulated only")

df_combined = df_combined.dropna(subset=FEATURES + ["delay_severity"])

X = df_combined[FEATURES]
y = df_combined["delay_severity"].astype(int)

# ─── Step 4: Train/Test Split 80/20 ──────────
print("\n→ Splitting 80/20...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y
)

print(f"  ✓ Train: {len(X_train)} rows")
print(f"  ✓ Test:  {len(X_test)} rows")

# ─── Step 5: Train Per Mode ──────────────────
os.makedirs("models/saved", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def classify_mode(route_type):
    if route_type == 1: return "subway"
    if route_type == 0: return "streetcar"
    return "bus"

df_combined["mode"] = df_combined["route_type"].apply(classify_mode)

for mode in ["bus", "streetcar", "subway"]:
    print(f"\n→ Training {mode} model...")

    mode_mask_train = df_combined.loc[X_train.index, "mode"] == mode
    mode_mask_test  = df_combined.loc[X_test.index,  "mode"] == mode

    X_tr = X_train[mode_mask_train]
    y_tr = y_train[mode_mask_train]
    X_te = X_test[mode_mask_test]
    y_te = y_test[mode_mask_test]

    if len(X_tr) == 0:
        print(f"  ⚠️ No {mode} data, skipping")
        continue

    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        objective        = "multi:softprob",
        num_class        = 4,
        eval_metric      = "mlogloss",
        random_state     = 42
    )

    model.fit(
        X_tr, y_tr,
        eval_set = [(X_te, y_te)],
        verbose  = False
    )

    y_pred       = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)

    acc = accuracy_score(y_te, y_pred)
    ll  = log_loss(y_te, y_pred_proba)

    print(f"  ✓ Accuracy:  {acc:.4f}")
    print(f"  ✓ Log Loss:  {ll:.4f}")
    print(classification_report(
        y_te, y_pred,
        target_names=["on_time","minor","moderate","severe"],
        zero_division=0
    ))

    path = f"models/saved/{mode}_xgb.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved to {path}")

    # ── Feature Importance Plot ───────────────
    plt.figure(figsize=(10, 6))
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=True)

    colors = [
        "red" if f in [
            "cumulative_dwell_time",
            "cumulative_leg_time",
            "cumulative_stops"
        ] else "steelblue"
        for f in importance.index
    ]

    importance.plot(kind="barh", color=colors)
    plt.title(
        f"XGBoost Feature Importance — {mode.capitalize()}\n"
        f"Red = Chen et al. (2007) variables",
        fontsize=14
    )
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"outputs/{mode}_feature_importance.png")
    plt.show()
    print(f"  ✓ Plot saved to outputs/{mode}_feature_importance.png")

print("\n✓ All models trained and saved")
print("✓ MAGI will now use XGBoost automatically")