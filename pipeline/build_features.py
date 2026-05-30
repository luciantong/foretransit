"""
pipeline/build_features.py (Path-Resilient & Optimized for Large Files)
Feature Engineering Pipeline for Foretransit
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

# Feature definitions
TIER_1 = ['cumulative_dwell_time', 'cumulative_leg_time', 'cumulative_stops', 'day_of_week', 'section_id', 'rain_mm', 'wind_speed', 'visibility_km']
TIER_2 = ['commercial_speed', 'occupancy_rate', 'schedule_adherence', 'hour_of_day', 'is_peak_hour']
TIER_3 = ['temperature_c', 'snow_mm', 'is_holiday', 'is_special_event', 'delay_trend', 'route_historical_avg']

TORONTO_HOLIDAYS_2026 = {"2026-01-01", "2026-02-16", "2026-04-03", "2026-05-18", "2026-07-01", "2026-08-03", "2026-09-07", "2026-10-12", "2026-12-25", "2026-12-26"}

def is_toronto_holiday(dt): return 1 if dt.strftime("%Y-%m-%d") in TORONTO_HOLIDAYS_2026 else 0

def is_toronto_game_day(dt):
    month, day_of_week, hour = dt.month, dt.weekday(), dt.hour
    is_season = (month >= 10) or (month <= 4)
    if not is_season: return 0
    if day_of_week in [2, 4, 5] and (18 <= hour <= 23): return 1
    if day_of_week == 6 and (13 <= hour <= 17): return 1
    return 0

def generate_simulated_raw_data():
    np.random.seed(42)
    records = []
    for trip_idx in range(100):
        mode = np.random.choice(['subway', 'streetcar', 'bus'])
        start_time = datetime(2026, 5, 29, np.random.randint(6, 22), np.random.randint(0, 59))
        cum_dist = 0.0
        for stop_seq in range(12):
            leg_distance = np.random.uniform(400, 900)
            cum_dist += leg_distance
            delay = np.random.exponential(scale=300) if np.random.rand() > 0.4 else 0.0
            actual_arrival = start_time + pd.Timedelta(seconds=int(leg_distance/12) + int(delay))
            records.append({
                'trip_id': f"TRIP_{1000+trip_idx}", 'stop_id': f"STOP_{100+stop_seq}", 'stop_sequence': stop_seq,
                'route_id': f"ROUTE_{mode.upper()}", 'arrival_time': actual_arrival.strftime("%H:%M:%S"),
                'departure_time': (actual_arrival + pd.Timedelta(seconds=30)).strftime("%H:%M:%S"),
                'shape_dist_traveled': cum_dist, 'timestamp': actual_arrival,
                'route_type': 1 if mode == 'subway' else (0 if mode == 'streetcar' else 3)
            })
    return pd.DataFrame(records)

def build_features():
    print("🚀 Starting Feature Engineering Pipeline...")
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    df = generate_simulated_raw_data()
    
    # 2. Transformations
    df = df.sort_values(by=["trip_id", "stop_sequence"])
    
    # Ensure stop_id is string and preserved
    df['stop_id'] = df['stop_id'].astype(str)
    
    df['cumulative_stops'] = df.groupby('trip_id').cumcount()
    df['cumulative_dwell_time'] = df['cumulative_stops'] * 30
    df['cumulative_leg_time'] = df['cumulative_stops'] * 120
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['section_id'] = df['trip_id'].astype("category").cat.codes
    
    # Metrics
    df['commercial_speed'] = 25.0
    df['occupancy_rate'] = 0.5
    df['schedule_adherence'] = np.random.uniform(-5, 10, size=len(df))
    df['is_peak_hour'] = df['hour_of_day'].apply(lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 18) else 0)
    df['temperature_c'] = 20.0
    df['rain_mm'] = 0.0
    df['wind_speed'] = 10.0
    df['visibility_km'] = 10.0
    df['snow_mm'] = 0.0
    df['is_holiday'] = df['timestamp'].apply(is_toronto_holiday)
    df['is_special_event'] = df['timestamp'].apply(is_toronto_game_day)
    df['delay_trend'] = 0.0
    df['route_historical_avg'] = 0.0
    
    # Categorize
    df['delay_severity_category'] = pd.cut(df['schedule_adherence'], bins=[-np.inf, 5, 10, 15, np.inf], labels=[0, 1, 2, 3]).astype(int)
    
    # Map modes
    def type_to_mode(t): return 'subway' if t == 1 else ('streetcar' if t in [0, 5] else 'bus')
    df['mode'] = df['route_type'].apply(type_to_mode)
    
    # 3. Export
    all_features = TIER_1 + TIER_2 + TIER_3
    cols_to_export = ['stop_id'] + all_features + ['delay_severity_category', 'schedule_adherence']
    
    for mode in ['subway', 'streetcar', 'bus']:
        mode_df = df[df['mode'] == mode].copy()
        
        # Ensure stop_id exists and is not null
        mode_df['stop_id'] = mode_df['stop_id'].fillna("UNKNOWN")
        
        # Select only the explicitly defined columns and forcibly strip any unwanted artifacts
        clean_df = mode_df[cols_to_export].copy()
        
        # Explicit drop of any artifact that shouldn't be there
        if 'schedule_adherence.1' in clean_df.columns:
            clean_df = clean_df.drop(columns=['schedule_adherence.1'])
            
        clean_df = clean_df.dropna(subset=all_features + ['delay_severity_category'])
        
        if len(clean_df) >= 5:
            train_df, test_df = train_test_split(clean_df, test_size=0.20, random_state=42, stratify=clean_df['delay_severity_category'])
            train_df.to_csv(f"{output_dir}/{mode}_train.csv", index=False)
            test_df.to_csv(f"{output_dir}/{mode}_test.csv", index=False)
            print(f"   ↳ {mode.upper()}: Created {len(train_df)} train records. Schema sanitized.")

if __name__ == "__main__":
    build_features()