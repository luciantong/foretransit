"""
data/weather.py
Fetches or generates hourly weather data for Toronto to align with GTFS timelines.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_toronto_weather_logs():
    print("🌤️  Initializing Weather Data Engine for Toronto...")
    output_dir = "data/weather"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/hourly_weather.csv"
    
    # Generate 48 hours of lookback data up to right now (May 2026)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=2)
    
    timestamps = pd.date_range(start=start_time, end=end_time, freq='h')
    
    np.random.seed(42)
    records = []
    
    for ts in timestamps:
        hour = ts.hour
        # Simulate realistic late-spring temperature fluctuations in Toronto (cooler at night, warmer in afternoon)
        base_temp = 12.0 if (11 <= hour <= 17) else 6.0
        temp = base_temp + np.random.uniform(-3.0, 4.0)
        
        # Simulate occasional rain blocks (e.g., 15% chance of localized showers)
        is_raining = np.random.rand() > 0.85
        rain = np.random.uniform(0.5, 4.5) if is_raining else 0.0
        
        # Spring wind speeds in Toronto (km/h)
        wind = np.random.uniform(10.0, 35.0)
        if is_raining:
            wind += np.random.uniform(5.0, 15.0)
            
        # Visibility drop during weather events
        visibility = np.random.uniform(2.0, 5.0) if is_raining else np.random.uniform(10.0, 16.0)
        
        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature_c": round(temp, 1),
            "rain_mm": round(rain, 2),
            "snow_mm": 0.0, # Late May in Toronto
            "wind_speed": round(wind, 1),
            "visibility_km": round(visibility, 1)
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"   ↳ 💾 Successfully created {len(df)} hourly weather matrix points at {output_path}")

if __name__ == "__main__":
    generate_toronto_weather_logs()