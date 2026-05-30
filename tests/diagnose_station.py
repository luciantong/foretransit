import pandas as pd
import os

def diagnose():
    csv_path = "data/processed/bus_train.csv"
    
    print(f"--- DIAGNOSING: {csv_path} ---")
    
    if not os.path.exists(csv_path):
        print("❌ ERROR: File does not exist!")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ File loaded. Total rows: {len(df)}")
    
    # Check if 'stop_id' exists (you used it in api/station_forecast.py)
    if 'stop_id' not in df.columns:
        print("❌ ERROR: 'stop_id' column MISSING in CSV. Columns found:", df.columns.tolist())
        return
    else:
        print("✅ 'stop_id' column found.")
        print(f"🔍 Unique stops available: {df['stop_id'].nunique()}")
        print(f"   First 5 stops: {df['stop_id'].unique()[:5].tolist()}")

    # Test a specific lookup
    test_stop = df['stop_id'].iloc[0]
    print(f"\n🧪 Testing lookup for stop: {test_stop}")
    
    match = df[df['stop_id'].astype(str) == str(test_stop)]
    if not match.empty:
        print("✅ Lookup successful! Data snippet:")
        print(match.head(1))
    else:
        print("❌ Lookup failed! Data structure mismatch.")

if __name__ == "__main__":
    diagnose()