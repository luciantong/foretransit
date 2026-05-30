import joblib
import pandas as pd
import os
import pickle
import numpy as np

def diagnose_xgboost():
    # Corrected path to match the output of train_script.py
    # We load the 'bus' mode model as a representative diagnostic
    model_path = "models/saved/bus_xgb.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}")
        print("Please run 'python3 train_script.py' first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    df = pd.read_csv("data/processed/bus_train.csv")
    
    # These features MUST match the FEATURES list in train_script.py in exact order
    features = ["cumulative_dwell_time", "cumulative_leg_time", "cumulative_stops", 
                "day_of_week", "section_id", "hour_of_day", "is_sunday", "route_type"]
    
    sample_stops = df['stop_id'].unique()[:5]
    
    print(f"{'Stop ID':<15} | {'Reliability %':<15}")
    print("-" * 35)
    
    for stop_id in sample_stops:
        # Get the row data
        row = df[df['stop_id'] == str(stop_id)].iloc[0:1]
        
        # Build the input vector manually to ensure all 8 features exist
        # If a feature is missing in the CSV, we fill with 0
        input_data = pd.DataFrame(0, index=[0], columns=features)
        for col in features:
            if col in row.columns:
                input_data[col] = row[col].values[0]
        
        # XGBoost prediction
        # predict_proba returns [on_time, minor, moderate, severe]
        probs = model.predict_proba(input_data)[0]
        # Reliability defined as the probability of being 'on_time' (class 0)
        reliability = round(probs[0] * 100)
        
        print(f"{str(stop_id):<15} | {reliability:<15}")

if __name__ == "__main__":
    diagnose_xgboost()