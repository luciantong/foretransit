import pandas as pd
import joblib
import os
import statsmodels.api as sm # Add this
from models.probit_model import ProbitModel
from datetime import datetime

def train_and_save_probit():
    data_path = "data/processed/bus_train.csv" 
    if not os.path.exists(data_path):
        print(f"Error: Could not find training data at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Use a subset of features that are definitely not perfectly correlated
    features = ['cumulative_dwell_time', 'cumulative_leg_time', 'cumulative_stops']
    
    X = df[features]
    
    # CRITICAL: Add a constant (intercept) to X. 
    # This often solves the 'Singular Matrix' error by providing a baseline.
    X = sm.add_constant(X)
    
    y = (df['delay_severity_category'] > 0).astype(int)
    
    print("Training Probit model with added constant...")
    model = ProbitModel()
    model.fit(X, y)
    
    model_artifact = {
        "params": model.params,
        "features": features, # Note: original features list
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs("models/artifacts", exist_ok=True)
    joblib.dump(model_artifact, "models/artifacts/probit_model.joblib")
    
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_probit()