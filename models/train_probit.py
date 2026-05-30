import pandas as pd
import joblib
import os
import statsmodels.api as sm
from models.probit_model import ProbitModel
from datetime import datetime

def train_and_save_probit():
    data_path = "data/processed/bus_train.csv" 
    if not os.path.exists(data_path):
        print(f"Error: Could not find training data at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # FIX: Use only one workload feature (cumulative_stops) to prevent collinearity
    # and include hour_of_day for temporal variance.
    features = ['cumulative_stops', 'hour_of_day']
    
    X = df[features]
    
    # Add constant for the intercept
    X = sm.add_constant(X)
    
    # Ensure y is binary
    y = (df['delay_severity_category'] > 0).astype(int)
    
    print(f"Training Probit model with features: {features}...")
    model = ProbitModel()
    model.fit(X, y)
    
    model_artifact = {
        "params": model.params, # This saves the trained results
        "features": features,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs("models/artifacts", exist_ok=True)
    joblib.dump(model_artifact, "models/artifacts/probit_model.joblib")
    
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_probit()