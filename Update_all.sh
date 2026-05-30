#!/bin/bash
# This script runs the entire data-to-prediction pipeline in order.

echo "--- Data Pipeline Started: $(date) ---"

# 1. Fetch raw data
python3 -m pipeline.fetch_gtfs && \
python3 -m pipeline.fetch_historical_delays && \

# 2. Process features
python3 -m pipeline.build_features && \

# 3. Train models (Probit and XGBoost)
python3 -m models.train_probit && \
python3 -m models.train_xgboost

echo "--- Data Pipeline Finished: $(date) ---"