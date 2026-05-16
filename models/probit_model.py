import pandas as pd
import numpy as np
from datetime import datetime

# ─── Probit Model (Chen et al. 2007) ─────────
# Uses Chen et al. beta coefficients directly
# as a baseline before retraining on Toronto data
# β values from Table 4 of Chen et al. paper

CHEN_BETAS = {
    "cumulative_dwell_time": 0.2367,   # highest impact
    "cumulative_leg_time":   0.1862,
    "cumulative_stops":      0.0989,
    "is_sunday":            -0.7685,   # Sunday least delayed
    "intercept":            -0.2732
}

THRESHOLDS = {
    "mu1": 0.5,
    "mu2": 1.2,
    "mu3": 2.1
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def probit_predict(features):
    """
    Predicts delay severity using Chen et al. coefficients
    Returns probabilities for each severity level
    P(Y=0), P(Y=1), P(Y=2), P(Y=3)
    """
    # Calculate latent variable y*
    y_star = (
        CHEN_BETAS["intercept"] +
        CHEN_BETAS["cumulative_dwell_time"] * 
            features.get("cumulative_dwell_time", 0) +
        CHEN_BETAS["cumulative_leg_time"] * 
            features.get("cumulative_leg_time", 0) +
        CHEN_BETAS["cumulative_stops"] * 
            features.get("cumulative_stops", 0) +
        CHEN_BETAS["is_sunday"] * 
            features.get("is_sunday", 0)
    )

    # Calculate probabilities using normal CDF
    from scipy.stats import norm
    p0 = norm.cdf(THRESHOLDS["mu1"] - y_star)
    p1 = norm.cdf(THRESHOLDS["mu2"] - y_star) - p0
    p2 = norm.cdf(THRESHOLDS["mu3"] - y_star) - p0 - p1
    p3 = 1 - p0 - p1 - p2

    # Predicted class = highest probability
    probs     = [p0, p1, p2, p3]
    predicted = int(np.argmax(probs))

    return {
        "model":        "probit",
        "predicted":    predicted,
        "probabilities": {
            "on_time":   round(p0, 4),
            "minor":     round(p1, 4),
            "moderate":  round(p2, 4),
            "severe":    round(p3, 4)
        },
        "y_star":       round(y_star, 4)
    }

def calc_log_loss(predicted_probs, actual):
    """
    Log loss for model comparison in MAGI
    Lower = better
    """
    import math
    prob = predicted_probs[actual]
    return round(-math.log(prob + 1e-10), 4)