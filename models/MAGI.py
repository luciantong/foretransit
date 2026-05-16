#Referenced from EVA
#Multi-Answer Geographical Informations (MAGI) - Ensemble of Probit, XGBoost, and Jayden Method

import math
from pipeline.utils import (
    classify_delay,
    severity_label,
    severity_color
)
from models.jayden_model import (
    jayden_method,
)

from models.probit_model import (
    probit_predict,
    calc_log_loss as probit_log_loss
)
from models.xgboost_model import (
    xgboost_predict,
    calc_log_loss as xgb_log_loss
)

# ─── MAGI ────────────────────────────────────
# Multi-Answer Geographical Informations
# Runs all three models and picks the best one
# Best = lowest log loss

def run_magi(features, actual_severity=None):
    """
    Runs Probit, XGBoost, and Jayden Method
    Picks winner by log loss
    Returns final forecast

    features = dict of all input variables
    actual_severity = 0,1,2,3 if known (for training)
                      None if predicting live
    """

    results = {}

    # ── Model 1: Probit (Chen et al.) ────────
    probit = probit_predict(features)
    if probit:
        results["probit"] = probit

    # ── Model 2: XGBoost ─────────────────────
    mode    = features.get("mode", "bus")
    xgboost = xgboost_predict(features, mode)
    if xgboost:
        results["xgboost"] = xgboost

    # ── Model 3: Jayden Method ───────────────
    delay_seconds = features.get("delay_seconds", 0)
    gap_seconds   = features.get("gap_seconds", 0)
    jayden_score  = jayden_method(delay_seconds, gap_seconds)
    jayden_severity = classify_delay(delay_seconds)

    # Convert score to fake probabilities for log loss
    probs = [0.1, 0.1, 0.1, 0.1]
    probs[jayden_severity] = 0.7

    results["jayden"] = {
        "model":     "jayden",
        "predicted": jayden_severity,
        "score":     jayden_score,
        "probabilities": {
            "on_time":  probs[0],
            "minor":    probs[1],
            "moderate": probs[2],
            "severe":   probs[3]
        }
    }

    # ── Pick Winner ──────────────────────────
    if actual_severity is not None:
        # We know actual → calculate real log loss
        log_losses = {}

        for name, result in results.items():
            prob_list = list(
                result["probabilities"].values())
            ll = -math.log(
                prob_list[actual_severity] + 1e-10)
            log_losses[name] = round(ll, 4)
            result["log_loss"] = round(ll, 4)

        winner = min(log_losses, key=log_losses.get)

    else:
        # No actual known → use confidence (highest prob)
        confidences = {}
        for name, result in results.items():
            max_prob = max(
                result["probabilities"].values())
            confidences[name] = max_prob

        winner = max(confidences, key=confidences.get)

    # ── Build Final Output ───────────────────
    winning_result = results[winner]
    severity       = winning_result["predicted"]

    return {
        "model_used":   winner,
        "predicted":    severity,
        "label":        severity_label(severity),
        "color":        severity_color(severity),

        "all_models":   results,

        "model_info": {
            "winner":      winner,
            "method":      "MAGI - Multi-Answer Geographical Informations",
            "reference":   "Chen et al. 2007"
        }
    }