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

REQUIRED_FEATURES = {
    "probit": [
        "cumulative_dwell_time",
        "cumulative_leg_time",
        "cumulative_stops",
        "is_sunday",
    ],
    "xgboost": [
        "cumulative_dwell_time",
        "cumulative_leg_time",
        "cumulative_stops",
        "day_of_week",
        "section_id",
        "hour_of_day",
        "is_sunday",
        "mode",
    ],
    "jayden": [
        "delay_seconds",
        "gap_seconds",
    ],
}


def _is_feature_available(value):
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _missing_required_features(model_name, features):
    required = REQUIRED_FEATURES.get(model_name, [])
    return [key for key in required if not _is_feature_available(features.get(key))]


def _max_probability(probabilities):
    if not probabilities:
        return 0.0
    return float(max(probabilities.values()))


def _score_from_log_loss(log_loss):
    # Map lower log loss to higher confidence-like score in [0,100].
    scaled = 1 - min(max(log_loss, 0.0), 2.5) / 2.5
    return round(scaled * 100, 1)


def _feature_subset(features, keys):
    return {key: features.get(key) for key in keys}

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
    skipped_models = {}

    # ── Model 1: Probit (Chen et al.) ────────
    missing_probit = _missing_required_features("probit", features)
    if missing_probit:
        skipped_models["probit"] = {
            "reason": "missing_features",
            "missing": missing_probit,
        }
    else:
        probit = probit_predict(features)
        if probit:
            probit["inputs"] = _feature_subset(features, REQUIRED_FEATURES["probit"])
            results["probit"] = probit
        else:
            skipped_models["probit"] = {
                "reason": "model_failed",
                "missing": [],
            }

    # ── Model 2: XGBoost ─────────────────────
    missing_xgb = _missing_required_features("xgboost", features)
    if missing_xgb:
        skipped_models["xgboost"] = {
            "reason": "missing_features",
            "missing": missing_xgb,
        }
    else:
        mode = features.get("mode", "bus")
        xgboost = xgboost_predict(features, mode)
        if xgboost:
            xgboost["inputs"] = _feature_subset(features, REQUIRED_FEATURES["xgboost"])
            results["xgboost"] = xgboost
        else:
            skipped_models["xgboost"] = {
                "reason": "model_failed",
                "missing": [],
            }

    # ── Model 3: Jayden Method ───────────────
    missing_jayden = _missing_required_features("jayden", features)
    if missing_jayden:
        skipped_models["jayden"] = {
            "reason": "missing_features",
            "missing": missing_jayden,
        }
    else:
        delay_seconds = features.get("delay_seconds", 0)
        gap_seconds = features.get("gap_seconds", 0)
        jayden_score = jayden_method(delay_seconds, gap_seconds)
        jayden_severity = classify_delay(delay_seconds)

        # Convert score to pseudo-probabilities for consistency.
        probs = [0.1, 0.1, 0.1, 0.1]
        probs[jayden_severity] = 0.7

        results["jayden"] = {
            "model": "jayden",
            "predicted": jayden_severity,
            "score": jayden_score,
            "inputs": _feature_subset(features, REQUIRED_FEATURES["jayden"]),
            "probabilities": {
                "on_time": probs[0],
                "minor": probs[1],
                "moderate": probs[2],
                "severe": probs[3],
            },
        }

    if not results:
        return {
            "model_used": None,
            "predicted": 0,
            "label": severity_label(0),
            "color": severity_color(0),
            "all_models": {},
            "selection_context": {
                "strategy": "none_available",
                "skipped_models": skipped_models,
                "winner_score_100": 0.0,
            },
            "model_info": {
                "winner": None,
                "method": "MAGI - Multi-Answer Geographical Informations",
                "reference": "Chen et al. 2007",
            },
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
            result["score_100"] = _score_from_log_loss(ll)

        winner = min(log_losses, key=log_losses.get)
        strategy = "lowest_log_loss"

    else:
        # No actual known → use confidence (highest prob)
        confidences = {}
        for name, result in results.items():
            max_prob = _max_probability(result["probabilities"])
            confidences[name] = max_prob
            result["score_100"] = round(max_prob * 100, 1)

        winner = max(confidences, key=confidences.get)
        strategy = "highest_confidence"

    # ── Build Final Output ───────────────────
    winning_result = results[winner]
    severity       = winning_result["predicted"]

    return {
        "model_used":   winner,
        "predicted":    severity,
        "label":        severity_label(severity),
        "color":        severity_color(severity),

        "all_models":   results,
        "selection_context": {
            "strategy": strategy,
            "skipped_models": skipped_models,
            "winner_score_100": results[winner].get("score_100", 0.0),
        },

        "model_info": {
            "winner":      winner,
            "method":      "MAGI - Multi-Answer Geographical Informations",
            "reference":   "Chen et al. 2007"
        }
    }