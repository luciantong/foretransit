#Referenced from EVA
#Multi-Answer Geographical Informations (MAGI) - Ensemble of Probit, XGBoost, and Jayden Method

import math
from pipeline.utils import (
    classify_delay,
    severity_label,
    severity_color
)

from models.jayden_model import JaydenMethod
from models.probit_model import ProbitModel
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

# Maps predicted severity (0-3) to a base score_100
# 0 = on_time -> high score, 3 = severe -> low score
SEVERITY_BASE_SCORE = {0: 90, 1: 65, 2: 40, 3: 15}


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
    scaled = 1 - min(max(log_loss, 0.0), 2.5) / 2.5
    return round(scaled * 100, 1)


def _score_from_prediction(predicted, probabilities):
    """
    Derives score_100 from the model's actual predicted severity and confidence.
    Base score comes from severity band (0=90, 1=65, 2=40, 3=15).
    Confidence nudges the score +/-5 points within the band.
    """
    base = SEVERITY_BASE_SCORE.get(int(predicted), 50)
    confidence = _max_probability(probabilities)
    nudge = round((confidence - 0.55) * 10, 1)
    return round(min(100, max(0, base + nudge)), 1)


def _feature_subset(features, keys):
    return {key: features.get(key) for key in keys}


def _service_status_from_score(score_100):
    if score_100 >= 90:
        return "On time"
    if score_100 >= 50:
        return "Minor delays"
    return "Major delays"


def _advice_text_from_score(score_100):
    if score_100 >= 90:
        return "Service is running on time. No need to rush."
    if score_100 >= 50:
        return "Minor traffic detected. You might have a 5-minute wait."
    return "Major delays. Check for a nearby subway or ride-share."


# --- MAGI ---
def run_magi(features, actual_severity=None):
    """
    Runs Probit, XGBoost, and Jayden Method.
    Picks winner by log loss (if actual known) or by domain priority:
      - delay_seconds > 60  -> Jayden  (real-time delay signal)
      - delay_seconds <= 60 -> XGBoost (schedule/context features)
      - XGBoost unavailable -> Probit  (Chen baseline fallback)
    """

    results = {}
    skipped_models = {}

    # -- Model 1: Probit (Chen et al.) --
    missing_probit = _missing_required_features("probit", features)
    if missing_probit:
        skipped_models["probit"] = {"reason": "missing_features", "missing": missing_probit}
    else:
        try:
            _probit = ProbitModel()
            probs = _probit._predict_chen_baseline(features)
            predicted = int(probs.argmax())
            probs_dict = {
                "on_time":  round(float(probs[0]), 4),
                "minor":    round(float(probs[1]), 4),
                "moderate": round(float(probs[2]), 4),
                "severe":   round(float(probs[3]), 4),
            }
            results["probit"] = {
                "model":         "probit",
                "predicted":     predicted,
                "inputs":        _feature_subset(features, REQUIRED_FEATURES["probit"]),
                "probabilities": probs_dict,
                "score_100":     _score_from_prediction(predicted, probs_dict),
            }
        except Exception as e:
            skipped_models["probit"] = {"reason": "model_failed", "error": str(e)}

    # -- Model 2: XGBoost --
    missing_xgb = _missing_required_features("xgboost", features)
    if missing_xgb:
        skipped_models["xgboost"] = {"reason": "missing_features", "missing": missing_xgb}
    else:
        mode = features.get("mode", "bus")
        xgboost = xgboost_predict(features, mode)
        if xgboost:
            xgboost["inputs"]    = _feature_subset(features, REQUIRED_FEATURES["xgboost"])
            xgboost["score_100"] = _score_from_prediction(
                xgboost["predicted"], xgboost["probabilities"])
            results["xgboost"] = xgboost
        else:
            skipped_models["xgboost"] = {"reason": "model_failed", "missing": []}

    # -- Model 3: Jayden Method --
    missing_jayden = _missing_required_features("jayden", features)
    if missing_jayden:
        skipped_models["jayden"] = {"reason": "missing_features", "missing": missing_jayden}
    else:
        delay_seconds   = features.get("delay_seconds", 0)
        gap_seconds     = features.get("gap_seconds", 0)
        _jayden         = JaydenMethod()
        jayden_score    = _jayden.jayden_method(delay_seconds, gap_seconds)
        jayden_severity = classify_delay(delay_seconds)

        probs = [0.1, 0.1, 0.1, 0.1]
        probs[jayden_severity] = 0.7
        probs_dict = {
            "on_time":  probs[0],
            "minor":    probs[1],
            "moderate": probs[2],
            "severe":   probs[3],
        }

        results["jayden"] = {
            "model":         "jayden",
            "predicted":     jayden_severity,
            "score":         jayden_score,
            "inputs":        _feature_subset(features, REQUIRED_FEATURES["jayden"]),
            "probabilities": probs_dict,
            "score_100":     round(max(0, min(100, (10 - jayden_score) / 9 * 100)), 1),
        }

    # -- No models ran --
    if not results:
        fallback_score = 0.0
        return {
            "model_used": None,
            "predicted":  0,
            "label":      severity_label(0),
            "color":      severity_color(0),
            "all_models": {},
            "selection_context": {
                "strategy":         "none_available",
                "skipped_models":   skipped_models,
                "winner_score_100": fallback_score,
            },
            "service_status": _service_status_from_score(fallback_score),
            "advice_text":    _advice_text_from_score(fallback_score),
            "model_info": {
                "winner":    None,
                "method":    "MAGI - Multi-Answer Geographical Informations",
                "reference": "Chen et al. 2007",
            },
        }

    # -- Pick Winner --
    if actual_severity is not None:
        # Training mode: pick by lowest log loss against known ground truth
        log_losses = {}
        for name, result in results.items():
            probs_dict   = result["probabilities"]
            severity_key = actual_severity
            if severity_key not in probs_dict and str(severity_key) in probs_dict:
                severity_key = str(severity_key)
            target_prob      = probs_dict.get(severity_key, 1e-10)
            ll               = -math.log(target_prob + 1e-10)
            log_losses[name] = round(ll, 4)
            result["log_loss"]  = round(ll, 4)
            result["score_100"] = _score_from_log_loss(ll)

        winner   = min(log_losses, key=log_losses.get)
        strategy = "lowest_log_loss"

    else:
        # Live mode: domain-priority selection
        # Each model is best at a different signal:
        #   Jayden  -> real-time delay_seconds (direct measure)
        #   XGBoost -> schedule/context features (hour, stops, day)
        #   Probit  -> Chen et al. baseline (structural fallback)
        delay_seconds = features.get("delay_seconds", 0)

        if "jayden" in results and delay_seconds > 30:
            winner   = "jayden"
            strategy = "jayden_priority_delay_signal"
        elif "xgboost" in results and results["xgboost"].get("inference_ok", True):
            winner   = "xgboost"
            strategy = "xgboost_priority_schedule_features"
        else:
            confidences = {
                name: _max_probability(result["probabilities"])
                for name, result in results.items()
            }
            winner   = max(confidences, key=confidences.get)
            strategy = "probit_fallback_highest_confidence"

    # -- Build Final Output --
    winning_result = results[winner]
    severity       = winning_result["predicted"]
    winner_score   = results[winner].get("score_100", 0.0)

    return {
        "model_used": winner,
        "predicted":  severity,
        "label":      severity_label(severity),
        "color":      severity_color(severity),
        "all_models": results,
        "selection_context": {
            "strategy":         strategy,
            "skipped_models":   skipped_models,
            "winner_score_100": winner_score,
        },
        "service_status": _service_status_from_score(winner_score),
        "advice_text":    _advice_text_from_score(winner_score),
        "model_info": {
            "winner":    winner,
            "method":    "MAGI - Multi-Answer Geographical Informations",
            "reference": "Chen et al. 2007",
        },
    }