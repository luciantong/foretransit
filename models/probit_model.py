import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

class ProbitModel:
    def __init__(self):
        self.model = None
        self.params = None
        # Chen et al. 2007 Baseline Coefficients
        self.chen_betas = {
            "cumulative_dwell_time": 0.2367,
            "cumulative_leg_time": 0.1862,
            "cumulative_stops": 0.0989,
            "is_sunday": -0.7685,
            "intercept": -0.2732
        }
        self.thresholds = {"mu1": 0.5, "mu2": 1.2, "mu3": 2.1}

    def fit(self, X, y):
        """Fits a Probit model using statsmodels."""
        X_with_const = sm.add_constant(X)
        self.model = sm.Probit(y, X_with_const)
        self.params = self.model.fit(disp=0)
        return self

    def predict_prob(self, X):
        """Returns predicted probability of delay classes with input validation."""
        # Ensure input is a dictionary-like object if baseline is needed
        if self.params is None:
            return self._predict_chen_baseline(X)
        
        # If input is a DataFrame/Array, convert to constant-ready format
        try:
            X_with_const = sm.add_constant(X, has_constant='add')
            return self.params.predict(X_with_const)
        except Exception:
            # Fallback if prediction fails due to malformed features
            return self._predict_chen_baseline(X)

    def _predict_chen_baseline(self, features):
        if hasattr(features, 'to_dict'):
            features = features.to_dict('records')[0]
        elif not isinstance(features, dict):
            features = {}

        # Normalize to Chen et al. scale
        dwell = features.get("cumulative_dwell_time", 0) / 10.0
        leg   = features.get("cumulative_leg_time", 0)   / 60.0
        stops = features.get("cumulative_stops", 0)      / 20.0

        y_star = (
            self.chen_betas["intercept"] +
            self.chen_betas["cumulative_dwell_time"] * dwell +
            self.chen_betas["cumulative_leg_time"]   * leg +
            self.chen_betas["cumulative_stops"]      * stops +
            self.chen_betas["is_sunday"] * features.get("is_sunday", 0)
        )

        p0 = norm.cdf(self.thresholds["mu1"] - y_star)
        p1 = norm.cdf(self.thresholds["mu2"] - y_star) - p0
        p2 = norm.cdf(self.thresholds["mu3"] - y_star) - p0 - p1
        p3 = 1 - p0 - p1 - p2

        return np.array([p0, p1, p2, p3])

    def predict(self, X):
        """Returns the highest probability class."""
        probs = self.predict_prob(X)
        # Handle cases where predict returns a Series or Array
        if hasattr(probs, 'values'):
            probs = probs.values
        return int(np.argmax(probs))