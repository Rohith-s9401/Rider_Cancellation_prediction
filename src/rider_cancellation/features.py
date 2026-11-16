"""Explicit feature allowlist: downstream outcome columns are never consulted."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

REQUIRED = ("order_time", "allot_time", "first_mile_distance", "last_mile_distance")


class AllocationFeatures(BaseEstimator, TransformerMixin):
    """Prediction occurs immediately after allocation; distances must exist then."""

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        missing = set(REQUIRED) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required input columns: {sorted(missing)}")
        order = pd.to_datetime(X.order_time, errors="coerce")
        allocation = pd.to_datetime(X.allot_time, errors="coerce")
        if order.isna().any() or allocation.isna().any():
            raise ValueError("Invalid or missing order_time/allot_time")
        delay = (allocation - order).dt.total_seconds() / 60
        if (delay < 0).any():
            raise ValueError("Allocation precedes order creation")
        first = pd.to_numeric(X.first_mile_distance, errors="coerce")
        last = pd.to_numeric(X.last_mile_distance, errors="coerce")
        if (first.dropna() < 0).any() or (last.dropna() < 0).any():
            raise ValueError("Distances must be nonnegative")
        hour = allocation.dt.hour + allocation.dt.minute / 60
        features = pd.DataFrame({
            "first_mile_distance": first,
            "last_mile_distance": last,
            "total_distance": first + last,
            "first_mile_share": first / (first + last).replace(0, np.nan),
            "allocation_delay_minutes": delay,
            "log_allocation_delay": np.log1p(delay),
            "allocation_hour_sin": np.sin(2 * np.pi * hour / 24),
            "allocation_hour_cos": np.cos(2 * np.pi * hour / 24),
            "allocation_weekday": allocation.dt.dayofweek,
            "is_weekend": (allocation.dt.dayofweek >= 5).astype(int),
            "order_hour": order.dt.hour,
        }, index=X.index)
        return features.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "first_mile_distance", "last_mile_distance", "total_distance",
            "first_mile_share", "allocation_delay_minutes", "log_allocation_delay",
            "allocation_hour_sin", "allocation_hour_cos", "allocation_weekday",
            "is_weekend", "order_hour",
        ], dtype=object)
