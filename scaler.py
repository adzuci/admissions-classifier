"""Simple StandardScaler-like class (no sklearn)."""
import numpy as np


class SimpleScaler:
    """Minimal scaler with fit/transform for standardization."""

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_
