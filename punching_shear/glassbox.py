"""Glass-box additive models (lever 4): EBM and monotone GAM.

Both are *interpretable* (additive, one readable shape function per feature) rather
than a single closed-form equation. They are useful in two ways: (a) as honest
competitors to EC2 under researcher-held-out CV, and (b) to reveal the true shape
of each feature effect, which informs/validates the closed-form power-law.

- EBM: use scikit-learn-compatible ``ExplainableBoostingRegressor(interactions=0)``
  directly (it clones cleanly into the CV harness); :func:`ebm_shapes` extracts the
  per-feature graphs.
- GAM: ``GAMRegressor`` wraps ``pygam.LinearGAM`` with optional per-feature
  monotonicity (design-safe: stress increases with rho_l & fck, decreases with d).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .data import FEATURES

# Design-sensible monotonic constraints for the punching STRESS target.
STRESS_MONOTONE = {"rho_l": "inc", "fcm_cyl": "inc", "d": "dec"}


class GAMRegressor(BaseEstimator, RegressorMixin):
    """pygam LinearGAM with optional per-feature monotonicity, sklearn-compatible."""

    def __init__(self, n_splines=10, lam=0.6, monotone=None, feature_names=FEATURES):
        self.n_splines = n_splines
        self.lam = lam
        self.monotone = monotone
        self.feature_names = feature_names

    def _cols(self, X):
        return list(X.columns) if hasattr(X, "columns") else list(self.feature_names)

    def fit(self, X, y):
        from pygam import LinearGAM, s

        self.cols_ = self._cols(X)
        mono = self.monotone or {}
        terms = None
        for i, c in enumerate(self.cols_):
            con = None
            if c in mono:
                con = "monotonic_inc" if mono[c] == "inc" else "monotonic_dec"
            term = s(i, n_splines=self.n_splines, constraints=con)
            terms = term if terms is None else terms + term
        self.gam_ = LinearGAM(terms, lam=self.lam).fit(
            np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.gam_.predict(np.asarray(X, dtype=float))

    def shapes(self, n=100):
        """Per-feature partial-dependence curves (grid, effect)."""
        out = {}
        for i, c in enumerate(self.cols_):
            XX = self.gam_.generate_X_grid(term=i, n=n)
            out[c] = (XX[:, i], self.gam_.partial_dependence(term=i, X=XX))
        return out


def make_ebm(seed: int = 19):
    """Additive (interaction-free) Explainable Boosting Regressor."""
    from interpret.glassbox import ExplainableBoostingRegressor

    return ExplainableBoostingRegressor(interactions=0, random_state=seed)


def ebm_shapes(fitted_ebm):
    """Extract per-feature (bin-centres, score) graphs from a fitted EBM."""
    g = fitted_ebm.explain_global()
    data = g.data()
    out = {}
    for i, name in enumerate(fitted_ebm.term_names_):
        d = g.data(i)
        if d and "names" in d and "scores" in d:
            out[name] = (np.asarray(d["names"][:-1], dtype=float)
                         if len(d["names"]) == len(d["scores"]) + 1
                         else np.asarray(d["names"], dtype=float),
                         np.asarray(d["scores"], dtype=float))
    return out
