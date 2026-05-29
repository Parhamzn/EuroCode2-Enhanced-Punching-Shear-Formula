"""Symbolic regression (gplearn) wrapped as a leak-free, formula-yielding model.

Genetic programming searches the space of closed-form expressions directly, so the
fitted model *is* an equation. We constrain the operator set and apply a parsimony
penalty to keep the result compact and engineer-readable, then expose it via
``formula_()`` with the real feature names substituted for X0, X1, ...

Includes a compatibility shim: gplearn 0.4.2 calls the private
``BaseEstimator._validate_data`` that scikit-learn >= 1.6 removed; we forward it to
``sklearn.utils.validation.validate_data``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .data import FEATURES

# --- gplearn <-> scikit-learn >= 1.6 compatibility shim -----------------------
try:  # pragma: no cover - exercised whenever gplearn is installed
    from gplearn.genetic import BaseSymbolic

    if not hasattr(BaseSymbolic, "_validate_data"):
        from sklearn.utils.validation import validate_data as _sk_validate_data

        def _validate_data(self, X, y="no_validation", **kwargs):
            return _sk_validate_data(self, X, y, **kwargs)

        BaseSymbolic._validate_data = _validate_data
    _HAVE_GPLEARN = True
except Exception:  # pragma: no cover
    _HAVE_GPLEARN = False


DEFAULT_FUNCTIONS = ("add", "sub", "mul", "div", "sqrt", "log")


class SymbolicFormulaRegressor(BaseEstimator, RegressorMixin):
    """gplearn SymbolicRegressor restricted to a feature subset, yielding a formula.

    Operates on a DataFrame with :data:`FEATURES` columns (selects ``cols``), so it
    plugs into the same pipelines/CV as every other model.
    """

    def __init__(self, cols=("d", "rho_l", "fcm_cyl"), generations=20,
                 population_size=2000, parsimony_coefficient=0.005,
                 function_set=DEFAULT_FUNCTIONS, const_range=(-3.0, 3.0),
                 random_state=0):
        self.cols = cols
        self.generations = generations
        self.population_size = population_size
        self.parsimony_coefficient = parsimony_coefficient
        self.function_set = function_set
        self.const_range = const_range
        self.random_state = random_state

    def _select(self, X):
        if hasattr(X, "columns"):
            return np.asarray(X[list(self.cols)], dtype=float)
        arr = np.asarray(X, dtype=float)
        idx = [FEATURES.index(c) for c in self.cols]
        return arr[:, idx]

    def fit(self, X, y):
        if not _HAVE_GPLEARN:  # pragma: no cover
            raise ImportError("gplearn is required for SymbolicFormulaRegressor.")
        from gplearn.genetic import SymbolicRegressor

        self.sr_ = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            function_set=list(self.function_set),
            parsimony_coefficient=self.parsimony_coefficient,
            const_range=self.const_range,
            random_state=self.random_state,
            n_jobs=1,
            verbose=0,
        )
        self.sr_.fit(self._select(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.sr_.predict(self._select(X))

    def formula_(self) -> str:
        expr = str(self.sr_._program)
        for i, c in enumerate(self.cols):
            expr = expr.replace(f"X{i}", c)
        return f"v = {expr}   [MPa]"
