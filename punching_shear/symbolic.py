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


def pysr_available() -> bool:
    """True if PySR's Julia backend is provisioned and importable."""
    try:  # pragma: no cover - environment dependent
        from pysr import PySRRegressor  # noqa: F401
        return True
    except Exception:
        return False


class PySRFormulaRegressor(BaseEstimator, RegressorMixin):
    """PySR symbolic regression (constant-optimising; stronger than gplearn).

    ``mode='direct'``    : v = f(features) discovered directly.
    ``mode='correction'``: v = v_EC2 * K(features), K discovered on the residual
    ratio v/v_EC2 (the literature's recommended low-risk hybrid; floor = EC2).
    Requires the PySR Julia backend (see :func:`pysr_available`).
    """

    def __init__(self, cols=("d", "rho_l", "fcm_cyl"), mode="correction",
                 niterations=40, maxsize=22, random_state=0,
                 binary_operators=("+", "-", "*", "/"),
                 unary_operators=("sqrt", "square", "cube", "log")):
        self.cols = cols
        self.mode = mode
        self.niterations = niterations
        self.maxsize = maxsize
        self.random_state = random_state
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators

    def _select(self, X):
        if hasattr(X, "columns"):
            return np.asarray(X[list(self.cols)], dtype=float)
        idx = [FEATURES.index(c) for c in self.cols]
        return np.asarray(X, dtype=float)[:, idx]

    def _ec2(self, X, fit=False, y=None):
        from .eurocode import ec2_stress, refit_CRdc
        d = np.asarray(X["d"], float); rho = np.asarray(X["rho_l"], float)
        fck = np.clip(np.asarray(X["fcm_cyl"], float) - 8.0, 12.0, 90.0)
        if fit:
            self.C_Rdc_ = refit_CRdc(d, rho, fck, y)
        return ec2_stress(d, rho, fck, C_Rdc=self.C_Rdc_)

    def fit(self, X, y):
        from pysr import PySRRegressor

        y = np.asarray(y, dtype=float)
        target = y
        if self.mode == "correction":
            base = self._ec2(X, fit=True, y=y)
            target = y / base
        self.model_ = PySRRegressor(
            niterations=self.niterations, maxsize=self.maxsize,
            binary_operators=list(self.binary_operators),
            unary_operators=list(self.unary_operators),
            random_state=self.random_state, deterministic=True,
            parallelism="serial", progress=False, verbosity=0,
            temp_equation_file=True,
        )
        self.model_.fit(self._select(X), target)
        return self

    def predict(self, X):
        pred = self.model_.predict(self._select(X))
        if self.mode == "correction":
            return self._ec2(X) * pred
        return pred

    def formula_(self) -> str:
        eq = str(self.model_.get_best()["equation"])
        for i, c in enumerate(self.cols):
            eq = eq.replace(f"x{i}", c)
        if self.mode == "correction":
            return f"v = v_EC2(C={self.C_Rdc_:.3f}) * [ {eq} ]   [MPa]"
        return f"v = {eq}   [MPa]"
