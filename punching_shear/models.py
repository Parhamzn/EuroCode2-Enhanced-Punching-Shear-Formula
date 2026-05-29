"""The model zoo: every estimator as a leak-free pipeline on the shared folds.

All scaling happens inside the pipeline, so the scaler is refit on the training
part of each fold (no test-set statistics leak in). Regularised and kernel models
are wrapped in :class:`GridSearchCV`, whose inner CV only ever sees the outer
training fold -- i.e. nested cross-validation. Trees/forests are left unscaled
(scaling is irrelevant to them) but kept in a pipeline for a uniform interface.

The Eurocode 2 baseline (:class:`~punching_shear.eurocode.EC2Regressor`) is added
as just another model so it is scored on identical folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .data import FEATURES
from .eurocode import EC2Regressor


class InteractionAdder(BaseEstimator, TransformerMixin):
    """Append the product of two named features (default d * fcm_cyl).

    Reproduces the original study's non-linear OLS term, but built leak-free and
    mean-centred-friendly (StandardScaler downstream removes the main-effect
    collinearity that inflated the original in-sample R^2).
    """

    def __init__(self, a: str = "d", b: str = "fcm_cyl", feature_names=FEATURES):
        # Store params unchanged so sklearn.clone() works (no list() copy here).
        self.a = a
        self.b = b
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            cols = list(X.columns)
            Xv = X.to_numpy(dtype=float)
        else:
            cols = list(self.feature_names)
            Xv = np.asarray(X, dtype=float)
        ia, ib = cols.index(self.a), cols.index(self.b)
        inter = (Xv[:, ia] * Xv[:, ib]).reshape(-1, 1)
        return np.hstack([Xv, inter])


def _inner_cv(seed: int) -> KFold:
    return KFold(n_splits=5, shuffle=True, random_state=seed)


def _grid(estimator, param_grid, seed: int) -> GridSearchCV:
    return GridSearchCV(
        estimator,
        param_grid,
        scoring="neg_mean_squared_error",
        cv=_inner_cv(seed),
        n_jobs=-1,
    )


def build_models(seed: int = 19, include_ec2: bool = True) -> dict:
    """Return an ordered dict of ``name -> estimator`` (all sklearn-compatible).

    Every estimator predicts the punching shear stress [MPa] from :data:`FEATURES`.
    """
    scaler = lambda: StandardScaler()
    models: dict[str, object] = {}

    if include_ec2:
        # C_Rd,c refit per training fold -> EC2's functional form, no safety margin.
        models["EC2 (refit C_Rd,c)"] = EC2Regressor(C_Rdc=None, apply_caps=True)

    models["OLS"] = Pipeline([("scale", scaler()), ("lr", LinearRegression())])

    models["NLR (d x fcm)"] = Pipeline([
        ("interact", InteractionAdder("d", "fcm_cyl")),
        ("scale", scaler()),
        ("lr", LinearRegression()),
    ])

    models["Lasso"] = _grid(
        Pipeline([("scale", scaler()), ("m", Lasso(max_iter=50000))]),
        {"m__alpha": np.logspace(-4, 0.5, 25)},
        seed,
    )
    models["Ridge"] = _grid(
        Pipeline([("scale", scaler()), ("m", Ridge())]),
        {"m__alpha": np.logspace(-3, 2, 25)},
        seed,
    )
    models["ElasticNet"] = _grid(
        Pipeline([("scale", scaler()), ("m", ElasticNet(max_iter=50000))]),
        {"m__alpha": np.logspace(-4, 0.5, 15),
         "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        seed,
    )

    models["SVR (linear)"] = _grid(
        Pipeline([("scale", scaler()), ("m", SVR(kernel="linear"))]),
        {"m__C": [0.1, 1, 5, 10, 50], "m__epsilon": [0.01, 0.05, 0.1, 0.2]},
        seed,
    )
    models["SVR (poly-3)"] = _grid(
        Pipeline([("scale", scaler()), ("m", SVR(kernel="poly", degree=3))]),
        {"m__C": [1, 5, 10, 50], "m__gamma": ["scale", 0.1, 0.5],
         "m__epsilon": [0.05, 0.1]},
        seed,
    )
    models["SVR (RBF)"] = _grid(
        Pipeline([("scale", scaler()), ("m", SVR(kernel="rbf"))]),
        {"m__C": [1, 5, 10, 50, 100], "m__gamma": ["scale", 0.05, 0.1, 0.5, 1.0],
         "m__epsilon": [0.01, 0.05, 0.1]},
        seed,
    )

    models["Decision Tree"] = _grid(
        Pipeline([("m", DecisionTreeRegressor(random_state=seed))]),
        {"m__max_depth": [2, 3, 4, 5, 8, None],
         "m__min_samples_leaf": [1, 3, 5, 10],
         "m__ccp_alpha": [0.0, 1e-4, 1e-3, 1e-2]},
        seed,
    )
    models["Random Forest"] = _grid(
        # n_jobs=1 here: GridSearchCV owns the parallelism (avoid nested over-subscription).
        Pipeline([("m", RandomForestRegressor(random_state=seed, n_jobs=1))]),
        {"m__n_estimators": [300],
         "m__max_depth": [None, 6, 10],
         "m__max_features": [1.0, 0.5, "sqrt"],
         "m__min_samples_leaf": [1, 3, 5]},
        seed,
    )
    return models


def fit_full(model, X, y):
    """Clone-free convenience: fit a model on all data (for inspection plots)."""
    from sklearn.base import clone

    est = clone(model)
    est.fit(X, y)
    return est


def build_formula_models(seed: int = 19) -> dict:
    """Explainable / closed-form models + feature-engineering variants.

    Used to answer "can an interpretable, formula-yielding model beat EC2, and
    does mechanics-informed feature engineering help?". Evaluated on the same folds
    as everything else. Includes EC2, plain OLS and raw Random Forest as references.
    """
    from .eurocode import EC2Regressor
    from .features import MechanicsFeatures
    from .greybox import (
        EC2CorrectionRegressor,
        EC2FreeExponentRegressor,
        PowerLawRegressor,
    )
    from .symbolic import SymbolicFormulaRegressor

    models: dict[str, object] = {
        # --- references (same as the main study) ---
        "EC2 (refit C_Rd,c)": EC2Regressor(C_Rdc=None, apply_caps=True),
        "OLS (raw)": Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]),
        # Fixed (untuned) RF references: this set compares features & formulas, not RF tuning.
        "Random Forest (raw)": Pipeline([
            ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                        random_state=seed, n_jobs=-1))]),
        # --- closed-form / interpretable candidates ---
        "Power-law (d,rho,fck)": PowerLawRegressor(cols=("d", "rho_l", "fcm_cyl")),
        "Power-law (+geometry)": PowerLawRegressor(
            cols=("d", "rho_l", "fcm_cyl", "col_area", "u0_perim")),
        "EC2 free-exponent": EC2FreeExponentRegressor(),
        "EC2 x correction (grey-box)": EC2CorrectionRegressor(
            corr_cols=("col_area", "u0_perim", "d")),
        "Symbolic regression": SymbolicFormulaRegressor(
            cols=("d", "rho_l", "fcm_cyl"), generations=20, population_size=2000,
            parsimony_coefficient=0.005, random_state=seed),
        # --- feature-engineering lever (mechanics-informed inputs) ---
        "OLS + mechanics feats": Pipeline([
            ("feat", MechanicsFeatures()), ("scale", StandardScaler()),
            ("lr", LinearRegression())]),
        "Random Forest + mechanics feats": Pipeline([
            ("feat", MechanicsFeatures()),
            ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                        random_state=seed, n_jobs=-1))]),
    }
    return models
