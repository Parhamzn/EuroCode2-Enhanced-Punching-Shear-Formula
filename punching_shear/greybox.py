"""Grey-box, closed-form regressors that stay code-like and interpretable.

Two estimators, both yielding a *neat mathematical formula*:

``PowerLawRegressor``
    v = C · ∏_j x_j^{a_j}, fit by OLS in log space. With the mechanics columns
    (d, rho_l, fcm_cyl) this is a free-exponent generalisation of the EC2 form;
    the data tend to recover EC2's cube-root structure (b, c ≈ 1/3).

``EC2FreeExponentRegressor``
    v = C · k(d) · (rho_l[%]·fck)^p, with k = 1+√(200/d) ≤ 2. This is EC2 with the
    cube-root exponent *freed*: fit (C, p) by least squares. EC2 is the special
    case C = C_Rd,c, p = 1/3 — so the fitted (C, p) is directly comparable.

Both expose ``formula_()`` returning a human-readable equation string.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from .data import FEATURES
from .eurocode import FCK_CAP, FCK_FLOOR, K_CAP, RHO_CAP_PCT


def _col(X, name):
    if hasattr(X, "columns"):
        return np.asarray(X[name], dtype=float)
    return np.asarray(X, dtype=float)[:, FEATURES.index(name)]


def _fck(fcm):
    return np.clip(fcm - 8.0, FCK_FLOOR, FCK_CAP)


class PowerLawRegressor(BaseEstimator, RegressorMixin):
    """v = C · ∏ x_j^{a_j} via log-log OLS on the chosen (positive) columns.

    ``use_fck`` transforms an ``fcm_cyl`` column to the clamped characteristic
    strength before taking logs (keeps the power law on the EC2 strength variable).
    """

    def __init__(self, cols=("d", "rho_l", "fcm_cyl"), use_fck=True, fit_method="nls"):
        self.cols = cols
        self.use_fck = use_fck
        self.fit_method = fit_method  # 'nls' (physical units) or 'log' (log-OLS)

    def _design(self, X):
        feats = []
        labels = []
        for c in self.cols:
            v = _col(X, c)
            if c == "fcm_cyl" and self.use_fck:
                v = _fck(v)
                labels.append("fck")
            else:
                labels.append(c)
            feats.append(v)
        self._feat_mat = np.column_stack(feats)
        self.labels_ = labels
        return self._feat_mat

    def fit(self, X, y):
        Z = self._design(X)            # raw (un-logged) feature columns
        y = np.asarray(y, dtype=float)
        # Warm start from log-OLS.
        lr = LinearRegression().fit(np.log(Z), np.log(y))
        a0 = lr.coef_
        C0 = float(np.exp(lr.intercept_))
        if self.fit_method == "log":
            self.C_, self.exponents_ = C0, a0
            return self
        # Refit in PHYSICAL units (avoids the log-retransformation / Jensen bias).
        def model(Zc, C, *a):
            out = np.full(Zc.shape[0], C, dtype=float)
            for j, aj in enumerate(a):
                out = out * Zc[:, j] ** aj
            return out
        try:
            popt, _ = curve_fit(model, Z, y, p0=[C0, *a0], maxfev=20000)
            self.C_, self.exponents_ = float(popt[0]), np.asarray(popt[1:])
        except Exception:  # pragma: no cover - fall back to the log fit
            self.C_, self.exponents_ = C0, a0
        return self

    def predict(self, X):
        out = np.full(_col(X, self.cols[0]).shape, self.C_, dtype=float)
        for c, a in zip(self.cols, self.exponents_):
            v = _col(X, c)
            if c == "fcm_cyl" and self.use_fck:
                v = _fck(v)
            out = out * v ** a
        return out

    def formula_(self) -> str:
        terms = " * ".join(f"{lab}^{a:.3f}" for lab, a in zip(self.labels_, self.exponents_))
        return f"v = {self.C_:.4f} * {terms}   [MPa]"


class EC2FreeExponentRegressor(BaseEstimator, RegressorMixin):
    """v = C · k(d) · (rho_l[%]·fck)^p, fitting (C, p). EC2 = (C_Rd,c, 1/3)."""

    def __init__(self, p0=1.0 / 3.0, apply_caps=True):
        self.p0 = p0
        self.apply_caps = apply_caps

    def _parts(self, X):
        d = _col(X, "d")
        rho = _col(X, "rho_l")
        fck = _fck(_col(X, "fcm_cyl"))
        if self.apply_caps:
            rho = np.minimum(rho, RHO_CAP_PCT)
        k = 1.0 + np.sqrt(200.0 / d)
        if self.apply_caps:
            k = np.minimum(k, K_CAP)
        return k, rho * fck

    def fit(self, X, y):
        k, rf = self._parts(X)

        def model(_, C, p):
            return C * k * rf ** p

        (self.C_, self.p_), _ = curve_fit(
            model, np.zeros_like(k), np.asarray(y, dtype=float),
            p0=[0.18, self.p0], maxfev=10000,
        )
        return self

    def predict(self, X):
        k, rf = self._parts(X)
        return self.C_ * k * rf ** self.p_

    def formula_(self) -> str:
        return (f"v = {self.C_:.4f} * (1+sqrt(200/d)) * (100*rho_frac*fck)^{self.p_:.3f}   "
                f"[MPa]   (EC2: C=0.18, p=1/3)")


class EC2CorrectionRegressor(BaseEstimator, RegressorMixin):
    """Grey-box: v = v_EC2 · K, with K a power-law correction on features EC2 ignores.

    The base is the EC2 stress (``C_Rdc`` refit per fold by default), so the floor is
    "no worse than EC2"; the correction ``K = C'·∏ x_j^{a_j}`` (fit by NLS on the
    residual ratio v/v_EC2) only mops up residual scatter from geometry/size that the
    EC2 stress form drops. This is the literature's recommended low-risk hybrid.
    """

    def __init__(self, corr_cols=("col_area", "u0_perim", "d"),
                 C_Rdc=None, apply_caps=True):
        self.corr_cols = corr_cols
        self.C_Rdc = C_Rdc
        self.apply_caps = apply_caps

    def _base(self, X, fit=False, y=None):
        from .eurocode import ec2_stress, refit_CRdc
        d = _col(X, "d"); rho = _col(X, "rho_l"); fck = _fck(_col(X, "fcm_cyl"))
        if fit:
            self.C_Rdc_ = (refit_CRdc(d, rho, fck, y, apply_caps=self.apply_caps)
                           if self.C_Rdc is None else float(self.C_Rdc))
        return ec2_stress(d, rho, fck, C_Rdc=self.C_Rdc_, apply_caps=self.apply_caps)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        base = self._base(X, fit=True, y=y)
        ratio = y / base
        self.corr_ = PowerLawRegressor(cols=self.corr_cols, use_fck=False).fit(X, ratio)
        return self

    def predict(self, X):
        return self._base(X) * self.corr_.predict(X)

    def formula_(self) -> str:
        terms = " * ".join(f"{lab}^{a:.3f}"
                           for lab, a in zip(self.corr_.labels_, self.corr_.exponents_))
        return (f"v = v_EC2(C={self.C_Rdc_:.3f}) * [ {self.corr_.C_:.4f} * {terms} ]   [MPa]")
