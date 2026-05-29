"""Eurocode 2 (DIN EN 1992-1-1, 6.4) punching-shear resistance, in stress space.

The EC2 resistance of a slab without shear reinforcement is a *stress*::

    v_Rd,c = C_Rd,c * k * (100 * rho_l * fck) ** (1/3)        [MPa]

with the size-effect factor ``k = 1 + sqrt(200/d) <= 2.0`` (d in mm) and the
flexural ratio capped at ``rho_l <= 0.02`` (i.e. 2.0 % here). The ``k1*sigma_cp``
term is dropped (no in-plane-stress data in the dataset). Because our modelling
target is the stress ``v`` (not the load), EC2 enters the comparison directly as
this stress -- no control perimeter is needed.

``C_Rd,c`` carries the safety content:
  * ``0.18``        -- characteristic resistance (gamma_c = 1.0),
  * ``0.18/1.5 = 0.12`` -- design resistance (the value embedded in Data.xlsx's v_Rd).

For a fair comparison against data-driven models we *refit* ``C_Rd,c`` to the
measured stress (least squares through the origin), removing the built-in safety
margin so the baseline reflects EC2's functional form rather than its conservatism.

Note on units: ``(100 * rho_l_fraction)`` equals ``rho_l`` expressed in percent,
which is how the dataset stores it -- so the basis uses ``rho_l[%] * fck`` directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

CRD_C_CHAR = 0.18          # characteristic (gamma_c = 1.0)
CRD_C_DESIGN = 0.18 / 1.5  # design value embedded in the spreadsheet (= 0.12)

# EC2 caps / concrete-class validity range.
K_CAP = 2.0
RHO_CAP_PCT = 2.0          # rho_l <= 0.02 expressed as a percentage
FCK_FLOOR = 12.0
FCK_CAP = 90.0


def ec2_basis(d, rho_l_pct, fck, *, apply_caps: bool = True):
    """Return the EC2 shape function ``k * (rho_l[%] * fck) ** (1/3)`` (stress / C_Rd,c).

    With ``apply_caps`` the EC2 limits (k<=2, rho_l<=2 %, C12/15..C90/105) are
    enforced; set it to ``False`` to reproduce a spreadsheet that omitted them.
    """
    d = np.asarray(d, dtype=float)
    rho = np.asarray(rho_l_pct, dtype=float)
    fc = np.asarray(fck, dtype=float)

    k = 1.0 + np.sqrt(200.0 / d)
    if apply_caps:
        k = np.minimum(k, K_CAP)
        rho = np.minimum(rho, RHO_CAP_PCT)
        fc = np.clip(fc, FCK_FLOOR, FCK_CAP)
    return k * np.cbrt(rho * fc)


def ec2_stress(d, rho_l_pct, fck, C_Rdc: float = CRD_C_DESIGN, *, apply_caps: bool = True):
    """EC2 punching resistance stress ``v_Rd,c`` [MPa]."""
    return C_Rdc * ec2_basis(d, rho_l_pct, fck, apply_caps=apply_caps)


def refit_CRdc(d, rho_l_pct, fck, v_measured, *, apply_caps: bool = True) -> float:
    """Least-squares-through-origin estimate of ``C_Rd,c`` against measured stress.

    Closed form: ``C* = sum(v * b) / sum(b**2)`` with ``b`` the EC2 basis.
    """
    b = ec2_basis(d, rho_l_pct, fck, apply_caps=apply_caps)
    v = np.asarray(v_measured, dtype=float)
    denom = float(np.sum(b * b))
    if denom == 0.0:  # pragma: no cover - degenerate
        raise ValueError("Degenerate EC2 basis (all zero); cannot refit C_Rd,c.")
    return float(np.sum(v * b) / denom)


class EC2Regressor(BaseEstimator, RegressorMixin):
    """EC2 punching formula as an sklearn estimator (target = shear stress [MPa]).

    Reads ``d``, ``rho_l`` and ``fcm_cyl`` from the feature frame (fck is derived
    as ``fcm_cyl - 8`` and clamped). When ``C_Rdc`` is ``None`` the coefficient is
    *refit per training fold* (least squares through origin), which makes EC2 a
    leak-free competitor evaluated on exactly the same folds as the ML models.
    """

    def __init__(self, C_Rdc: float | None = None, apply_caps: bool = True,
                 fcm_to_fck_offset: float = 8.0):
        self.C_Rdc = C_Rdc
        self.apply_caps = apply_caps
        self.fcm_to_fck_offset = fcm_to_fck_offset

    def _unpack(self, X):
        # Accept a DataFrame (preferred) or an array with FEATURES column order.
        if hasattr(X, "columns"):
            d = np.asarray(X["d"], dtype=float)
            rho = np.asarray(X["rho_l"], dtype=float)
            fcm = np.asarray(X["fcm_cyl"], dtype=float)
        else:
            X = np.asarray(X, dtype=float)
            # FEATURES = [d, col_area, rho_l, fcm_cyl, u0_perim]
            d, rho, fcm = X[:, 0], X[:, 2], X[:, 3]
        fck = np.clip(fcm - self.fcm_to_fck_offset, FCK_FLOOR, FCK_CAP)
        return d, rho, fck

    def fit(self, X, y=None):
        d, rho, fck = self._unpack(X)
        if self.C_Rdc is None:
            if y is None:
                raise ValueError("EC2Regressor needs y to refit C_Rd,c.")
            self.C_Rdc_ = refit_CRdc(d, rho, fck, y, apply_caps=self.apply_caps)
        else:
            self.C_Rdc_ = float(self.C_Rdc)
        return self

    def predict(self, X):
        d, rho, fck = self._unpack(X)
        return ec2_stress(d, rho, fck, C_Rdc=self.C_Rdc_, apply_caps=self.apply_caps)
