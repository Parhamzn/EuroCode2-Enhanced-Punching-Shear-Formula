"""Mechanics-informed feature engineering for punching-shear stress.

The raw features (d, col_area, rho_l, fcm_cyl, u0_perim) are physical but not in
the form mechanics suggests. This transformer adds dimensionless / code-style
features that let even a linear model express EC2-like behaviour and test whether
better inputs (rather than fancier models) close the gap to EC2:

  * ``ec2_basis``   = k * (rho_l[%]·fck)^(1/3)            -- a linear fit on this alone IS EC2;
  * ``size_k``      = 1 + sqrt(200/d), capped at 2        -- EC2 size-effect factor;
  * ``cbrt_rho_fck``= (rho_l[%]·fck)^(1/3)                -- the material/reinforcement core;
  * ``log_d, log_rho, log_fck``                           -- power-law flexibility;
  * ``shape``       = sqrt(col_area)/(1000·u0_perim)      -- dimensionless column compactness.

``fck`` is ``fcm_cyl-8`` clamped to the EC2 class range; ``rho_l`` is capped at 2 %.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import FEATURES
from .eurocode import FCK_CAP, FCK_FLOOR, K_CAP, RHO_CAP_PCT

ENGINEERED = [
    "ec2_basis", "size_k", "cbrt_rho_fck",
    "log_d", "log_rho", "log_fck", "shape",
]


def _frame(X):
    if hasattr(X, "columns"):
        return pd.DataFrame(X).reset_index(drop=True)
    return pd.DataFrame(np.asarray(X, dtype=float), columns=FEATURES)


def engineer(X) -> pd.DataFrame:
    """Return the engineered feature frame (also usable outside a pipeline)."""
    df = _frame(X)
    d = df["d"].to_numpy(float)
    rho = np.minimum(df["rho_l"].to_numpy(float), RHO_CAP_PCT)
    fck = np.clip(df["fcm_cyl"].to_numpy(float) - 8.0, FCK_FLOOR, FCK_CAP)
    area = df["col_area"].to_numpy(float)
    u0 = df["u0_perim"].to_numpy(float)

    k = np.minimum(1.0 + np.sqrt(200.0 / d), K_CAP)
    cbrt = np.cbrt(rho * fck)
    out = pd.DataFrame({
        "ec2_basis": k * cbrt,
        "size_k": k,
        "cbrt_rho_fck": cbrt,
        "log_d": np.log(d),
        "log_rho": np.log(rho),
        "log_fck": np.log(fck),
        "shape": np.sqrt(area) / (1000.0 * u0),
    })
    return out[ENGINEERED]


class MechanicsFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer wrapping :func:`engineer` (stateless, leak-free)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return engineer(X).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(ENGINEERED, dtype=object)
