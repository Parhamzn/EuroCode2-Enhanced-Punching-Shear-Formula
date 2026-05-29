"""Interpretable ML for the punching-shear resistance of RC flat slabs.

A corrected, leak-free rebuild of the original ETH HS2021 SciML study. The two
defining changes versus the original notebooks are:

1. The modelling target is the punching shear **stress** ``v = V / (u1*d)`` [MPa],
   not the absolute failure load ``V_test`` [MN]. Absolute load is mechanically
   proportional to the control area ``u1*d``, so regressing load on the effective
   depth ``d`` mostly relearns a trivial size effect rather than punching physics
   (corr(d, V_test) = 0.89 vs corr(d, v) = -0.27 on this dataset).

2. Every model (including the Eurocode 2 baseline) is evaluated with the *same*
   cross-validation folds, leak-free pipelines (the scaler is refit inside each
   fold), and metrics reported in **physical units** (MPa) with confidence
   intervals and paired significance tests.

See ``punching_shear.data``, ``punching_shear.eurocode``,
``punching_shear.evaluation`` and ``punching_shear.models``.
"""

from .data import Dataset, load_dataset, FEATURES, FEATURE_LABELS
from .eurocode import ec2_stress, refit_CRdc, EC2Regressor, CRD_C_DESIGN, CRD_C_CHAR
from .evaluation import (
    regression_metrics,
    make_cv,
    cross_validate_models,
    paired_error_test,
    oof_predictions,
)
from .models import build_models

__all__ = [
    "Dataset",
    "load_dataset",
    "FEATURES",
    "FEATURE_LABELS",
    "ec2_stress",
    "refit_CRdc",
    "EC2Regressor",
    "CRD_C_DESIGN",
    "CRD_C_CHAR",
    "regression_metrics",
    "make_cv",
    "cross_validate_models",
    "paired_error_test",
    "oof_predictions",
    "build_models",
]
