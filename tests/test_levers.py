"""Guard tests for the lever models (aggregate/CSCT, glass-box; PySR smoke)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from punching_shear import (
    CSCTBasisRegressor,
    GAMRegressor,
    STRESS_MONOTONE,
    load_dataset,
    make_ebm,
)

ds = load_dataset()


def _dg_subset():
    dg = pd.to_numeric(ds.raw["dg"], errors="coerce")
    mask = dg.notna().to_numpy()
    X = ds.X[mask].copy()
    X["dg"] = dg[mask].to_numpy()
    return X, ds.y_stress[mask]


def test_csct_basis_fits_and_is_closed_form():
    X, y = _dg_subset()
    est = CSCTBasisRegressor().fit(X, y)
    pred = est.predict(X)
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)
    assert 0.2 < est.p_ < 0.45          # material exponent near EC2's 1/3
    assert est.lam_ >= 0                # non-negative size-aggregate weakening
    assert "dg" in est.formula_()


def test_gam_monotone_fits_and_exposes_shapes():
    est = GAMRegressor(monotone=STRESS_MONOTONE).fit(ds.X, ds.y_stress)
    pred = est.predict(ds.X)
    assert pred.shape == (len(ds),) and np.all(np.isfinite(pred))
    shapes = est.shapes()
    assert set(shapes) == set(ds.feature_names)


def test_ebm_clones_fits_predicts_on_groupfold():
    tr, te = next(GroupKFold(5).split(ds.X, ds.y_stress, ds.groups))
    est = clone(make_ebm(0))
    est.fit(ds.X.iloc[tr], ds.y_stress.iloc[tr])
    pred = est.predict(ds.X.iloc[te])
    assert pred.shape == (len(te),) and np.all(np.isfinite(pred))


def test_pysr_wrapper_constructs_without_julia():
    """The wrapper is importable/constructible without loading the Julia backend
    (pysr is imported only inside fit())."""
    from punching_shear.symbolic import PySRFormulaRegressor
    est = PySRFormulaRegressor(mode="correction", cols=("col_area", "u0_perim", "d"))
    assert est.mode == "correction" and est.cols[-1] == "d"
    assert clone(est).get_params()["mode"] == "correction"
