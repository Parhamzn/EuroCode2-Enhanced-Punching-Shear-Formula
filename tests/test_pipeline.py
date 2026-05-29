"""Sanity tests guarding the corrections that make this rebuild trustworthy.

Run with:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold

from punching_shear import (
    build_models,
    ec2_stress,
    load_dataset,
    refit_CRdc,
)
from punching_shear.data import COMPANION_PATH, FEATURES
from punching_shear.eurocode import CRD_C_CHAR, CRD_C_DESIGN, EC2Regressor

ds = load_dataset()


def test_dataset_shape_and_completeness():
    assert len(ds) == 336
    assert list(ds.X.columns) == FEATURES
    # The five modelling features must be fully observed (no silent row drops).
    assert ds.X.isna().sum().sum() == 0
    assert ds.groups.nunique() == 55


def test_stress_target_definition():
    # v = V_test * 1e6 / (u1*d); inverting must recover the load exactly.
    recovered = ds.y_stress * ds.beta / 1.0e6
    assert np.allclose(recovered, ds.y_load, rtol=1e-9, atol=1e-9)
    assert (ds.y_stress > 0).all()


def test_ec2_reproduces_spreadsheet_vrd():
    """Our EC2 stress formula (design C, caps on) must match Data.xlsx v_Rd."""
    comp = pd.read_excel(COMPANION_PATH)
    comp.columns = [str(c) for c in comp.columns]
    v_sheet = pd.to_numeric(comp["v_Rd"], errors="coerce").to_numpy()
    d, rho, fck = ds.ec2_inputs()
    v_ours = ec2_stress(d, rho, fck, C_Rdc=CRD_C_DESIGN, apply_caps=True)
    rel = np.abs(v_ours - v_sheet) / v_sheet
    assert np.median(rel) < 1e-6  # identical for the vast majority of rows


def test_refit_C_beats_textbook_constant():
    d, rho, fck = ds.ec2_inputs()
    y = ds.y_stress.to_numpy()
    C = refit_CRdc(d, rho, fck, y)
    mse_refit = np.mean((ec2_stress(d, rho, fck, C_Rdc=C) - y) ** 2)
    mse_char = np.mean((ec2_stress(d, rho, fck, C_Rdc=CRD_C_CHAR) - y) ** 2)
    assert mse_refit <= mse_char
    assert 0.15 < C < 0.4  # sane, near the literature ~0.23


def test_size_effect_artifact_is_real():
    """Documents WHY we model stress: load is dominated by the size effect."""
    r_load = np.corrcoef(ds.X["d"], ds.y_load)[0, 1]
    r_stress = np.corrcoef(ds.X["d"], ds.y_stress)[0, 1]
    assert r_load > 0.8           # load ~ proportional to depth (trivial)
    assert r_stress < 0.0         # stress is NOT explained by depth alone


def test_ec2_regressor_refits_per_fit():
    est = EC2Regressor(C_Rdc=None)
    est.fit(ds.X.iloc[:200], ds.y_stress.iloc[:200])
    c_a = est.C_Rdc_
    est.fit(ds.X.iloc[200:], ds.y_stress.iloc[200:])
    c_b = est.C_Rdc_
    assert c_a != c_b            # coefficient is data-dependent (refit each fold)
    pred = est.predict(ds.X.iloc[:5])
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)


def test_pipeline_scaler_has_no_leakage():
    """A pipeline fit on a subset must scale using only that subset's statistics."""
    models = build_models()
    ols = clone(models["OLS"])
    sub = slice(0, 100)
    ols.fit(ds.X.iloc[sub], ds.y_stress.iloc[sub])
    scaler = ols.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, ds.X.iloc[sub].mean().to_numpy(), rtol=1e-9)
    # And NOT the full-data mean (the original notebooks' leak).
    assert not np.allclose(scaler.mean_, ds.X.mean().to_numpy())


def test_all_models_clone_fit_predict():
    models = build_models()
    assert len(models) == 11
    # Fit each on a single train fold and predict the held-out fold; finite output.
    tr, te = next(KFold(5, shuffle=True, random_state=0).split(ds.X))
    for name, model in models.items():
        est = clone(model)
        est.fit(ds.X.iloc[tr], ds.y_stress.iloc[tr])
        pred = est.predict(ds.X.iloc[te])
        assert pred.shape == (len(te),), name
        assert np.all(np.isfinite(pred)), name
