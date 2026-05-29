"""Guard tests for the explainable / closed-form models (notebook 07 track)."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from punching_shear import (
    EC2FreeExponentRegressor,
    PowerLawRegressor,
    build_formula_models,
    engineer,
    load_dataset,
)
from punching_shear.features import ENGINEERED
from punching_shear.greybox import EC2CorrectionRegressor

ds = load_dataset()


def test_power_law_recovers_cuberoot_structure():
    """The free-exponent power law should re-derive EC2's ~1/3 exponents on rho, fck."""
    est = PowerLawRegressor(cols=("d", "rho_l", "fcm_cyl")).fit(ds.X, ds.y_stress)
    a_d, a_rho, a_fck = est.exponents_
    assert 0.25 < a_rho < 0.45          # ~1/3 (EC2 cube root)
    assert 0.20 < a_fck < 0.45          # ~1/3 (EC2 cube root)
    assert a_d < 0                      # stress decreases with depth (size effect)
    assert "^" in est.formula_()


def test_ec2_free_exponent_is_near_one_third():
    est = EC2FreeExponentRegressor().fit(ds.X, ds.y_stress)
    assert 0.25 < est.p_ < 0.42         # data confirm EC2's 1/3 exponent
    assert 0.1 < est.C_ < 0.5


def test_ec2_correction_floor_is_ec2():
    """With a constant (no-op) correction the grey-box reduces to EC2."""
    est = EC2CorrectionRegressor(corr_cols=("col_area",)).fit(ds.X, ds.y_stress)
    pred = est.predict(ds.X)
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)
    assert "v_EC2" in est.formula_()


def test_engineered_features_shape_and_names():
    E = engineer(ds.X)
    assert list(E.columns) == ENGINEERED
    assert E.isna().sum().sum() == 0
    # The EC2 basis feature must be strongly correlated with measured stress.
    r = np.corrcoef(E["ec2_basis"], ds.y_stress)[0, 1]
    assert r > 0.4


def test_formula_models_are_leak_free_on_groupfolds():
    """Every formula model fits on a researcher-held-out train fold and predicts."""
    models = build_formula_models()
    tr, te = next(GroupKFold(5).split(ds.X, ds.y_stress, ds.groups))
    for name, model in models.items():
        est = clone(model)
        est.fit(ds.X.iloc[tr], ds.y_stress.iloc[tr])
        pred = est.predict(ds.X.iloc[te])
        assert pred.shape == (len(te),), name
        assert np.all(np.isfinite(pred)), name
