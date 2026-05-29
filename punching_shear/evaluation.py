"""Leak-free evaluation: physical-unit metrics, shared CV folds, paired tests.

Design rules that fix the original study:
  * metrics are computed on the **unscaled** target (shear stress, MPa), so RMSE
    and MAE are interpretable engineering errors -- never a MinMax-scaled number;
  * every model is scored on the **same** folds (one splitter instance reused),
    so the comparison is apples-to-apples;
  * any hyper-parameter tuning lives *inside* the estimator (GridSearchCV), so it
    only ever sees the training part of each outer fold (nested CV);
  * we expose two CV protocols -- a standard repeated K-fold and a
    researcher-held-out GroupKFold -- because the data are clustered by lab and a
    random split leaks lab-specific signal.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    cross_val_predict,
)


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, MAPE (%), R^2 and the model/test mean & std ratios.

    The ratio statistics (mean and std of ``y_pred / y_true``) mirror the
    normalised mean/std the report borrowed from the literature, but here on a
    physically meaningful target.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ratio = y_pred / y_true
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(np.mean(np.abs(err / y_true)) * 100.0),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_std": float(np.std(ratio, ddof=1)),
    }


def make_cv(kind: str = "repeated", n_splits: int = 5, n_repeats: int = 5,
            random_state: int = 19):
    """Build a reusable CV splitter.

    kind="repeated" -> RepeatedKFold (random splits, for the headline estimate);
    kind="group"    -> GroupKFold    (whole researchers held out, OOD estimate).
    """
    if kind == "repeated":
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                             random_state=random_state)
    if kind == "group":
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"unknown cv kind: {kind!r}")


def cross_validate_models(models: Mapping[str, object], X, y, cv, groups=None
                          ) -> pd.DataFrame:
    """Score every model on identical folds; return per-fold metrics (long form).

    Each model is cloned and fit fresh on every training fold (so tuned estimators
    re-tune inside the fold). Metrics are computed per fold on the held-out part in
    physical stress units. Returns a tidy frame with one row per (model, fold).
    """
    from sklearn.base import clone

    X = X.reset_index(drop=True) if hasattr(X, "reset_index") else np.asarray(X)
    y = np.asarray(y, dtype=float)
    g = None if groups is None else np.asarray(groups)

    splits = list(cv.split(X, y, g))
    rows = []
    for name, model in models.items():
        for fold, (tr, te) in enumerate(splits):
            est = clone(model)
            X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
            X_te = X.iloc[te] if hasattr(X, "iloc") else X[te]
            est.fit(X_tr, y[tr])
            pred = est.predict(X_te)
            m = regression_metrics(y[te], pred)
            m.update({"model": name, "fold": fold, "n_test": len(te)})
            rows.append(m)
    return pd.DataFrame(rows)


def summarize_cv(per_fold: pd.DataFrame, metrics=("rmse", "mae", "mape_pct", "r2"),
                 ) -> pd.DataFrame:
    """Mean +/- std and 95% normal CI of the mean across folds, per model."""
    out = []
    for name, grp in per_fold.groupby("model", sort=False):
        row = {"model": name, "n_folds": len(grp)}
        for met in metrics:
            vals = grp[met].to_numpy()
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            sem = std / np.sqrt(len(vals))
            row[f"{met}_mean"] = mean
            row[f"{met}_std"] = std
            row[f"{met}_ci95"] = 1.96 * sem
        out.append(row)
    return pd.DataFrame(out)


def oof_predictions(models: Mapping[str, object], X, y, cv, groups=None
                    ) -> pd.DataFrame:
    """Out-of-fold predictions (one per sample) for every model on shared folds.

    Used for paired significance tests and a single honest pooled R^2. Pass a
    *non-repeated* splitter (KFold or GroupKFold) so each sample is predicted once.
    """
    y = np.asarray(y, dtype=float)
    g = None if groups is None else np.asarray(groups)
    preds = {"y_true": y}
    for name, model in models.items():
        preds[name] = cross_val_predict(model, X, y, cv=cv, groups=g)
    return pd.DataFrame(preds)


def paired_error_test(oof: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Wilcoxon signed-rank test on per-sample absolute OOF errors (A vs B).

    Negative ``median_diff`` means model_a has the smaller typical error.
    """
    y = oof["y_true"].to_numpy()
    ea = np.abs(oof[model_a].to_numpy() - y)
    eb = np.abs(oof[model_b].to_numpy() - y)
    diff = ea - eb
    nz = diff[diff != 0]
    if len(nz) == 0:  # pragma: no cover - identical predictions
        return {"a": model_a, "b": model_b, "p_value": 1.0, "median_diff": 0.0, "n": 0}
    stat, p = stats.wilcoxon(ea, eb)
    return {
        "a": model_a,
        "b": model_b,
        "statistic": float(stat),
        "p_value": float(p),
        "median_abs_err_diff": float(np.median(diff)),
        "a_better_frac": float(np.mean(ea < eb)),
        "n": int(len(y)),
    }
