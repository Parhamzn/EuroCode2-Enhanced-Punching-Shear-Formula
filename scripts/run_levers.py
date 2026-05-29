#!/usr/bin/env python
"""Test levers 1-4 for beating EC2 while staying explainable, on the honest bar.

  1. Aggregate size dg (CSCT crack-width) as extra signal  -> dg-complete subset.
  3. CSCT-form closed-form regressor (size via aggregate denominator).
  4. Glass-box additive models: EBM and monotone GAM (full data).
  2. PySR symbolic regression for the EC2 correction factor (if Julia backend ok).

Everything is scored with researcher-held-out GroupKFold (mean +/- 95% CI) and
paired Wilcoxon vs EC2 on grouped out-of-fold errors. Writes results/levers_*.

Run from repo root:  python scripts/run_levers.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from punching_shear import (
    cross_validate_models,
    load_dataset,
    make_cv,
    oof_predictions,
    paired_error_test,
)
from punching_shear.eurocode import EC2Regressor
from punching_shear.evaluation import summarize_cv
from punching_shear.glassbox import STRESS_MONOTONE, GAMRegressor, make_ebm
from punching_shear.greybox import CSCTBasisRegressor, PowerLawRegressor

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SEED = 19
EC2 = "EC2 (refit)"


def banner(m):
    print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)


def report(models, X, y, groups, tag):
    """Grouped-CV summary + paired Wilcoxon vs EC2; returns the summary frame."""
    cv = make_cv("group", n_splits=5)
    summ = summarize_cv(cross_validate_models(models, X, y, cv, groups=groups)
                        ).sort_values("rmse_mean")
    summ.to_csv(RESULTS / f"levers_{tag}_grouped.csv", index=False)
    ec2 = float(summ.set_index("model").loc[EC2, "rmse_mean"])
    oof = oof_predictions(models, X, y, cv, groups=groups)
    print(f"\n  researcher-held-out GroupKFold (stress RMSE [MPa]); EC2={ec2:.4f}")
    for _, r in summ.iterrows():
        sig = ""
        if r["model"] != EC2:
            t = paired_error_test(oof, r["model"], EC2)
            better = t["median_abs_err_diff"] < 0 and t["p_value"] < 0.05
            sig = f"  p={t['p_value']:.2g} " + ("** beats EC2 **" if better else
                   ("(worse)" if t["p_value"] < 0.05 else "(n.s.)"))
        flag = "  <-- EC2" if r["model"] == EC2 else sig
        print(f"    {r['model']:34s} RMSE={r['rmse_mean']:.4f} +/-{r['rmse_ci95']:.4f}"
              f"  R2={r['r2_mean']:+.3f}{flag}")
    return summ


def main():
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    ds = load_dataset()
    formulas = []

    # ===== Levers 1 & 3 : aggregate size dg + CSCT form (dg-complete subset) =====
    banner("Levers 1 & 3 -- aggregate size dg / CSCT form (dg-complete subset)")
    dg = pd.to_numeric(ds.raw["dg"], errors="coerce")
    mask = dg.notna().to_numpy()
    Xa = ds.X[mask].copy()
    Xa["dg"] = dg[mask].to_numpy()
    Xa["d_dg"] = Xa["d"] / (16.0 + Xa["dg"])         # CSCT size/aggregate ratio
    ya = ds.y_stress[mask]
    ga = ds.groups[mask]
    print(f"subset: {len(Xa)} tests, {ga.nunique()} researchers")

    models_a = {
        EC2: EC2Regressor(C_Rdc=None),
        "Power-law (d,rho,fck)": PowerLawRegressor(("d", "rho_l", "fcm_cyl")),
        "Power-law +dg (L1)": PowerLawRegressor(("d", "rho_l", "fcm_cyl", "dg")),
        "Power-law d/(16+dg) (L1)": PowerLawRegressor(("d_dg", "rho_l", "fcm_cyl")),
        "CSCT basis (L3)": CSCTBasisRegressor(),
    }
    report(models_a, Xa, ya, ga, "dg")
    for name in ["Power-law +dg (L1)", "Power-law d/(16+dg) (L1)", "CSCT basis (L3)"]:
        est = models_a[name].fit(Xa, ya)
        formulas.append(f"[{name}]\n  {est.formula_()}")

    # ===== Lever 4 : glass-box additive models (full data) =====
    banner("Lever 4 -- glass-box additive models EBM & monotone GAM (full data)")
    models_b = {
        EC2: EC2Regressor(C_Rdc=None),
        "Power-law (d,rho,fck)": PowerLawRegressor(("d", "rho_l", "fcm_cyl")),
        "EBM (additive)": make_ebm(SEED),
        "GAM (monotone)": GAMRegressor(monotone=STRESS_MONOTONE),
    }
    report(models_b, ds.X, ds.y_stress, ds.groups, "glassbox")
    # Save GAM shape functions (per-feature partial dependence).
    gam = GAMRegressor(monotone=STRESS_MONOTONE).fit(ds.X, ds.y_stress)
    sh = gam.shapes()
    pd.concat({k: pd.DataFrame({"x": v[0], "effect": v[1]}) for k, v in sh.items()},
              names=["feature"]).to_csv(RESULTS / "levers_gam_shapes.csv")

    # ===== Lever 2 : PySR EC2-correction (if the Julia backend is available) =====
    banner("Lever 2 -- PySR symbolic correction factor")
    import os
    from punching_shear.symbolic import PySRFormulaRegressor, pysr_available
    if os.environ.get("PUNCHING_SKIP_PYSR") == "1":
        print("PUNCHING_SKIP_PYSR=1 -- PySR experiment deferred (run separately).")
    elif pysr_available():
        print("PySR backend available -- running (this is slow)...")
        models_c = {
            EC2: EC2Regressor(C_Rdc=None),
            "Power-law (d,rho,fck)": PowerLawRegressor(("d", "rho_l", "fcm_cyl")),
            "PySR x EC2 correction (L2)": PySRFormulaRegressor(
                cols=("col_area", "u0_perim", "d"), mode="correction",
                niterations=30, random_state=SEED),
            "PySR direct (L2)": PySRFormulaRegressor(
                cols=("d", "rho_l", "fcm_cyl"), mode="direct",
                niterations=30, random_state=SEED),
        }
        report(models_c, ds.X, ds.y_stress, ds.groups, "pysr")
        for name in ["PySR x EC2 correction (L2)", "PySR direct (L2)"]:
            est = models_c[name].fit(ds.X, ds.y_stress)
            formulas.append(f"[{name}]\n  {est.formula_()}")
    else:
        print("PySR Julia backend NOT available in this environment -- SKIPPED.\n"
              "  (gplearn symbolic regression in run_formula_models.py is the "
              "available substitute; it does not beat EC2 under grouped CV.)")
        formulas.append("[PySR (L2)]\n  SKIPPED -- Julia backend unavailable.")

    (RESULTS / "levers_formulas.txt").write_text("\n\n".join(formulas))
    banner(f"Done in {time.time()-t0:.0f}s. Formulas -> results/levers_formulas.txt")


if __name__ == "__main__":
    main()
