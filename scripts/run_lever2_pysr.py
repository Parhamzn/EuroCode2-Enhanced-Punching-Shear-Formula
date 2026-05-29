#!/usr/bin/env python
"""Lever 2: PySR symbolic regression vs EC2 (researcher-held-out, grouped OOF).

Separate from run_levers.py because PySR is slow (Julia backend) and is evaluated
with a single grouped out-of-fold pass (pooled metrics + paired Wilcoxon) rather
than repeated CV. Writes results/levers_pysr_oof.csv and appends the discovered
equations to results/levers_formulas.txt.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from punching_shear import (
    load_dataset,
    make_cv,
    oof_predictions,
    paired_error_test,
    regression_metrics,
)
from punching_shear.eurocode import EC2Regressor
from punching_shear.greybox import PowerLawRegressor
from punching_shear.symbolic import PySRFormulaRegressor, pysr_available

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SEED = 19
EC2 = "EC2 (refit)"


def main():
    if not pysr_available():
        print("PySR backend unavailable -- aborting.")
        return
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    ds = load_dataset()

    models = {
        EC2: EC2Regressor(C_Rdc=None),
        "Power-law (d,rho,fck)": PowerLawRegressor(("d", "rho_l", "fcm_cyl")),
        "PySR x EC2 correction (L2)": PySRFormulaRegressor(
            cols=("col_area", "u0_perim", "d"), mode="correction",
            niterations=30, random_state=SEED),
        "PySR direct (L2)": PySRFormulaRegressor(
            cols=("d", "rho_l", "fcm_cyl"), mode="direct",
            niterations=30, random_state=SEED),
    }

    print(f"Grouped OOF (GroupKFold 5) over {len(models)} models...")
    cv = make_cv("group", n_splits=5)
    oof = oof_predictions(models, ds.X, ds.y_stress, cv, groups=ds.groups)
    oof.to_csv(RESULTS / "levers_pysr_oof_preds.csv", index=False)

    rows = []
    y = oof["y_true"].to_numpy()
    for name in models:
        m = regression_metrics(y, oof[name].to_numpy())
        if name != EC2:
            t = paired_error_test(oof, name, EC2)
            m["p_vs_ec2"] = t["p_value"]
            m["beats_ec2"] = bool(t["median_abs_err_diff"] < 0 and t["p_value"] < 0.05)
        m["model"] = name
        rows.append(m)
    res = pd.DataFrame(rows).set_index("model").sort_values("rmse")
    res.to_csv(RESULTS / "levers_pysr_oof.csv")
    ec2_rmse = res.loc[EC2, "rmse"]
    print(f"\nGrouped OOF pooled metrics (stress [MPa]); EC2 RMSE={ec2_rmse:.4f}")
    for name, r in res.iterrows():
        flag = "  <-- EC2" if name == EC2 else (
            f"  p={r.get('p_vs_ec2', float('nan')):.2g} "
            + ("** beats EC2 **" if r.get("beats_ec2") else "(n.s./worse)"))
        print(f"  {name:30s} RMSE={r['rmse']:.4f}  R2={r['r2']:+.3f}{flag}")

    # Closed-form equations on full data.
    formulas = []
    for name in ["PySR x EC2 correction (L2)", "PySR direct (L2)"]:
        est = models[name].fit(ds.X, ds.y_stress)
        f = est.formula_()
        print(f"\n[{name}]\n  {f}")
        formulas.append(f"[{name}]\n  {f}")
    fpath = RESULTS / "levers_formulas.txt"
    prev = fpath.read_text() if fpath.exists() else ""
    fpath.write_text(prev + "\n\n" + "\n\n".join(formulas))
    print(f"\nDone in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
