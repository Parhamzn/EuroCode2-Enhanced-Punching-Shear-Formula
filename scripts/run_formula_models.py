#!/usr/bin/env python
"""Can an explainable, closed-form model beat Eurocode 2 — honestly?

Evaluates grey-box / symbolic / feature-engineered models on the SAME protocol as
the main study (identical folds, physical units), with the researcher-held-out
GroupKFold as the primary, honest bar. Extracts the fitted closed-form equations.

Writes results/formula_*.csv and results/formulas.txt. Run from the repo root:

    python scripts/run_formula_models.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone

from punching_shear import (
    build_formula_models,
    cross_validate_models,
    load_dataset,
    load_space_metrics,
    make_cv,
    oof_predictions,
    paired_error_test,
)
from punching_shear.evaluation import summarize_cv

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SEED = 19
EC2 = "EC2 (refit C_Rd,c)"


def banner(m):
    print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)


def main():
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    ds = load_dataset()
    models = build_formula_models(seed=SEED)
    banner(f"{len(models)} explainable / formula models on {len(ds)} tests "
           f"({ds.groups.nunique()} researchers). Target = stress [MPa].")
    print("Models:", list(models))

    # ---- primary, honest bar: researcher-held-out GroupKFold --------------------
    cv_grp = make_cv("group", n_splits=5)
    grp = summarize_cv(cross_validate_models(models, ds.X, ds.y_stress, cv_grp,
                                             groups=ds.groups)).sort_values("rmse_mean")
    grp.to_csv(RESULTS / "formula_metrics_grouped.csv", index=False)
    print("\n--- Researcher-held-out GroupKFold (the honest bar) -- stress RMSE [MPa] ---")
    ec2_rmse = float(grp.set_index("model").loc[EC2, "rmse_mean"])
    for _, r in grp.iterrows():
        flag = "  <-- EC2" if r["model"] == EC2 else (
            "  ** beats EC2 **" if r["rmse_mean"] < ec2_rmse else "")
        print(f"  {r['model']:34s} RMSE={r['rmse_mean']:.4f} +/- {r['rmse_ci95']:.4f}  "
              f"R2={r['r2_mean']:+.3f}{flag}")

    # ---- random repeated K-fold (optimistic, for contrast) ----------------------
    cv_rand = make_cv("repeated", n_splits=5, n_repeats=2, random_state=SEED)
    rand = summarize_cv(cross_validate_models(models, ds.X, ds.y_stress, cv_rand)
                        ).sort_values("rmse_mean")
    rand.to_csv(RESULTS / "formula_metrics_random.csv", index=False)
    print("\n--- Random repeated 5x2 K-fold (optimistic) -- stress RMSE [MPa] ---")
    for _, r in rand.iterrows():
        print(f"  {r['model']:34s} RMSE={r['rmse_mean']:.4f} +/- {r['rmse_ci95']:.4f}  R2={r['r2_mean']:+.3f}")

    # ---- researcher-held-out OOF: load-space (MN) metrics + significance vs EC2 -
    # Grouped OOF (each lab held out once) so the paired test matches the honest bar.
    cv_oof = make_cv("group", n_splits=5)
    oof = oof_predictions(models, ds.X, ds.y_stress, cv_oof, groups=ds.groups)
    oof.to_csv(RESULTS / "formula_oof_predictions.csv", index=False)
    mn = load_space_metrics(oof, ds.beta, ds.y_load)
    mn.to_csv(RESULTS / "formula_metrics_loadMN.csv", index=False)
    print("\n--- Same predictions in LOAD units [MN] (researcher-held-out OOF) ---")
    for _, r in mn.iterrows():
        print(f"  {r['model']:34s} RMSE={r['rmse']:.4f} MN  MAPE={r['mape_pct']:.1f}%  R2={r['r2']:+.3f}")

    sig = pd.DataFrame([paired_error_test(oof, m, EC2) for m in models if m != EC2])
    sig.to_csv(RESULTS / "formula_paired_vs_ec2.csv", index=False)
    print(f"\n--- Paired Wilcoxon vs {EC2} (stress abs error, researcher-held-out OOF) ---")
    for _, r in sig.iterrows():
        better = r["median_abs_err_diff"] < 0
        verdict = "n.s." if r["p_value"] > 0.05 else ("BETTER than EC2" if better else "worse than EC2")
        print(f"  {r['a']:34s} p={r['p_value']:.3g}  ({verdict})")

    # ---- extract the closed-form equations (fit on all data) --------------------
    banner("Fitted closed-form equations (full data)")
    lines = []
    for name in ["Power-law (d,rho,fck)", "Power-law (+geometry)",
                 "EC2 free-exponent", "EC2 x correction (grey-box)",
                 "Symbolic regression"]:
        est = clone(models[name]).fit(ds.X, ds.y_stress)
        f = est.formula_()
        print(f"  [{name}]\n    {f}")
        lines.append(f"[{name}]\n  {f}\n")
    (RESULTS / "formulas.txt").write_text("\n".join(lines))

    banner(f"Done in {time.time()-t0:.0f}s. EC2 grouped RMSE = {ec2_rmse:.4f} MPa.")


if __name__ == "__main__":
    main()
