#!/usr/bin/env python
"""End-to-end corrected punching-shear analysis.

Produces the headline comparison the original study should have produced:
  * target = punching shear STRESS [MPa] (not size-confounded load),
  * EC2 baseline (refit per fold) and every ML model on IDENTICAL folds,
  * metrics in physical units with confidence intervals,
  * a random K-fold estimate AND a researcher-held-out (GroupKFold) estimate,
  * paired Wilcoxon significance tests vs the best model and vs EC2,
  * permutation feature importance.

Writes CSVs and figures to ``results/``. Run from the repo root:

    python scripts/run_analysis.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from punching_shear import (
    FEATURE_LABELS,
    build_models,
    cross_validate_models,
    load_dataset,
    make_cv,
    oof_predictions,
    paired_error_test,
)
from punching_shear.evaluation import summarize_cv
from punching_shear.models import fit_full

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SEED = 19


def banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)


def size_effect_table(ds) -> pd.DataFrame:
    """Show why predicting load is the wrong target: d-correlation, load vs stress."""
    rows = []
    for tgt, name in [(ds.y_load, "load V_test [MN]"), (ds.y_stress, "stress v [MPa]")]:
        for feat in ds.feature_names:
            r = float(np.corrcoef(ds.X[feat], tgt)[0, 1])
            rows.append({"target": name, "feature": feat, "pearson_r": r, "r2_single": r ** 2})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    ds = load_dataset()
    banner(f"Dataset: {len(ds)} tests, {ds.groups.nunique()} researchers, "
           f"5 features. Target = punching shear stress [MPa].")

    # ---- 1. The size-effect artifact -------------------------------------------------
    se = size_effect_table(ds)
    se.to_csv(RESULTS / "size_effect_correlations.csv", index=False)
    piv = se.pivot(index="feature", columns="target", values="r2_single").round(3)
    print("Single-feature R^2 (why load is the wrong target):")
    print(piv.to_string())
    print(f"\n-> corr(d, load)={np.corrcoef(ds.X['d'], ds.y_load)[0,1]:+.3f} "
          f"but corr(d, stress)={np.corrcoef(ds.X['d'], ds.y_stress)[0,1]:+.3f}: "
          "regressing load on d relearns the trivial size effect.")

    # ---- 2. Models -------------------------------------------------------------------
    models = build_models(seed=SEED)
    banner(f"Evaluating {len(models)} models on identical folds: {list(models)}")

    # ---- 3a. Random repeated K-fold (headline estimate) ------------------------------
    cv_rand = make_cv("repeated", n_splits=5, n_repeats=5, random_state=SEED)
    per_fold_rand = cross_validate_models(models, ds.X, ds.y_stress, cv_rand)
    summ_rand = summarize_cv(per_fold_rand).sort_values("rmse_mean")
    summ_rand.to_csv(RESULTS / "metrics_random_kfold.csv", index=False)
    print("\nRepeated 5x5 K-fold -- RMSE [MPa] mean +/- 95% CI (sorted, lower=better):")
    for _, r in summ_rand.iterrows():
        print(f"  {r['model']:22s} RMSE={r['rmse_mean']:.4f} +/- {r['rmse_ci95']:.4f}"
              f"   MAE={r['mae_mean']:.4f}  R2={r['r2_mean']:+.3f}  MAPE={r['mape_pct_mean']:.1f}%")

    # ---- 3b. Researcher-held-out GroupKFold (honest OOD estimate) --------------------
    cv_grp = make_cv("group", n_splits=5)
    per_fold_grp = cross_validate_models(models, ds.X, ds.y_stress, cv_grp, groups=ds.groups)
    summ_grp = summarize_cv(per_fold_grp).sort_values("rmse_mean")
    summ_grp.to_csv(RESULTS / "metrics_grouped_kfold.csv", index=False)
    print("\nResearcher-held-out GroupKFold -- RMSE [MPa] mean +/- 95% CI:")
    for _, r in summ_grp.iterrows():
        print(f"  {r['model']:22s} RMSE={r['rmse_mean']:.4f} +/- {r['rmse_ci95']:.4f}"
              f"   R2={r['r2_mean']:+.3f}")

    # ---- 4. Significance vs best & vs EC2 (single 10-fold OOF) -----------------------
    cv_oof = make_cv("repeated", n_splits=10, n_repeats=1, random_state=SEED)
    oof = oof_predictions(models, ds.X, ds.y_stress, cv_oof)
    oof.to_csv(RESULTS / "oof_predictions.csv", index=False)
    best = summ_rand.iloc[0]["model"]
    sig_rows = []
    for name in models:
        if name == best:
            continue
        sig_rows.append(paired_error_test(oof, best, name))
    if "EC2 (refit C_Rd,c)" in models and best != "EC2 (refit C_Rd,c)":
        for name in models:
            if name not in (best, "EC2 (refit C_Rd,c)"):
                sig_rows.append(paired_error_test(oof, "EC2 (refit C_Rd,c)", name))
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(RESULTS / "paired_significance.csv", index=False)
    print(f"\nPaired Wilcoxon vs best model ({best}) on per-sample abs error:")
    for _, r in sig[sig["a"] == best].iterrows():
        verdict = "n.s." if r["p_value"] > 0.05 else ("BETTER" if r["median_abs_err_diff"] < 0 else "worse")
        print(f"  {best} vs {r['b']:22s} p={r['p_value']:.3g}  ({verdict})")

    # ---- 5. Permutation feature importance (best ML model, on stress) ----------------
    ml_models = {k: v for k, v in models.items() if k != "EC2 (refit C_Rd,c)"}
    best_ml = summ_rand[summ_rand["model"].isin(ml_models)].iloc[0]["model"]
    fitted = fit_full(models[best_ml], ds.X, ds.y_stress)
    pi = permutation_importance(fitted, ds.X, ds.y_stress, n_repeats=30,
                                random_state=SEED, scoring="neg_mean_squared_error")
    imp = (pd.DataFrame({"feature": ds.feature_names,
                         "importance": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("importance", ascending=False))
    imp.to_csv(RESULTS / "feature_importance.csv", index=False)
    print(f"\nPermutation importance ({best_ml}, target=stress) -- "
          "does column geometry still vanish?")
    print(imp.to_string(index=False))

    # ---- 6. Figures ------------------------------------------------------------------
    make_figures(ds, summ_rand, oof, imp, best, best_ml, se)

    summary = {
        "n": len(ds), "n_researchers": int(ds.groups.nunique()),
        "best_overall": best, "best_ml": best_ml,
        "random_kfold_rmse": {r["model"]: round(r["rmse_mean"], 4)
                              for _, r in summ_rand.iterrows()},
        "grouped_kfold_rmse": {r["model"]: round(r["rmse_mean"], 4)
                               for _, r in summ_grp.iterrows()},
        "runtime_s": round(time.time() - t0, 1),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    banner(f"Done in {summary['runtime_s']}s. Wrote results/ "
           f"(best overall: {best}; best ML: {best_ml}).")


def make_figures(ds, summ_rand, oof, imp, best, best_ml, se) -> None:
    # 6a. RMSE comparison with CI
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = summ_rand.sort_values("rmse_mean")
    ax.barh(s["model"], s["rmse_mean"], xerr=s["rmse_ci95"],
            color=["#c44" if m.startswith("EC2") else "#48c" for m in s["model"]])
    ax.set_xlabel("CV RMSE on punching stress [MPa]  (lower = better)")
    ax.set_title("Model comparison (repeated 5x5 K-fold, identical folds)")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(RESULTS / "fig_model_comparison.png", dpi=130); plt.close(fig)

    # 6b. Predicted vs measured stress: best model vs EC2
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    lim = [0, float(ds.y_stress.max()) * 1.05]
    for ax, name in zip(axes, [best, "EC2 (refit C_Rd,c)"]):
        if name not in oof:
            continue
        ax.scatter(ds.y_stress, oof[name], s=14, alpha=0.55, edgecolor="none")
        ax.plot(lim, lim, "k-", lw=1.5)
        r = float(np.corrcoef(ds.y_stress, oof[name])[0, 1])
        ax.set_title(f"{name}  (OOF R²={r**2:.3f})")
        ax.set_xlabel("measured stress [MPa]"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("predicted stress [MPa]")
    fig.suptitle("Out-of-fold predicted vs measured punching stress")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_pred_vs_measured.png", dpi=130); plt.close(fig)

    # 6c. Permutation importance
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([FEATURE_LABELS.get(f, f) for f in imp["feature"]], imp["importance"],
            xerr=imp["std"], color="#4a8")
    ax.set_xlabel("permutation importance (Δ MSE)")
    ax.set_title(f"Feature importance on STRESS target ({best_ml})")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(RESULTS / "fig_feature_importance.png", dpi=130); plt.close(fig)

    # 6d. The size-effect artifact: d vs load and d vs stress
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(ds.X["d"], ds.y_load, s=14, alpha=0.5, color="#c44")
    axes[0].set_title(f"load vs d (r={np.corrcoef(ds.X['d'], ds.y_load)[0,1]:+.2f}) — confounded")
    axes[0].set_xlabel("effective depth d [mm]"); axes[0].set_ylabel("V_test [MN]")
    axes[1].scatter(ds.X["d"], ds.y_stress, s=14, alpha=0.5, color="#48c")
    axes[1].set_title(f"stress vs d (r={np.corrcoef(ds.X['d'], ds.y_stress)[0,1]:+.2f}) — the real signal")
    axes[1].set_xlabel("effective depth d [mm]"); axes[1].set_ylabel("v [MPa]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Why target choice matters: absolute load is dominated by the size effect")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_size_effect.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
