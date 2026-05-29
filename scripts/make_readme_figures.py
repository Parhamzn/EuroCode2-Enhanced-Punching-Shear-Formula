#!/usr/bin/env python
"""Generate polished figures for the GitHub README into assets/.

Reads the CSVs written by run_analysis.py / run_formula_models.py / run_levers.py
/ run_lever2_pysr.py (so run those first) and the dataset itself. Run:

    python scripts/run_analysis.py && python scripts/run_formula_models.py
    python scripts/run_levers.py && python scripts/run_lever2_pysr.py   # optional (PySR)
    python scripts/make_readme_figures.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from punching_shear import load_dataset

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
ASSETS = REPO / "assets"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})
EC2_C, ML_C, WIN_C, NS_C = "#d1495b", "#3a7ca5", "#2a9d4a", "#9aa0a6"


def short(name: str) -> str:
    return (name.replace(" (refit C_Rd,c)", "").replace(" (d,rho,fck)", "")
            .replace(" (raw)", "").replace(" (grey-box)", "").replace(" (L1)", "")
            .replace(" (L2)", "").replace(" (L3)", "").replace(" feats", ""))


def fig_leakage():
    """Hero: random vs researcher-held-out CV — the leakage lesson."""
    r = pd.read_csv(RESULTS / "metrics_random_kfold.csv").set_index("model")
    g = pd.read_csv(RESULTS / "metrics_grouped_kfold.csv").set_index("model")
    keep = [m for m in r.index if m != "SVR (poly-3)"]          # poly-3 overfits, off-scale
    order = r.loc[keep, "rmse_mean"].sort_values().index
    y = np.arange(len(order)); h = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(y + h/2, r.loc[order, "rmse_mean"], h, xerr=r.loc[order, "rmse_ci95"],
            color=ML_C, label="random K-fold (optimistic)")
    ax.barh(y - h/2, g.loc[order, "rmse_mean"], h, xerr=g.loc[order, "rmse_ci95"],
            color="#e0a458", label="researcher-held-out (honest)")
    ax.axvline(g.loc["EC2 (refit C_Rd,c)", "rmse_mean"], color=EC2_C, ls="--", lw=1.6,
               label="EC2 (honest)")
    ax.set_yticks(y); ax.set_yticklabels([short(m) for m in order]); ax.invert_yaxis()
    ax.set_xlabel("CV RMSE on punching stress  [MPa]   (lower = better)")
    ax.set_title("ML 'beats' Eurocode 2 only under leaky random splits")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    fig.text(0.5, 0.015, "Random splits let flexible models memorise lab-specific signal; with whole "
             "labs held out the ranking collapses and EC2 leads.",
             ha="center", va="bottom", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(ASSETS / "fig_leakage.png"); plt.close(fig)


def fig_size_effect(ds):
    """Why we model stress, not load."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    rl = np.corrcoef(ds.X["d"], ds.y_load)[0, 1]
    rs = np.corrcoef(ds.X["d"], ds.y_stress)[0, 1]
    ax[0].scatter(ds.X["d"], ds.y_load, s=16, alpha=.55, color=EC2_C, edgecolor="none")
    ax[0].set(title=f"WRONG target: load  (r = {rl:+.2f})",
              xlabel="effective depth  d  [mm]", ylabel="failure load  V_test  [MN]")
    ax[1].scatter(ds.X["d"], ds.y_stress, s=16, alpha=.55, color=ML_C, edgecolor="none")
    ax[1].set(title=f"RIGHT target: stress  (r = {rs:+.2f})",
              xlabel="effective depth  d  [mm]", ylabel="punching stress  v  [MPa]")
    fig.suptitle("Predicting absolute load just relearns the size effect",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(ASSETS / "fig_size_effect.png"); plt.close(fig)


def fig_explainable():
    """Explainable formula models vs EC2 (researcher-held-out R²), winners highlighted."""
    g = pd.read_csv(RESULTS / "formula_metrics_grouped.csv").set_index("model")
    ec2 = g.loc["EC2 (refit C_Rd,c)", "rmse_mean"]
    # Significance vs EC2 from the paired Wilcoxon (grouped OOF), NOT just RMSE order.
    sig = pd.read_csv(RESULTS / "formula_paired_vs_ec2.csv").set_index("a")
    beats = {m: (sig.loc[m, "p_value"] < 0.05 and sig.loc[m, "median_abs_err_diff"] < 0)
             for m in sig.index}
    keep = [m for m in g.index if m != "Symbolic regression"]
    order = g.loc[keep, "rmse_mean"].sort_values().index
    colors = [EC2_C if m == "EC2 (refit C_Rd,c)" else
              (WIN_C if beats.get(m, False) else NS_C) for m in order]
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.barh([short(m) for m in order], g.loc[order, "rmse_mean"],
            xerr=g.loc[order, "rmse_ci95"], color=colors)
    ax.axvline(ec2, color=EC2_C, ls="--", lw=1.4)
    ax.invert_yaxis()
    ax.set_xlabel("researcher-held-out RMSE on stress  [MPa]")
    ax.set_title("Explainable, closed-form models vs Eurocode 2")
    ax.legend(handles=[Patch(color=WIN_C, label="significantly beats EC2 (p<0.05)"),
                       Patch(color=EC2_C, label="EC2 baseline"),
                       Patch(color=NS_C, label="not significant")],
              loc="lower right", framealpha=0.9, fontsize=9)
    fig.text(0.5, 0.015, r"best explainable formula:  "
             r"$v = 1.38\,d^{-0.19}\,\rho_l^{0.33}\,f_{ck}^{0.31}$  [MPa]   (p = 2e-5 vs EC2)",
             ha="center", va="bottom", fontsize=10, color=WIN_C)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(ASSETS / "fig_explainable.png"); plt.close(fig)


def fig_pred_vs_measured(ds):
    """OOF predicted vs measured stress: power-law winner vs EC2."""
    oof = pd.read_csv(RESULTS / "formula_oof_predictions.csv")
    pl = [c for c in oof.columns if c.startswith("Power-law (d")][0]
    ec = [c for c in oof.columns if c.startswith("EC2 (refit")][0]
    lim = [0, float(oof["y_true"].max()) * 1.05]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for a, col, name, c in [(ax[0], pl, "Power-law (winner)", WIN_C),
                            (ax[1], ec, "Eurocode 2", EC2_C)]:
        a.scatter(oof["y_true"], oof[col], s=16, alpha=.55, color=c, edgecolor="none")
        a.plot(lim, lim, "k-", lw=1.3)
        r2 = 1 - np.sum((oof[col]-oof["y_true"])**2)/np.sum((oof["y_true"]-oof["y_true"].mean())**2)
        a.set(title=f"{name}   (OOF R² = {r2:.2f})", xlim=lim, ylim=lim,
              xlabel="measured stress  [MPa]")
    ax[0].set_ylabel("predicted stress  [MPa]")
    fig.suptitle("Out-of-fold predictions (researcher-held-out)", fontweight="bold")
    fig.tight_layout(); fig.savefig(ASSETS / "fig_pred_vs_measured.png"); plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    ds = load_dataset()
    fig_leakage()
    fig_size_effect(ds)
    fig_explainable()
    fig_pred_vs_measured(ds)
    print("Wrote:", *[p.name for p in sorted(ASSETS.glob("*.png"))])


if __name__ == "__main__":
    main()
