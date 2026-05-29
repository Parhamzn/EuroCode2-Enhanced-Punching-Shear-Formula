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

# --- cohesive, modern, colour-blind-friendly palette -------------------------
OPTIMISTIC = "#e9c46a"   # muted gold  — random K-fold (optimistic)
HONEST = "#264653"       # deep teal   — researcher-held-out (honest)
EC2_LINE = "#e76f51"     # coral       — EC2 reference line / baseline bar
WIN = "#2a9d8f"          # teal-green  — significantly beats EC2
NS = "#adb5bd"           # muted grey  — not significant
INK = "#222222"
GRID = "#9aa3ab"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Inter", "DejaVu Sans"],
    "font.size": 11.5,
    "axes.titlesize": 14, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.labelsize": 11.5, "axes.labelcolor": INK, "axes.labelweight": "medium",
    "axes.edgecolor": "#555", "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.alpha": 0.22, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "xtick.color": "#333", "ytick.color": "#333", "text.color": INK,
    "ytick.left": False,
})
EBAR = dict(ecolor="#6b6b6b", elinewidth=1.1, capsize=2.6, capthick=1.1)


def short(name: str) -> str:
    return (name.replace(" (refit C_Rd,c)", "").replace(" (d,rho,fck)", "")
            .replace(" (raw)", "").replace(" (grey-box)", "").replace(" (L1)", "")
            .replace(" (L2)", "").replace(" (L3)", "").replace(" feats", ""))


def _top_legend(ax, handles=None, labels=None, ncol=3):
    """Horizontal legend above the axes (never overlaps the bars)."""
    kw = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
              frameon=False, handlelength=1.4, columnspacing=1.8, fontsize=9.5)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


def fig_leakage():
    """Hero: random vs researcher-held-out CV — the leakage lesson."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    r = pd.read_csv(RESULTS / "metrics_random_kfold.csv").set_index("model")
    g = pd.read_csv(RESULTS / "metrics_grouped_kfold.csv").set_index("model")
    keep = [m for m in r.index if m != "SVR (poly-3)"]          # poly-3 overfits, off-scale
    order = r.loc[keep, "rmse_mean"].sort_values().index
    y = np.arange(len(order)); h = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(y + h/2, r.loc[order, "rmse_mean"], h, xerr=r.loc[order, "rmse_ci95"],
            color=OPTIMISTIC, error_kw=EBAR, zorder=3)
    ax.barh(y - h/2, g.loc[order, "rmse_mean"], h, xerr=g.loc[order, "rmse_ci95"],
            color=HONEST, error_kw=EBAR, zorder=3)
    ec2_g = g.loc["EC2 (refit C_Rd,c)", "rmse_mean"]
    ax.axvline(ec2_g, color=EC2_LINE, ls="--", lw=1.8, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([short(m) for m in order])
    ax.invert_yaxis(); ax.set_xlim(0, None); ax.margins(x=0.02)
    ax.set_xlabel("cross-validated RMSE on punching stress  [MPa]   (lower is better)")
    ax.set_title("ML “beats” Eurocode 2 only under leaky random splits", pad=34)
    handles = [Patch(color=OPTIMISTIC, label="random K-fold (optimistic)"),
               Patch(color=HONEST, label="researcher-held-out (honest)"),
               Line2D([0], [0], color=EC2_LINE, ls="--", lw=1.8, label="Eurocode 2 (honest)")]
    _top_legend(ax, handles, [h.get_label() for h in handles], ncol=3)
    fig.text(0.5, -0.02, "Random splits let flexible models memorise lab-specific signal; with whole "
             "laboratories held out the ranking collapses and Eurocode 2 leads.",
             ha="center", va="top", fontsize=9, color="#666", style="italic")
    fig.savefig(ASSETS / "fig_leakage.png"); plt.close(fig)


def fig_size_effect(ds):
    """Why we model stress, not load."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    rl = np.corrcoef(ds.X["d"], ds.y_load)[0, 1]
    rs = np.corrcoef(ds.X["d"], ds.y_stress)[0, 1]
    for a in ax:
        a.spines["left"].set_visible(True)
    ax[0].scatter(ds.X["d"], ds.y_load, s=20, alpha=.6, color=EC2_LINE, edgecolor="none")
    ax[0].set_title(f"WRONG target — load   (r = {rl:+.2f})", pad=8)
    ax[0].set(xlabel="effective depth  d  [mm]", ylabel="failure load  V_test  [MN]")
    ax[1].scatter(ds.X["d"], ds.y_stress, s=20, alpha=.6, color=WIN, edgecolor="none")
    ax[1].set_title(f"RIGHT target — stress   (r = {rs:+.2f})", pad=8)
    ax[1].set(xlabel="effective depth  d  [mm]", ylabel="punching stress  v  [MPa]")
    fig.suptitle("Predicting absolute load just relearns the size effect",
                 fontsize=14, fontweight="semibold")
    fig.savefig(ASSETS / "fig_size_effect.png"); plt.close(fig)


def fig_explainable():
    """Explainable formula models vs EC2 (researcher-held-out), coloured by significance."""
    from matplotlib.patches import Patch
    g = pd.read_csv(RESULTS / "formula_metrics_grouped.csv").set_index("model")
    ec2 = g.loc["EC2 (refit C_Rd,c)", "rmse_mean"]
    sig = pd.read_csv(RESULTS / "formula_paired_vs_ec2.csv").set_index("a")
    beats = {m: (sig.loc[m, "p_value"] < 0.05 and sig.loc[m, "median_abs_err_diff"] < 0)
             for m in sig.index}
    keep = [m for m in g.index if m != "Symbolic regression"]
    order = g.loc[keep, "rmse_mean"].sort_values().index
    colors = [EC2_LINE if m == "EC2 (refit C_Rd,c)" else
              (WIN if beats.get(m, False) else NS) for m in order]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh([short(m) for m in order], g.loc[order, "rmse_mean"],
            xerr=g.loc[order, "rmse_ci95"], color=colors, error_kw=EBAR, zorder=3)
    ax.axvline(ec2, color=EC2_LINE, ls="--", lw=1.6, zorder=2)
    ax.invert_yaxis(); ax.set_xlim(0, None); ax.margins(x=0.02)
    ax.set_xlabel("researcher-held-out RMSE on punching stress  [MPa]")
    ax.set_title("Explainable, closed-form models vs Eurocode 2", pad=34)
    handles = [Patch(color=WIN, label="significantly beats EC2 (p < 0.05)"),
               Patch(color=EC2_LINE, label="Eurocode 2 baseline"),
               Patch(color=NS, label="not significant")]
    _top_legend(ax, handles, [h.get_label() for h in handles], ncol=3)
    fig.text(0.5, -0.02, r"best explainable formula:   "
             r"$v = 1.38 \cdot d^{-0.19} \cdot \rho_l^{0.33} \cdot f_{ck}^{0.31}$  [MPa]   "
             r"($p = 2\times10^{-5}$ vs EC2)",
             ha="center", va="top", fontsize=10.5, color=WIN)
    fig.savefig(ASSETS / "fig_explainable.png"); plt.close(fig)


def fig_pred_vs_measured(ds):
    """OOF predicted vs measured stress: power-law winner vs EC2."""
    oof = pd.read_csv(RESULTS / "formula_oof_predictions.csv")
    pl = [c for c in oof.columns if c.startswith("Power-law (d")][0]
    ec = [c for c in oof.columns if c.startswith("EC2 (refit")][0]
    lim = [0, float(oof["y_true"].max()) * 1.05]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.1), sharex=True, sharey=True,
                           layout="constrained")
    for a, col, name, c in [(ax[0], pl, "Power-law (best explainable)", WIN),
                            (ax[1], ec, "Eurocode 2", EC2_LINE)]:
        a.spines["left"].set_visible(True)
        a.scatter(oof["y_true"], oof[col], s=20, alpha=.6, color=c, edgecolor="none", zorder=3)
        a.plot(lim, lim, color="#444", lw=1.3, zorder=2)
        r2 = 1 - np.sum((oof[col]-oof["y_true"])**2)/np.sum((oof["y_true"]-oof["y_true"].mean())**2)
        a.set_title(f"{name}   (OOF R² = {r2:.2f})", pad=8)
        a.set(xlim=lim, ylim=lim, xlabel="measured stress  [MPa]")
    ax[0].set_ylabel("predicted stress  [MPa]")
    fig.suptitle("Out-of-fold predictions (researcher-held-out)",
                 fontsize=14, fontweight="semibold")
    fig.savefig(ASSETS / "fig_pred_vs_measured.png"); plt.close(fig)


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
