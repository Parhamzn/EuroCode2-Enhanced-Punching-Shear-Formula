#!/usr/bin/env python
"""Generate the six clean analysis notebooks from the punching_shear package.

Heavy cross-validation numbers are loaded from ``results/`` (produced by
``run_analysis.py``); the notebooks themselves do light, fast, leak-free demos so
they execute in a couple of minutes. Run:

    python scripts/build_notebooks.py     # writes notebooks/*.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

BOOT = (
    "import sys, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "from pathlib import Path\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import punching_shear as ps\n"
    "REPO = Path(ps.__file__).resolve().parent.parent\n"
    "RESULTS = REPO / 'results'\n"
    "ds = ps.load_dataset()\n"
    "print(f'{len(ds)} tests, {ds.groups.nunique()} researchers, features={ds.feature_names}')"
)


def nb(*cells):
    n = new_notebook()
    n.cells = list(cells)
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return n


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


# ---------------------------------------------------------------------------
# 01 — Data overview & the target problem
# ---------------------------------------------------------------------------
nb01 = nb(
    md("# 01 · Data overview & the target problem\n\n"
       "The Siburg compilation: **336** flat-slab punching tests from **55** labs. "
       "This notebook does the EDA *and* establishes the single most important "
       "decision in the rebuild: **what to predict.**\n\n"
       "The original study predicted the absolute failure **load** `V_test` [MN]. "
       "But punching capacity is `V = v · u₁ · d` — load is mechanically "
       "proportional to the control area `u₁·d`. So regressing load on the "
       "effective depth `d` mostly relearns a trivial size effect. We instead "
       "predict the punching **stress** `v = V/(u₁·d)` [MPa]."),
    code(BOOT),
    md("### Features and targets"),
    code("display(ds.X.describe().T.round(2))\n"
         "print('Primary target  v_test [MPa]:', ds.y_stress.describe()[['min','50%','max']].round(3).to_dict())\n"
         "print('Original target V_test [MN]:', ds.y_load.describe()[['min','50%','max']].round(3).to_dict())"),
    md("### Missingness (why we model only 5 features)\n"
       "`dg`, `fym`, `Esm`, `c2` are heavily missing, so they are dropped; the five "
       "modelling features are fully observed."),
    code("raw = ds.raw\n"
         "miss = raw.isna().mean().sort_values(ascending=False)\n"
         "miss = miss[miss > 0]\n"
         "ax = miss.plot.bar(figsize=(9,3), color='#c66'); ax.set_ylabel('fraction missing')\n"
         "ax.set_title('Missingness by column'); plt.tight_layout(); plt.show()\n"
         "print('core features missing:', int(ds.X.isna().sum().sum()))"),
    md("### Collinearity"),
    code("import seaborn as sns\n"
         "M = ds.X.copy(); M['v_test']=ds.y_stress; M['V_test']=ds.y_load\n"
         "plt.figure(figsize=(7,5)); sns.heatmap(M.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)\n"
         "plt.title('Pearson correlation'); plt.tight_layout(); plt.show()"),
    md("## The central issue: load vs stress\n"
       "Single-feature R² of each predictor against **load** and against **stress**."),
    code("rows=[]\n"
         "for tgt,name in [(ds.y_load,'load V_test'),(ds.y_stress,'stress v')]:\n"
         "    for f in ds.feature_names:\n"
         "        r=np.corrcoef(ds.X[f],tgt)[0,1]; rows.append({'feature':f,'target':name,'R2':r**2})\n"
         "tab=pd.DataFrame(rows).pivot(index='feature',columns='target',values='R2').round(3)\n"
         "display(tab)\n"
         "fig,ax=plt.subplots(1,2,figsize=(11,4))\n"
         "ax[0].scatter(ds.X['d'],ds.y_load,s=12,alpha=.5,c='#c44'); ax[0].set_title(f\"load vs d  (r={np.corrcoef(ds.X['d'],ds.y_load)[0,1]:+.2f}) — confounded\"); ax[0].set_xlabel('d [mm]'); ax[0].set_ylabel('V_test [MN]')\n"
         "ax[1].scatter(ds.X['d'],ds.y_stress,s=12,alpha=.5,c='#48c'); ax[1].set_title(f\"stress vs d  (r={np.corrcoef(ds.X['d'],ds.y_stress)[0,1]:+.2f}) — real signal\"); ax[1].set_xlabel('d [mm]'); ax[1].set_ylabel('v [MPa]')\n"
         "[a.grid(alpha=.3) for a in ax]; plt.tight_layout(); plt.show()"),
    md("**Takeaway.** Against **load**, `d` looks all-important (R²≈0.80). Against "
       "**stress**, `d` explains almost nothing (R²≈0.07) and the real drivers are "
       "the reinforcement ratio `rho_l` and concrete strength `fcm_cyl` — exactly "
       "what punching mechanics predicts. Everything downstream targets stress."),
)

# ---------------------------------------------------------------------------
# 02 — Eurocode 2 baseline (in stress space)
# ---------------------------------------------------------------------------
nb02 = nb(
    md("# 02 · Eurocode 2 baseline\n\n"
       "EC2 punching resistance of a slab without shear reinforcement is a **stress**:\n\n"
       "$$v_{Rd,c}=C_{Rd,c}\\,k\\,(100\\,\\rho_l f_{ck})^{1/3},\\quad k=1+\\sqrt{200/d}\\le2,\\ \\rho_l\\le0.02$$\n\n"
       "We (1) implement it, (2) validate it reproduces the spreadsheet `v_Rd`, "
       "(3) refit `C_Rd,c` to the measured stress to remove the built-in safety "
       "margin, giving a fair functional-form baseline for the ML models."),
    code(BOOT),
    md("### Validate our formula against the spreadsheet's `v_Rd` (design $C=0.12$)"),
    code("comp = pd.read_excel(REPO/'data'/'Data.xlsx'); comp.columns=[str(c) for c in comp.columns]\n"
         "v_sheet = pd.to_numeric(comp['v_Rd'],errors='coerce').to_numpy()\n"
         "d,rho,fck = ds.ec2_inputs()\n"
         "v_ours = ps.ec2_stress(d,rho,fck,C_Rdc=ps.CRD_C_DESIGN,apply_caps=True)\n"
         "rel = np.abs(v_ours-v_sheet)/v_sheet\n"
         "print(f'median rel. error vs spreadsheet v_Rd = {np.median(rel):.3e}  (max abs = {np.max(np.abs(v_ours-v_sheet)):.3f} MPa)')"),
    md("### Refit `C_Rd,c` on measured stress and compare"),
    code("C = ps.refit_CRdc(d,rho,fck,ds.y_stress)\n"
         "print(f'refit C_Rd,c = {C:.4f}  (vs characteristic 0.18, design 0.12)')\n"
         "pred = ps.ec2_stress(d,rho,fck,C_Rdc=C)\n"
         "m = ps.regression_metrics(ds.y_stress, pred)\n"
         "print('in-sample RMSE [MPa]=%.3f  R2=%.3f'%(m['rmse'],m['r2']))\n"
         "lim=[0,ds.y_stress.max()*1.05]\n"
         "plt.figure(figsize=(5,5)); plt.scatter(ds.y_stress,pred,s=12,alpha=.5)\n"
         "plt.plot(lim,lim,'k-'); plt.xlim(lim); plt.ylim(lim); plt.grid(alpha=.3)\n"
         "plt.xlabel('measured stress [MPa]'); plt.ylabel('EC2 predicted stress [MPa]'); plt.title(f'EC2 (refit C={C:.3f})'); plt.show()"),
    md("### The real safety factor\n"
       "The README/report quote a safety factor of ≈1.8. The data say otherwise:"),
    code("SF = (ds.y_load / (pd.to_numeric(comp['V_Rd'],errors='coerce'))).dropna()\n"
         "print(f'mean V_test/V_Rd (design) = {SF.mean():.2f} ± {SF.std():.2f}  (min {SF.min():.2f}, max {SF.max():.2f})')\n"
         "print(f'against characteristic resistance (×1.5): {SF.mean()/1.5:.2f}')\n"
         "SF.plot.hist(bins=30, figsize=(7,3), color='#69b'); plt.axvline(SF.mean(),c='r'); plt.xlabel('V_test / V_Rd'); plt.title('Observed safety factor'); plt.show()"),
    md("**Takeaway.** Our EC2 stress formula reproduces the spreadsheet exactly "
       "(median error ~1e-7). The genuine mean safety factor is **2.28**, not 1.8. "
       "Refitting `C_Rd,c`≈0.26 removes the safety margin so EC2 competes on "
       "functional form alone in the model comparison."),
)

# ---------------------------------------------------------------------------
# 03 — Regression family (leak-free)
# ---------------------------------------------------------------------------
nb03 = nb(
    md("# 03 · Regression family (leak-free)\n\n"
       "OLS, a non-linear OLS with a `d×fcm` interaction, and Lasso/Ridge/ElasticNet "
       "— all as **pipelines** (scaler refit inside each fold) with hyper-parameters "
       "tuned by nested CV. We show a single leak-free split for intuition, then the "
       "honest cross-validated numbers from `results/`."),
    code(BOOT),
    md("### Single leak-free 70/30 split (illustration)"),
    code("from sklearn.model_selection import train_test_split\n"
         "from sklearn.base import clone\n"
         "models = ps.build_models()\n"
         "reg = ['OLS','NLR (d x fcm)','Lasso','Ridge','ElasticNet']\n"
         "Xtr,Xte,ytr,yte = train_test_split(ds.X, ds.y_stress, test_size=.3, random_state=19)\n"
         "out=[]\n"
         "for name in reg:\n"
         "    est=clone(models[name]).fit(Xtr,ytr)\n"
         "    m=ps.regression_metrics(yte, est.predict(Xte)); m['model']=name; out.append(m)\n"
         "display(pd.DataFrame(out)[['model','rmse','mae','r2','mape_pct']].round(3))"),
    md("### Standardized OLS coefficients (interpretable importance)\n"
       "Because features are standardized in the pipeline, coefficient magnitude is a "
       "fair importance measure — on the **stress** target."),
    code("ols = clone(models['OLS']).fit(ds.X, ds.y_stress)\n"
         "coef = pd.Series(ols.named_steps['lr'].coef_, index=ds.feature_names).sort_values(key=abs, ascending=False)\n"
         "display(coef.round(3).to_frame('std. coef'))\n"
         "coef.plot.barh(figsize=(7,3), color='#48c'); plt.gca().invert_yaxis(); plt.title('OLS standardized coefficients (stress)'); plt.tight_layout(); plt.show()"),
    md("### Honest cross-validated comparison (from `results/`)"),
    code("rand = pd.read_csv(RESULTS/'metrics_random_kfold.csv')\n"
         "cols=['model','rmse_mean','rmse_ci95','r2_mean','mape_pct_mean']\n"
         "display(rand[rand.model.isin(reg)][cols].round(4))"),
    md("**Takeaway.** Regularization (Lasso/Ridge/ElasticNet) does **not** beat plain "
       "OLS here — with only 5 well-conditioned features there is little to shrink. "
       "The `d×fcm` interaction gives a marginal gain. All linear models land around "
       "RMSE ≈ 0.29 MPa (R² ≈ 0.68)."),
)

# ---------------------------------------------------------------------------
# 04 — Support Vector Regression
# ---------------------------------------------------------------------------
nb04 = nb(
    md("# 04 · Support Vector Regression\n\n"
       "Linear, degree-3 polynomial and RBF kernels. Hyper-parameters are tuned by "
       "`GridSearchCV` **on the training split only** (never on the test set, unlike "
       "the original), then evaluated once on the held-out 30%."),
    code(BOOT),
    code("from sklearn.model_selection import train_test_split\n"
         "from sklearn.base import clone\n"
         "models = ps.build_models()\n"
         "svr = ['SVR (linear)','SVR (poly-3)','SVR (RBF)']\n"
         "Xtr,Xte,ytr,yte = train_test_split(ds.X, ds.y_stress, test_size=.3, random_state=19)\n"
         "out=[]\n"
         "for name in svr:\n"
         "    gs=clone(models[name]).fit(Xtr,ytr)\n"
         "    best=gs.best_estimator_.named_steps['m']\n"
         "    nsv=int(best.support_.shape[0])\n"
         "    m=ps.regression_metrics(yte, gs.predict(Xte))\n"
         "    out.append({'model':name,'rmse':m['rmse'],'r2':m['r2'],'n_SV':nsv,'best':gs.best_params_})\n"
         "display(pd.DataFrame(out))"),
    md("### Cross-validated SVR numbers (from `results/`)"),
    code("rand = pd.read_csv(RESULTS/'metrics_random_kfold.csv')\n"
         "display(rand[rand.model.isin(svr)][['model','rmse_mean','rmse_ci95','r2_mean']].round(4))"),
    md("**Takeaway.** The **RBF** kernel is the strongest SVR (CV R² ≈ 0.75) and the "
       "only one competitive with Random Forest. The **polynomial** kernel "
       "overfits badly (negative R²). But note: this is on *random* splits — "
       "see notebook 06 for what happens under researcher-held-out validation."),
)

# ---------------------------------------------------------------------------
# 05 — Trees, forest & feature importance
# ---------------------------------------------------------------------------
nb05 = nb(
    md("# 05 · Decision tree, random forest & feature importance\n\n"
       "Tuned CART and a Random Forest, and the **permutation feature importance** "
       "that overturns the original study's headline."),
    code(BOOT),
    code("from sklearn.model_selection import train_test_split\n"
         "from sklearn.base import clone\n"
         "models = ps.build_models()\n"
         "tree = ['Decision Tree','Random Forest']\n"
         "Xtr,Xte,ytr,yte = train_test_split(ds.X, ds.y_stress, test_size=.3, random_state=19)\n"
         "out=[]\n"
         "for name in tree:\n"
         "    gs=clone(models[name]).fit(Xtr,ytr)\n"
         "    m=ps.regression_metrics(yte, gs.predict(Xte)); m['model']=name; m['best']=gs.best_params_; out.append(m)\n"
         "display(pd.DataFrame(out)[['model','rmse','r2','mape_pct','best']])"),
    md("### Permutation importance on the STRESS target\n"
       "Original study (load target): *“`d` dominates, column profile is irrelevant.”* "
       "On the corrected stress target:"),
    code("imp = pd.read_csv(RESULTS/'feature_importance.csv')\n"
         "display(imp.round(4))\n"
         "plt.figure(figsize=(7,3)); plt.barh(imp.feature, imp.importance, xerr=imp['std'], color='#4a8')\n"
         "plt.gca().invert_yaxis(); plt.xlabel('permutation importance (Δ MSE)'); plt.title('Random Forest, stress target'); plt.tight_layout(); plt.show()"),
    md("**Takeaway.** On the stress target the dominant features are **`rho_l`** "
       "(reinforcement ratio) and **`fcm_cyl`** (concrete strength) — the actual "
       "mechanical drivers — while `d` is minor. The original *“`d` dominates”* "
       "conclusion was a load/size-effect artifact. However, the column-geometry "
       "features (`col_area`) remain negligible, so the original conclusion that "
       "**column profile can be dropped** does survive the correction."),
)

# ---------------------------------------------------------------------------
# 06 — The comparison & the leakage lesson
# ---------------------------------------------------------------------------
nb06 = nb(
    md("# 06 · Model comparison — and the leakage lesson\n\n"
       "The payoff. All 11 models (incl. the EC2 baseline) were scored on **identical "
       "folds**, with metrics in physical units (MPa). We compare two validation "
       "protocols:\n\n"
       "1. **Random** repeated 5×5 K-fold — what the original study effectively used.\n"
       "2. **Researcher-held-out** GroupKFold — whole labs held out, the honest test "
       "of generalization to a *new* experiment."),
    code(BOOT),
    code("rand = pd.read_csv(RESULTS/'metrics_random_kfold.csv').set_index('model')\n"
         "grp  = pd.read_csv(RESULTS/'metrics_grouped_kfold.csv').set_index('model')\n"
         "cmp = pd.DataFrame({'RMSE random':rand.rmse_mean,'R2 random':rand.r2_mean,\n"
         "                    'RMSE grouped':grp.rmse_mean,'R2 grouped':grp.r2_mean}).round(3)\n"
         "cmp = cmp.sort_values('RMSE random')\n"
         "display(cmp)"),
    code("fig,ax=plt.subplots(figsize=(9,5))\n"
         "order=cmp.index\n"
         "y=np.arange(len(order)); w=.4\n"
         "ax.barh(y-w/2, rand.loc[order,'rmse_mean'], w, xerr=rand.loc[order,'rmse_ci95'], label='random K-fold', color='#48c')\n"
         "ax.barh(y+w/2, grp.loc[order,'rmse_mean'], w, xerr=grp.loc[order,'rmse_ci95'], label='researcher-held-out', color='#e8a')\n"
         "ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis(); ax.set_xlabel('CV RMSE on stress [MPa]'); ax.legend(); ax.set_title('Random vs researcher-held-out validation'); plt.tight_layout(); plt.show()"),
    md("### Paired significance (random-split out-of-fold errors)"),
    code("sig = pd.read_csv(RESULTS/'paired_significance.csv')\n"
         "display(sig.round(4))"),
    md("## The lesson\n\n"
       "- Under **random** K-fold, Random Forest and SVR-RBF clearly beat EC2 "
       "(Wilcoxon p < 1e-8) — this reproduces the original *“ML beats Eurocode”* story.\n"
       "- Under **researcher-held-out** CV the ranking **collapses**: **EC2 is the "
       "best model**, every ML model falls back to the linear baseline, and SVR-RBF "
       "and the trees degrade sharply.\n\n"
       "The apparent ML superiority was largely **lab leakage** — with many specimens "
       "per researcher, a random split lets flexible models memorize lab-specific "
       "offsets. The honest conclusion: on this dataset, **no ML model reliably "
       "out-generalizes the Eurocode 2 formula** to a new laboratory. The value of "
       "the ML study is interpretive (it identifies `rho_l`/`fcm` as the drivers and "
       "rules out column profile), not a replacement for the code formula."),
)

NOTEBOOKS = {
    "01_data_overview.ipynb": nb01,
    "02_eurocode_baseline.ipynb": nb02,
    "03_regression_models.ipynb": nb03,
    "04_svm_models.ipynb": nb04,
    "05_tree_models.ipynb": nb05,
    "06_model_comparison.ipynb": nb06,
}

if __name__ == "__main__":
    for fname, notebook in NOTEBOOKS.items():
        path = NB_DIR / fname
        nbf.write(notebook, path)
        print("wrote", path.relative_to(NB_DIR.parent))
