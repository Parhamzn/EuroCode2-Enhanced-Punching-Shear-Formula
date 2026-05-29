# Interpretable ML for the Punching-Shear Resistance of RC Flat Slabs

A **corrected, leak-free rebuild** of an ETH Zürich AEC SciML semester project
(HS2021). It reconstructs and benchmarks the **Eurocode 2 / DIN EN 1992-1-1**
empirical punching-shear formula against interpretable ML models, on 336
published flat-slab punching tests (the Siburg compilation).

> This is a re-engineered version of the original study. The original notebooks
> are preserved under [`notebooks/legacy/`](notebooks/legacy/). The rebuild fixes
> a target-definition flaw, data leakage, an un-interpretable metric, and the
> absence of grouped cross-validation — corrections that **change the study's
> headline conclusion**. See [What changed and why](#what-changed-and-why).

## The question

The EC2 punching formula carries an apparent safety factor (the genuine mean of
`V_test/V_Rd` on this data is **2.28**, not the ≈1.8 sometimes quoted) to absorb
scatter between predicted and measured punching loads. Can interpretable ML
trained on the same physical features shrink that scatter, and which features
actually drive punching behaviour?

## Headline result

All 11 models — including the Eurocode 2 baseline — are scored on **identical
cross-validation folds**, predicting the punching **stress** `v = V/(u₁·d)` [MPa],
with errors in physical units. Two validation protocols tell very different
stories:

**Random repeated 5×5 K-fold** (what the original study effectively used):

| Model | CV RMSE [MPa] | R² |
|---|---|---|
| **Random Forest** | **0.234** | **0.79** |
| SVR (RBF) | 0.252 | 0.75 |
| NLR (`d×fcm`) | 0.285 | 0.68 |
| OLS / Ridge / Lasso / ElasticNet / SVR-linear | ≈0.288 | ≈0.68 |
| Decision Tree | 0.307 | 0.63 |
| EC2 (refit `C_Rd,c`) | 0.310 | 0.63 |
| SVR (poly-3) | 0.655 | −0.90 *(overfits)* |

→ Random Forest and SVR-RBF beat EC2 (paired Wilcoxon *p* < 1e-8). This *looks*
like "ML beats Eurocode."

**Researcher-held-out GroupKFold** (whole labs held out — the honest test of
generalizing to a *new* experiment):

| Model | CV RMSE [MPa] | R² |
|---|---|---|
| **EC2 (refit `C_Rd,c`)** | **0.310** | **0.61** |
| Ridge / OLS / ElasticNet / Lasso / SVR-linear / NLR | ≈0.316 | ≈0.58 |
| Random Forest | 0.317 | 0.58 |
| SVR (RBF) | 0.362 | 0.33 |
| Decision Tree | 0.372 | 0.42 |
| SVR (poly-3) | 1.40 | −12.7 |

→ The ranking **collapses**. **No ML model out-generalizes EC2** to a new lab;
the flexible models (RBF, trees) degrade most. The apparent ML superiority under
random splits was largely **lab leakage** — with many specimens per researcher, a
random split lets flexible models memorize lab-specific offsets.

**Where ML still adds value is interpretive.** On the corrected stress target,
permutation importance shows the punching stress is driven by the **reinforcement
ratio `rho_l`** and **concrete strength `fcm_cyl`** — the actual mechanical
drivers — not by the effective depth `d` (which dominated only because the
original study predicted absolute *load*). The categorical **column profile
contributes essentially nothing**, so it could be dropped from future code
formulas — the one original conclusion that survives the correction.

## What changed and why

| # | Original study | Problem | Rebuild |
|---|---|---|---|
| 1 | Predicts absolute **load** `V_test` [MN] | Load ∝ control area `u₁·d`, so regressing on `d` relearns a trivial size effect (corr(d, load)=0.89 vs corr(d, **stress**)=−0.27). This produced the "`d` dominates" headline. | Predicts the **stress** `v = V/(u₁·d)` [MPa] — EC2's actual output. |
| 2 | MSE on **MinMax-scaled** target ("per-mille") | Not an interpretable engineering error; ranking pooled different datasets/splits/scalers. | **RMSE/MAE/MAPE/R² in physical units (MPa)** on one dataset, shared folds. |
| 3 | Scaler fit on **full data** before split; Ridge fit on the **test set**; SVR tuned on the **test set** | Data leakage → optimistic scores. | Every model is an sklearn **`Pipeline`** (scaler refit inside each fold); tuning is **nested CV** that never sees held-out data. |
| 4 | Single random 70/30 split, reported to 0.1‰ | n=336 from 55 labs is **not i.i.d.**; one split treats it so (the original even stratified *by* researcher). | **Repeated K-fold + researcher-held-out GroupKFold**, with 95% CIs and paired Wilcoxon tests. |
| 5 | "EC2 baseline" trusts a spreadsheet column; SF quoted as 1.8 | No formula implemented; SF mislabeled. | EC2 **implemented from raw inputs** (with `k≤2`, `ρ_l≤2%`, fck class caps), **validated** to reproduce the spreadsheet (median error 1e-7), `C_Rd,c` **refit per fold**; true SF reported (2.28). |
| 6 | Notebooks don't run (`fcmcyl` vs `fcm,cyl`, `Umfang`, `V_adj` …) | Not reproducible. | Clean package + **executed** notebooks + a test suite; runs end-to-end from a fresh clone. |

A full written audit of the original notebooks (65 verified findings) motivated
these changes.

## Can an *explainable* model beat EC2 — honestly? (research)

Since EC2 wins under researcher-held-out CV, the real question is whether an
**interpretable model that reduces to a neat formula** can beat it on the *honest*
bar. `scripts/run_formula_models.py` (notebook [`07`](notebooks/07_explainable_formulas.ipynb))
evaluates grey-box, symbolic and feature-engineered models on the same
researcher-held-out folds, and extracts the fitted equations.

**Yes — modestly, and only with the right structure.** Three closed-form models
beat EC2 under researcher-held-out CV at paired-Wilcoxon *p* < 0.01:

| Model | grouped RMSE [MPa] | R² | vs EC2 (Wilcoxon) |
|---|---|---|---|
| **Power-law** `v = 1.38·d^(−0.19)·rho^0.33·fck^0.31` | **0.285** | **0.67** | **p = 2e-5** |
| OLS + mechanics features | 0.299 | 0.64 | p = 1.6e-4 |
| EC2 × correction (grey-box) | 0.301 | 0.63 | p = 1.9e-4 |
| EC2 free-exponent `v=C·k·(100ρ_l·fck)^p`, p=0.31 | 0.308 | 0.61 | p = 0.36 (n.s.) |
| **EC2 (refit C_Rd,c)** | 0.310 | 0.61 | — |
| Random Forest (+features) | 0.302–0.314 | 0.59–0.63 | n.s. |
| Symbolic regression (gplearn) | 0.347 | 0.46 | n.s. |

Three findings worth keeping:

1. **The data re-derive EC2's form.** The free-exponent power-law lands at
   `rho^0.33 · fck^0.31` (≈ the EC2 cube root) and EC2-with-a-freed-exponent gives
   `p = 0.31` — *not* significantly different from `1/3`. EC2's functional form is
   **validated**; the gains come from freeing the size term (`d^(−0.19)` instead of
   the saturating `k`) and from the multiplicative structure.
2. **Structure beats flexibility.** Flexible black-boxes (RBF-SVR, deep trees, raw
   Random Forest) do **not** out-generalize EC2 here; raw symbolic regression
   (gplearn, which fits constants poorly) doesn't either. The wins are all
   low-complexity, mechanics-anchored forms.
3. **Feature engineering is the cheapest lever.** Feeding mechanics-informed inputs
   (the EC2 composite `(100·ρ_l·fck)^(1/3)`, the size factor `k`, dimensionless
   column compactness) lifts even plain OLS above EC2.

**The improvement is real but small** — RMSE ≈ −8 %, R² +0.06, with overlapping
CIs — consistent with the literature, where good symbolic/GP punching formulas
land at CoV ≈ 0.14–0.21, comparable to (not dramatically better than) EC2. The
honest takeaway: *a freed power-law with mechanics-informed features is a tidy,
slightly-better-than-EC2 design equation; a bigger model is not the lever.*

**Levers tried / recommended next:** ✓ stress target, ✓ mechanics-informed &
dimensionless features, ✓ free-exponent power-law, ✓ grey-box EC2 correction
factor, ✓ symbolic regression. Not yet: add aggregate size `dg` (CSCT crack-width
proxy `d/(16+dg)`) with imputation; try **PySR** (Julia backend, optimises
constants — stronger than gplearn) for the correction factor; a monotonic
Explainable Boosting Machine for shape vetting.

## Repository layout

```
sciml-punching-shear/
├── punching_shear/            # the reusable, leak-free package
│   ├── data.py                #   load/clean; stress target; researcher groups; fck caps
│   ├── eurocode.py            #   EC2 stress formula (+caps), refit C_Rd,c, EC2Regressor
│   ├── evaluation.py          #   physical-unit metrics (stress & load), shared folds, paired tests
│   ├── models.py              #   11-model zoo + explainable/formula model set
│   ├── features.py            #   mechanics-informed feature engineering
│   ├── greybox.py             #   power-law, EC2 free-exponent, EC2 x correction (closed-form)
│   └── symbolic.py            #   gplearn symbolic regression (+ sklearn>=1.6 shim)
├── scripts/
│   ├── run_analysis.py        # main study -> results/ tables + figures  (~9 min)
│   ├── run_formula_models.py  # explainable/formula models vs EC2 -> results/formula_* (~3 min)
│   └── build_notebooks.py     # regenerate the notebooks from the package
├── notebooks/                 # clean, executed notebooks (01-07)
│   └── legacy/                # original HS2021 notebooks (preserved, superseded)
├── results/                   # generated CSV tables + PNG figures (committed)
├── tests/                     # pytest sanity/guard suite
├── data/                      # Daten_Siburg.xlsx (raw, +Forscher), Data.xlsx (+control area)
├── references/, docs/
├── pyproject.toml             # installable package + optional extras
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebooks,dev]"      # package + jupyter/seaborn + pytest

pytest -q                              # 13 guard tests (~25 s)
python scripts/run_analysis.py         # main study -> results/  (~9 min, nested CV)
python scripts/run_formula_models.py   # explainable-formula study -> results/  (~3 min)
jupyter lab notebooks/                 # the narrative, 01 -> 07
```

The package resolves data paths relative to itself, so notebooks and scripts work
from a fresh clone without path edits.

## Dataset

336 published flat-slab punching tests compiled by Dr. Karl Friedrich Siburg.
Modelling uses five fully-observed features — effective depth `d` [mm], column
area `col_area` [mm²], reinforcement ratio `rho_l` [%], cylinder strength
`fcm_cyl` [MPa], and load perimeter `u0_perim` [m] — to predict the punching
**stress** `v_test` [MPa]. The companion `Data.xlsx` supplies the EC2 control
area `beta = u₁·d`, so `v = V_test·10⁶/beta`. `Forscher` (source lab) drives the
grouped CV. `dg`, `fym`, `Esm`, `c2` are dropped (heavily missing). See
[`data/README.md`](data/README.md).

EC2 reference (resistance stress, without shear reinforcement):

```
v_Rd,c = C_Rd,c · k · (100·rho_l·f_ck)^(1/3)
k = 1 + √(200/d) ≤ 2.0,  rho_l ≤ 0.02,  f_ck = fcm − 8 clamped to C12/15…C90/105
```

## Extending this work

- Add `dg`, `fym`, `Esm` back with proper imputation rather than dropping them.
- Fit the **exponents** of the EC2 form (true grey-box SciML), not just `C_Rd,c`.
- Physics-informed regularization toward the EC2 functional form.
- Broaden the dataset beyond interior columns (edge/corner, footings) so the
  grouped-CV generalization test spans the cases EC2 actually differentiates.

## License

MIT — see [`LICENSE`](LICENSE).
