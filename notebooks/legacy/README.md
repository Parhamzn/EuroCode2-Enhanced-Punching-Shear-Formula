# Legacy notebooks (original HS2021 submission)

These are the **original** notebooks from the ETH HS2021 SciML project, kept
verbatim for provenance. They are **superseded** by the clean notebooks in the
parent `notebooks/` directory and the `punching_shear` package.

They are retained as-is and are **known not to run end-to-end** against the
committed data, and to contain the methodological issues documented in the
project audit. The main ones:

- **Wrong target.** They regress the absolute failure load `V_test` [MN], which
  is mechanically proportional to the control area `u1·d`, so the headline
  "effective depth `d` dominates" is largely a size-effect artifact
  (corr(d, load)=0.89 vs corr(d, stress)=−0.27). The rebuild targets the shear
  *stress*.
- **Leakage.** `MinMaxScaler` is fit on the full dataset before the train/test
  split (incl. the target); in `03_regression_models.ipynb` the Ridge model is
  fit on the *test* set; in `04_svm_models.ipynb` hyper-parameters are tuned on
  the test set and `GridSearchCV` is fit on the full data.
- **Metric.** Errors are reported as MSE on the MinMax-scaled target (a "per-mille"
  number), not in physical units, and the cross-notebook ranking mixes datasets,
  splits and scalers.
- **No grouped CV.** A single random 70/30 split treats researcher-clustered data
  as i.i.d.; `05` even stratifies *by* researcher. The rebuild shows that under
  honest researcher-held-out CV, no ML model beats Eurocode 2.
- **Reproducibility.** Hard-coded column-name typos (`fcmcyl` vs `fcm,cyl`),
  references to non-existent columns (`Umfang`, `V_adj`), etc.

See the parent `notebooks/`, `scripts/run_analysis.py`, and the project `README.md`
for the corrected analysis.
