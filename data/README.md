# Dataset

## `Daten_Siburg.xlsx`

336 published flat-slab punching-test results compiled by Dr. Karl Friedrich Siburg. This is the primary dataset, used by every notebook for training and evaluation.

### Features

| Column | Unit | Meaning |
|---|---|---|
| `Bez Vers.Ber` | – | Test designation (string identifier) |
| `Forscher` | – | Researcher / source publication |
| `Platte` | – | Slab identifier |
| `Last` | – | Load type |
| `Lasteinleitung` | m | Perimeter of load-introduction surface `u_0` |
| `h` | mm | Slab thickness |
| `d` | mm | Effective static depth |
| `Stütze` | – | Column profile (`k`=round, `q`=square, `r`=rectangular) |
| `c1` | mm | Column side length / diameter |
| `c2` | mm | Second column side length (if rectangular) |
| `rho_l` | % | Longitudinal reinforcement ratio |
| `fcm,cyl` | MPa | Mean concrete cylinder compressive strength |
| `dg` | mm | Maximum aggregate (grain) diameter |
| `fym` | MPa | Mean steel yield strength |
| `Esm` | GPa | Mean steel modulus |
| `V_test` | MN | **Target** — measured punching load |

The `Fläche` (column cross-section area) feature used in the notebooks is **derived** from `Stütze`, `c1`, `c2`.

### Known data-quality notes
- `c2` is present only for rectangular columns (~14/336 rows).
- `dg` has ≈39 % missing values; `Esm`, `fym` also patchy.
- `h` and `d` are nearly collinear (r ≈ 0.99) — pick one.

## `Data.xlsx`

Row-aligned to `Daten_Siburg.xlsx` (verified on `V_test`), with extra precomputed
Eurocode columns. The rebuild uses two of them:

- **`beta`** — despite the name, this is the EC2 **control area `u₁·d`** [mm²] (the
  basic control perimeter at 2·d times the effective depth), **not** the EC2
  eccentricity factor. Verified: `V_Rd = v_Rd · beta` and `v_test = V_test·10⁶ / beta`.
- **`v_test`**, **`v_Rd`** — measured and EC2 punching **stresses** [MPa]. `v_test`
  is the rebuild's modelling target; the `punching_shear` package recomputes it from
  `V_test` and `beta`.

(`Daten_Siburg.xlsx` lacks these columns but carries `Forscher`, needed for the
researcher-held-out cross-validation, so the package merges `beta` from `Data.xlsx`.)

## References for the data

- C. Siburg (2014), *Zur einheitlichen Bemessung gegen Durchstanzen…* (doctoral
  thesis, RWTH Aachen) — the source of the compiled test database and the critical
  EC2 / fib MC2010 comparison; see `references/Siburg2014_PunchingShear_DesignFoundations.pdf`.
- DIN EN 1992-1-1 (Eurocode 2) — see `references/`.

The genuine EC2-vs-fib-MC2010 critical comparison is in the Siburg 2014 thesis
(Chapter 2). (A previously bundled `Ricker_Siburg_EC2_fib_critical_review.pdf` was
removed: it was mis-named and actually contained an unrelated 2021 paper on
explainable AI, not the Ricker/Siburg review.)
