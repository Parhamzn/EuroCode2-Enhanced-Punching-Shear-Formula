"""Loading and cleaning of the Siburg punching-shear dataset.

Canonical source ``data/Daten_Siburg.xlsx`` (336 flat-slab punching tests, with
the ``Forscher`` source label needed for grouped cross-validation). The companion
``data/Data.xlsx`` is row-aligned and carries the EC2 control area
``beta = u1*d`` plus the spreadsheet's precomputed EC2 columns; we merge ``beta``
so the target can be expressed as a shear *stress*.

Nothing here is fit on the data, so importing/loading has no leakage implications;
all scaling and model fitting happen later, inside cross-validation folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Repository ``data/`` directory (this file lives at <repo>/punching_shear/data.py).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "Daten_Siburg.xlsx"
COMPANION_PATH = DATA_DIR / "Data.xlsx"

# Original (mixed German/English) -> clean ASCII identifiers used everywhere downstream.
RENAME = {
    "Fläche": "col_area",        # column cross-section area [mm^2]
    "fcm,cyl": "fcm_cyl",        # mean cylinder compressive strength [MPa]
    "Lasteinleitung": "u0_perim",  # perimeter of the load-introduction area u0 [m]
    # 'd' (effective depth, mm) and 'rho_l' (reinforcement ratio, %) keep their names.
}

# Feature set used by every model. All five are complete (zero missing values).
FEATURES = ["d", "col_area", "rho_l", "fcm_cyl", "u0_perim"]

FEATURE_LABELS = {
    "d": "effective depth d [mm]",
    "col_area": "column area [mm^2]",
    "rho_l": "reinf. ratio rho_l [%]",
    "fcm_cyl": "concrete strength fcm,cyl [MPa]",
    "u0_perim": "load perimeter u0 [m]",
}

# EC2 concrete-class validity bounds for the characteristic strength fck [MPa]
# (C12/15 ... C90/105). fck is derived as fcm - 8 and clamped to this range.
FCK_FLOOR = 12.0
FCK_CAP = 90.0
FCM_TO_FCK_OFFSET = 8.0


@dataclass
class Dataset:
    """Container for the cleaned dataset.

    Attributes
    ----------
    X : feature matrix (columns == :data:`FEATURES`), physical units, no scaling.
    y_stress : punching shear stress v = V_test / (u1*d) [MPa]  -- the PRIMARY target.
    y_load : measured failure load V_test [MN]  -- the original (size-confounded) target.
    beta : EC2 control area u1*d [mm^2], used to convert stress <-> load.
    groups : ``Forscher`` source label, for GroupKFold (researcher-held-out CV).
    fck : characteristic concrete strength [MPa], clamped to the EC2 class range.
    raw : the full cleaned frame (all columns), for reference / EDA.
    """

    X: pd.DataFrame
    y_stress: pd.Series
    y_load: pd.Series
    beta: pd.Series
    groups: pd.Series
    fck: pd.Series
    raw: pd.DataFrame
    feature_names: list = field(default_factory=lambda: list(FEATURES))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.X)

    def ec2_inputs(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return (d [mm], rho_l [%], fck [MPa]) aligned to :attr:`X`."""
        return self.X["d"], self.X["rho_l"], self.fck

    def stress_to_load(self, v_mpa) -> np.ndarray:
        """Convert a predicted stress [MPa] back to load [MN] via V = v * u1*d."""
        return np.asarray(v_mpa) * self.beta.to_numpy() / 1.0e6


def _read_clean(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Drop fully-unnamed spreadsheet padding columns.
    df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith("unnamed")]]
    df.columns = [str(c) for c in df.columns]
    # Sentinel '-' marks missing. Opt into the future no-downcast behaviour so
    # replace() doesn't emit a FutureWarning, then restore object inference.
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace("-", np.nan).infer_objects(copy=False)
    return df


def load_dataset(data_dir: str | Path | None = None) -> Dataset:
    """Load, clean and assemble the punching-shear dataset.

    Parameters
    ----------
    data_dir : optional override for the directory holding the two Excel files.
    """
    raw_path = RAW_PATH
    companion_path = COMPANION_PATH
    if data_dir is not None:
        data_dir = Path(data_dir)
        raw_path = data_dir / RAW_PATH.name
        companion_path = data_dir / COMPANION_PATH.name

    raw = _read_clean(raw_path)
    companion = _read_clean(companion_path)

    if len(raw) != len(companion):
        raise ValueError(
            f"Row count mismatch: {raw_path.name}={len(raw)} vs "
            f"{companion_path.name}={len(companion)}"
        )

    # The two files are published in the same row order; verify with V_test before
    # trusting a positional merge of the control area.
    v_raw = pd.to_numeric(raw["V_test"], errors="coerce").to_numpy()
    v_comp = pd.to_numeric(companion["V_test"], errors="coerce").to_numpy()
    if not np.allclose(v_raw, v_comp, equal_nan=True, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "Daten_Siburg.xlsx and Data.xlsx are not row-aligned on V_test; "
            "refusing to merge the control area positionally."
        )

    raw = raw.rename(columns=RENAME)
    for col in ["d", "col_area", "rho_l", "fcm_cyl", "u0_perim", "V_test"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    beta = pd.to_numeric(companion["beta"], errors="coerce")  # u1*d [mm^2]
    beta.index = raw.index

    # --- targets ---------------------------------------------------------------
    # Primary: shear stress [MPa] = V_test[MN] * 1e6 / (u1*d)[mm^2].
    y_load = raw["V_test"].astype(float)             # [MN]
    y_stress = y_load * 1.0e6 / beta                 # [MPa]
    y_stress.name = "v_test"

    # --- features --------------------------------------------------------------
    X = raw[FEATURES].astype(float).copy()

    missing = X.isna().sum()
    if missing.any():
        raise ValueError(f"Unexpected missing values in core features:\n{missing[missing > 0]}")

    # --- groups & fck ----------------------------------------------------------
    groups = raw["Forscher"].astype(str).str.strip()
    fck = (raw["fcm_cyl"] - FCM_TO_FCK_OFFSET).clip(lower=FCK_FLOOR, upper=FCK_CAP)
    fck.name = "fck"

    raw["v_test"] = y_stress
    raw["beta"] = beta
    raw["fck"] = fck

    return Dataset(
        X=X,
        y_stress=y_stress,
        y_load=y_load,
        beta=beta,
        groups=groups,
        fck=fck,
        raw=raw,
    )
