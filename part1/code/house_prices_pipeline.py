"""
House Prices (Ames) - data cleaning, feature engineering and EDA pipeline.

Single source of truth for the deliverables in this folder. Run directly to
regenerate every processed table:

    python code/house_prices_pipeline.py

Outputs
-------
data/train_clean.csv       cleaned train, original columns + flags + log target
data/test_clean.csv        cleaned test, same treatment (no target)
data/train_features.csv    model-ready numeric matrix (engineered + one-hot)
data/test_features.csv     model-ready numeric matrix (aligned to train)
reports/audit.json         raw-data audit figures quoted in the report
reports/missing_values.csv missingness in both splits, with the reason
tables/*.csv               small tables used by the report
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APPENDIX_ROOT = Path(__file__).resolve().parents[2]
RAW_TRAIN = APPENDIX_ROOT / "data" / "raw" / "train.csv"
RAW_TEST = APPENDIX_ROOT / "data" / "raw" / "test.csv"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
TABLES = ROOT / "tables"
for _d in (DATA, REPORTS, TABLES):
    _d.mkdir(exist_ok=True)

SEED = 42
TARGET = "SalePrice"

# ---------------------------------------------------------------------------
# 1. Column roles
# ---------------------------------------------------------------------------
# Quality / condition scales: Ex > Gd > TA > Fa > Po, and "no feature" -> 0.
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
# Columns where the raw file encodes "no feature" as NaN and the data
# dictionary says NA = "No Basement / No Fireplace / No Garage ...".
NONE_LIKE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
BSMT_COLS = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
GARAGE_COLS = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]

ORDINAL_MAPS = {
    "LotShape": {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0},
    "LandContour": {"Lvl": 3, "Bnk": 2, "HLS": 1, "Low": 0},
    "LandSlope": {"Gtl": 2, "Mod": 1, "Sev": 0},
    "ExterQual": QUALITY_MAP,
    "ExterCond": QUALITY_MAP,
    "BsmtQual": QUALITY_MAP,
    "BsmtCond": QUALITY_MAP,
    "HeatingQC": QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": QUALITY_MAP,
    "GarageQual": QUALITY_MAP,
    "GarageCond": QUALITY_MAP,
    "PoolQC": QUALITY_MAP,
    "BsmtExposure": {"Gd": 3, "Av": 2, "Mn": 1, "No": 0, "None": -1},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
    "PavedDrive": {"Y": 2, "P": 1, "N": 0},
    "Fence": {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0},
    "CentralAir": {"Y": 1, "N": 0},
    "Street": {"Pave": 1, "Grvl": 0},
    "Alley": {"Pave": 2, "Grvl": 1, "None": 0},
    "Utilities": {"AllPub": 3, "NoSewr": 2, "NoSeWa": 1, "ELO": 0},
}

# Nominal variables that stay categorical and are one-hot encoded at the end.
NOMINAL = [
    "MSZoning", "MSSubClass", "LotConfig", "Neighborhood", "Condition1",
    "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating",
    "GarageType", "MiscFeature", "SaleType", "SaleCondition", "MoSold",
    "YrSold",
]
# Numeric columns that are really labels, not magnitudes.
AS_CATEGORICAL = ["MSSubClass", "MoSold", "YrSold"]

CONTINUOUS = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]
DISCRETE = [
    "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFullBath",
    "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageYrBlt",
]

# ---------------------------------------------------------------------------
# 2. Audit
# ---------------------------------------------------------------------------

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two raw files exactly as distributed by Kaggle."""
    train = pd.read_csv(RAW_TRAIN)
    test = pd.read_csv(RAW_TEST)
    return train, test


def audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Quantities quoted in the Dataset Description / Data Cleaning sections."""
    miss_tr = train.isna().sum()
    miss_te = test.isna().sum()
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(train.shape[1] - 2),  # minus Id and SalePrice
        "n_nominal": len(NOMINAL),
        "duplicate_ids": int(train.Id.duplicated().sum() + test.Id.duplicated().sum()),
        "duplicate_rows": int(train.drop(columns=["Id"]).duplicated().sum()),
        "cols_with_missing_train": int((miss_tr > 0).sum()),
        "cols_with_missing_test": int((miss_te > 0).sum()),
        "total_missing_cells_train": int(miss_tr.sum()),
        "total_missing_cells_test": int(miss_te.sum()),
        "pct_missing_cells_train": round(float(miss_tr.sum()) / train.size * 100, 2),
        "unique_values": int(train.drop(columns=["Id"]).nunique().sum()),
        "target_mean": round(float(train[TARGET].mean()), 2),
        "target_median": round(float(train[TARGET].median()), 2),
        "target_std": round(float(train[TARGET].std()), 2),
        "target_min": int(train[TARGET].min()),
        "target_max": int(train[TARGET].max()),
        "target_skew": round(float(train[TARGET].skew()), 3),
        "target_log_skew": round(float(np.log(train[TARGET]).skew()), 3),
        "quality_price_corr": round(float(train.OverallQual.corr(train[TARGET])), 3),
        "grlivarea_price_corr": round(float(train.GrLivArea.corr(train[TARGET])), 3),
        "yearbuilt_price_corr": round(float(train.YearBuilt.corr(train[TARGET])), 3),
        "totalbsmt_1stflr_corr": round(float(train.TotalBsmtSF.corr(train["1stFlrSF"])), 3),
        "garagecars_area_corr": round(float(train.GarageCars.corr(train.GarageArea)), 3),
        "partial_sales": int((train.SaleCondition == "Partial").sum()),
        "pool_homes": int((train.PoolArea > 0).sum()),
    }


def missing_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Missingness per column, with the interpretation used downstream."""
    reason = {}
    for c in NONE_LIKE:
        reason[c] = "Structural: NA = no feature (basement/garage/pool/...)"
    reason.update({
        "LotFrontage": "True missing: not recorded in the assessor file",
        "MasVnrArea": "Structural: 0 sq ft of veneer",
        "GarageYrBlt": "Structural: no garage",
        "Electrical": "True missing (1 row)",
        "BsmtFinSF1": "Structural: no basement",
        "BsmtFinSF2": "Structural: no basement",
        "BsmtUnfSF": "Structural: no basement",
        "TotalBsmtSF": "Structural: no basement",
    })
    rows = []
    for c in train.columns:
        if c in ("Id", TARGET) or c not in test.columns:
            continue
        n_tr, n_te = int(train[c].isna().sum()), int(test[c].isna().sum())
        if n_tr == 0 and n_te == 0:
            continue
        rows.append({
            "column": c,
            "missing_train": n_tr,
            "pct_train": round(n_tr / len(train) * 100, 2),
            "missing_test": n_te,
            "interpretation": reason.get(c, "True missing"),
        })
    tab = pd.DataFrame(rows).sort_values("missing_train", ascending=False)
    tab.to_csv(REPORTS / "missing_values.csv", index=False)
    return tab


# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------

def _impute_lot_frontage(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Neighbourhood median from the training split only (no test leakage).

    LotFrontage is missing for 17.7% of sales and is not missing at random:
    cul-de-sacs and corner lots lose it far more often than inside lots.
    A neighbourhood median respects the local street layout, and fitting it on
    train only keeps the test split untouched.
    """
    neigh_med = train.groupby("Neighborhood")["LotFrontage"].median()
    global_med = train["LotFrontage"].median()
    tr = train["LotFrontage"].fillna(train["Neighborhood"].map(neigh_med)).fillna(global_med)
    te = test["LotFrontage"].fillna(test["Neighborhood"].map(neigh_med)).fillna(global_med)
    return tr, te


def clean(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Repair, impute and type-cast both splits with identical rules."""
    tr, te = train.copy(), test.copy()
    log: dict = {}

    # -- 3.1 structural NA -> explicit "None" ---------------------------------
    for c in NONE_LIKE:
        tr[c] = tr[c].fillna("None")
        te[c] = te[c].fillna("None")
    log["none_filled"] = len(NONE_LIKE)

    # -- 3.2 basement / garage / veneer: absence means zero -------------------
    bsmt_zero = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]
    for c in bsmt_zero + ["MasVnrArea", "GarageCars", "GarageArea"]:
        tr[c] = tr[c].fillna(0.0)
        te[c] = te[c].fillna(0.0)
    for c in ["BsmtFullBath", "BsmtHalfBath"]:
        tr[c] = tr[c].fillna(0).astype(int)
        te[c] = te[c].fillna(0).astype(int)

    # GarageYrBlt: no garage -> no garage year. Impute with YearBuilt so that
    # "garage age" stays on the same scale instead of becoming a nonsense 0.
    for df in (tr, te):
        no_garage = df.GarageType.eq("None")
        df.loc[no_garage, "GarageYrBlt"] = np.nan
    tr["GarageYrBlt"] = tr["GarageYrBlt"].fillna(tr["YearBuilt"])
    te["GarageYrBlt"] = te["GarageYrBlt"].fillna(te["YearBuilt"])

    # -- 3.3 true missing values ---------------------------------------------
    tr["LotFrontage"], te["LotFrontage"] = _impute_lot_frontage(tr, te)
    log["lotfrontage_imputed_train"] = int(train.LotFrontage.isna().sum())
    log["lotfrontage_imputed_test"] = int(test.LotFrontage.isna().sum())

    # Electical: one row; MSZoning / KitchenQual / SaleType / Functional /
    # Exterior1st / Exterior2nd / Utilities: a handful of rows in test.
    mode_cols = ["Electrical", "MSZoning", "KitchenQual", "SaleType",
                 "Functional", "Exterior1st", "Exterior2nd", "Utilities"]
    filled = {}
    for c in mode_cols:
        n_tr, n_te = int(tr[c].isna().sum()), int(te[c].isna().sum())
        if n_tr or n_te:
            mode = tr[c].mode(dropna=True).iat[0]
            tr[c] = tr[c].fillna(mode)
            te[c] = te[c].fillna(mode)
            filled[c] = {"train": n_tr, "test": n_te, "value": mode}
    log["mode_imputed"] = filled

    # -- 3.4 data errors ------------------------------------------------------
    # Garage built after the house was sold is impossible; clip to YrSold.
    bad_garage = 0
    for df in (tr, te):
        mask = df.GarageYrBlt > df.YrSold
        bad_garage += int(mask.sum())
        df.loc[mask, "GarageYrBlt"] = df.loc[mask, "YrSold"]
    log["garage_year_clipped"] = bad_garage

    # MasVnrType "None" with positive area -> the area is the reliable signal.
    veneer_fix = 0
    for df in (tr, te):
        mask = df.MasVnrType.eq("None") & df.MasVnrArea.gt(0)
        veneer_fix += int(mask.sum())
        df.loc[mask, "MasVnrType"] = "BrkFace"
    log["veneer_type_repaired"] = veneer_fix

    # -- 3.5 type casting -----------------------------------------------------
    for c in AS_CATEGORICAL:
        tr[c] = tr[c].astype(int).astype(str)
        te[c] = te[c].astype(str)
    # Ids are ints; everything else that is count-like stays int.
    log["constant_columns"] = [c for c in tr.columns if tr[c].nunique(dropna=False) <= 1]
    return tr, te, log


def ordinal_encode(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn ordered labels into integers (Ex=5 ... Po=1, absence=0)."""
    tr, te = train.copy(), test.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col not in tr.columns:
            continue
        tr[col] = tr[col].map(mapping).astype("float")
        te[col] = te[col].map(mapping).astype("float")

    # The 16 MSSubClass codes are a nominal list, not a magnitude.
    tr["MSSubClass"] = tr["MSSubClass"].astype(str)
    te["MSSubClass"] = te["MSSubClass"].astype(str)
    return tr, te


def collapse_rare(train: pd.DataFrame, test: pd.DataFrame, min_count: int = 10,
                  skip: tuple[str, ...] = ("Neighborhood",)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fold levels seen fewer than `min_count` times in train into "Other".

    Frequencies come from the training split only, so the mapping is unchanged
    by the test data. Keeps one-hot width sane and avoids levels that a model
    can never estimate (e.g. RoofMatl='Membran', n=1). `Neighborhood` is kept
    intact: it is the strongest location signal, and the smoothed target
    encoding deals with its rare levels instead of erasing them.
    """
    tr, te = train.copy(), test.copy()
    mapping_log = {}
    for c in NOMINAL:
        if c not in tr.columns or c in skip:
            continue
        counts = tr[c].value_counts()
        keep = counts[counts >= min_count].index
        rare = [v for v in counts.index if v not in set(keep)]
        if rare:
            tr[c] = tr[c].where(tr[c].isin(keep), "Other")
            te[c] = te[c].where(te[c].isin(keep), "Other")
            mapping_log[c] = {"n_levels_raw": int(counts.size),
                              "n_levels_kept": int(len(keep)),
                              "collapsed": rare}
    # Test-only levels that never appear in train cannot be estimated.
    for c in NOMINAL:
        if c in tr.columns:
            te[c] = te[c].where(te[c].isin(set(tr[c].unique())), "Other")
    return tr, te, mapping_log


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable features; no target is touched here."""
    d = df.copy()

    # -- size ---------------------------------------------------------------
    d["TotalSF"] = d.TotalBsmtSF + d["1stFlrSF"] + d["2ndFlrSF"]
    d["TotalFinSF"] = d.GrLivArea + d.BsmtFinSF1 + d.BsmtFinSF2
    d["TotalBath"] = (d.BsmtFullBath + d.FullBath
                      + 0.5 * (d.BsmtHalfBath + d.HalfBath))
    d["TotalPorchSF"] = (d.OpenPorchSF + d.EnclosedPorch + d["3SsnPorch"]
                         + d.ScreenPorch + d.WoodDeckSF)
    d["AvgRoomSF"] = d.GrLivArea / d.TotRmsAbvGrd.replace(0, np.nan)
    d["BsmtFinSF"] = d.BsmtFinSF1 + d.BsmtFinSF2

    # -- quality ------------------------------------------------------------
    d["OverallGrade"] = d.OverallQual * d.OverallCond
    qual_terms = ["ExterQual", "KitchenQual", "BsmtQual", "HeatingQC",
                  "FireplaceQu", "GarageQual"]
    d["QualScore"] = d[qual_terms].sum(axis=1)
    d["QualxSF"] = d.OverallQual * d.GrLivArea
    d["QualxTotalSF"] = d.OverallQual * d.TotalSF
    d["CondScore"] = d[["ExterCond", "BsmtCond", "GarageCond"]].sum(axis=1)

    # -- age ----------------------------------------------------------------
    d["HouseAge"] = d.YrSold.astype(int) - d.YearBuilt
    d["RemodAge"] = d.YrSold.astype(int) - d.YearRemodAdd
    d["GarageAge"] = d.YrSold.astype(int) - d.GarageYrBlt
    d["IsRemodeled"] = (d.YearRemodAdd != d.YearBuilt).astype(int)
    d["IsNew"] = (d.YrSold.astype(int) == d.YearBuilt).astype(int)
    d["AgeAtRemod"] = d.YearRemodAdd - d.YearBuilt

    # -- presence flags -----------------------------------------------------
    d["HasBsmt"] = (d.TotalBsmtSF > 0).astype(int)
    d["HasGarage"] = (d.GarageArea > 0).astype(int)
    d["HasFireplace"] = (d.Fireplaces > 0).astype(int)
    d["Has2ndFlr"] = (d["2ndFlrSF"] > 0).astype(int)
    d["HasPool"] = (d.PoolArea > 0).astype(int)
    d["HasMasVnr"] = (d.MasVnrArea > 0).astype(int)
    d["HasDeck"] = (d.WoodDeckSF > 0).astype(int)
    d["HasPorch"] = (d.TotalPorchSF > 0).astype(int)

    # -- ratios -------------------------------------------------------------
    d["BsmtFinRatio"] = np.where(d.TotalBsmtSF > 0, d.BsmtFinSF / d.TotalBsmtSF.replace(0, np.nan), 0.0)
    d["LivAreaRatio"] = np.where(d.TotalSF > 0, d.GrLivArea / d.TotalSF, 0.0)
    d["BedroomRatio"] = d.BedroomAbvGr / d.TotRmsAbvGrd.replace(0, np.nan)
    d["KitchenRatio"] = d.KitchenAbvGr / d.TotRmsAbvGrd.replace(0, np.nan)
    d["GarageAreaPerCar"] = np.where(d.GarageCars > 0, d.GarageArea / d.GarageCars.replace(0, np.nan), 0.0)

    # -- seasonality --------------------------------------------------------
    month = d.MoSold.astype(int)
    d["MoSoldSin"] = np.sin(2 * np.pi * month / 12)
    d["MoSoldCos"] = np.cos(2 * np.pi * month / 12)

    # -- log transforms for heavy-tailed area variables ---------------------
    # Every area and frontage is strictly positive, so the plain log is
    # defined and is written log(x) throughout the report.
    for c in ["LotArea", "GrLivArea", "TotalSF", "LotFrontage"]:
        d[f"log_{c}"] = np.log(d[c])

    return d


def target_encode(train: pd.DataFrame, test: pd.DataFrame, col: str = "Neighborhood",
                  m: float = 10.0, n_splits: int = 5) -> tuple[pd.Series, pd.Series]:
    """Out-of-fold smoothed target encoding, fitted on train only.

    Each training row gets the mean log-price of *other* folds, so the encoding
    carries no information about that row's own SalePrice. Test rows use the
    full-train mapping. This is the only feature that touches the target and it
    must be recomputed if the train/validation split changes.
    """
    from sklearn.model_selection import KFold

    y = np.log(train[TARGET])
    prior = y.mean()
    oof = pd.Series(index=train.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fit_idx, hold_idx in kf.split(train):
        stats = y.iloc[fit_idx].groupby(train[col].iloc[fit_idx]).agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + prior * m) / (stats["count"] + m)
        oof.iloc[hold_idx] = train[col].iloc[hold_idx].map(enc).fillna(prior).values
    stats = y.groupby(train[col]).agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + prior * m) / (stats["count"] + m)
    test_enc = test[col].map(enc).fillna(prior)
    return oof, test_enc


# ---------------------------------------------------------------------------
# 5. Model-ready matrix
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")


def to_matrix(train_f: pd.DataFrame, test_f: pd.DataFrame,
              drop_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode nominal columns against a shared column space."""
    drop_cols = drop_cols or []
    tr = train_f.drop(columns=[c for c in drop_cols if c in train_f.columns])
    te = test_f.drop(columns=[c for c in drop_cols if c in test_f.columns])

    # pandas 3 stores text in StringDtype, so test for "not numeric" rather
    # than for the old object dtype.
    cat_cols = [c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]
    tr_x = pd.get_dummies(tr, columns=cat_cols, dtype=float)
    te_x = pd.get_dummies(te, columns=cat_cols, dtype=float)
    te_x = te_x.reindex(columns=tr_x.columns, fill_value=0.0)
    tr_x.columns = [_sanitize(c) for c in tr_x.columns]
    te_x.columns = [_sanitize(c) for c in te_x.columns]
    return tr_x, te_x, list(tr_x.columns)


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------

def build() -> dict:
    raw_tr, raw_te = load_raw()
    info = audit(raw_tr, raw_te)

    miss = missing_table(raw_tr, raw_te)
    clean_tr, clean_te, clean_log = clean(raw_tr, raw_te)

    # Flag the two partial sales that break the size-price relationship.
    outlier_mask = (clean_tr.GrLivArea > 4000) & (clean_tr.SalePrice < 300000)
    clean_tr["SalePriceOutlier"] = outlier_mask.astype(int)
    clean_te["SalePriceOutlier"] = 0
    info["outliers_flagged"] = int(outlier_mask.sum())
    info["outlier_ids"] = clean_tr.loc[outlier_mask, "Id"].tolist()

    clean_tr["SalePriceLog"] = np.log(clean_tr[TARGET])
    clean_tr[["Id", "SalePriceOutlier", "SalePriceLog"] + [c for c in clean_tr.columns
             if c not in ("Id", "SalePriceOutlier", "SalePriceLog")]].to_csv(DATA / "train_clean.csv", index=False)
    clean_te.to_csv(DATA / "test_clean.csv", index=False)

    # Ordinal encoding + rare-level collapse + engineered features.
    ord_tr, ord_te = ordinal_encode(clean_tr.drop(columns=["SalePriceLog"]).copy(),
                                    clean_te.copy())
    rare_tr, rare_te, rare_log = collapse_rare(ord_tr, ord_te)
    feat_tr, feat_te = engineer(rare_tr), engineer(rare_te)
    engineered_cols = [c for c in feat_tr.columns if c not in rare_tr.columns]

    te_col, te_test = target_encode(rare_tr, rare_te)
    feat_tr["NeighborhoodPriceTE"] = te_col.values
    feat_te["NeighborhoodPriceTE"] = te_test.values

    X_tr, X_te, cols = to_matrix(feat_tr, feat_te,
                                 drop_cols=[TARGET, "SalePriceOutlier", "Id"])
    # Keep the flagged rows in *_clean.csv but out of the modelling matrix.
    keep = feat_tr.SalePriceOutlier.eq(0).values
    X_tr, y_tr = X_tr.loc[keep], feat_tr.loc[keep, TARGET]
    X_tr.insert(0, "SalePrice", y_tr.values)
    X_tr.insert(0, "Id", feat_tr.loc[keep, "Id"].values)
    X_te.insert(0, "Id", feat_te.Id.values)
    X_tr.to_csv(DATA / "train_features.csv", index=False)
    X_te.to_csv(DATA / "test_features.csv", index=False)

    info.update({
        "rows_modelled": int(len(X_tr)),
        "n_features_engineered": int(X_tr.shape[1]),
        "n_engineered_added": len(engineered_cols) + 1,  # +1 = NeighborhoodPriceTE
        "engineered_cols": engineered_cols,
        "clean_log": clean_log,
        "rare_collapse": rare_log,
        "matrix": {"train": list(X_tr.shape), "test": list(X_te.shape)},
    })

    # Small tables for the report.
    pd.DataFrame({"feature": X_tr.columns}).to_csv(TABLES / "matrix_columns.csv", index=False)
    info["n_zero_variance_matrix_cols"] = int((X_tr.drop(columns=["Id", "SalePrice"]).std(numeric_only=True) == 0).sum())
    with open(REPORTS / "audit.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)
    return {"info": info, "missing": miss, "train": X_tr, "test": X_te,
            "features": feat_tr, "clean": clean_tr, "rare_log": rare_log}


if __name__ == "__main__":
    out = build()
    info = out["info"]
    print(json.dumps({k: v for k, v in info.items() if k not in ("clean_log", "rare_collapse")},
                     indent=2, default=str)[:3000])
    print("\nmatrix:", out["train"].shape, out["test"].shape)
    print("train_clean.csv rows:", len(out["clean"]))
