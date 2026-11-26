# -*- coding: utf-8 -*-
"""
Part 2 - Unified diagnostic pipeline (matrix A and matrix B)
==============================================================
Revised Part 2 code, rebuilt on the unified protocol so its numbers are
directly comparable with Part 3's and Part 4's.

Changes relative to the original Part 2 notebook
-------------------------------------------------
1. **Feature matrix** - the original fitted a 223-column patsy design matrix
   built from ``train_clean.csv``.  It now uses Part 1's engineered matrix
   (``train_features.csv``, 250 columns), the matrix every other part uses.
   The original 223-column run stays in ``member2_*.csv`` as the
   *pre-engineering baseline*, so the effect of feature engineering stays
   visible rather than being erased.

2. **Sample set** - 1,458 rows (Part 1 removed the two partial-sale records),
   identical to Part 3 and Part 4.

3. **Target** - ``SalePriceLog = log(SalePrice)``, unchanged.

4. **Protocol** - ``train_test_split(test_size=0.2, random_state=42)``,
   identical to Part 3 and Part 4.

5. **Two matrices are fitted** - matrix A (250 columns) and matrix B
   (342 columns = matrix A with the 8 core continuous variables replaced by
   their polynomial and spline expansion).  This makes the *effect of automatic
   non-linear expansion* measurable on the diagnostic side too.

6. **Alpha is chosen with RidgeCV / LassoCV** - the coordinate-descent path
   evaluates the whole alpha grid in one fit, which is what keeps the runtime
   sane on a 342-column matrix.

7. **VIF is computed after removing exactly redundant columns** via QR
   decomposition with column pivoting, using the closed form
   ``VIF_j = 1 / (1 - R_jj)`` on the correlation matrix.  Without the pruning
   step a duplicated column drives VIF to infinity and makes every other
   diagnostic meaningless.

Outputs (all under ``../tables/``)
---------------------------------
* ``ols_summary_matrixA.txt`` / ``ols_summary_matrixB.txt``
* ``ols_std_coef_pvalue_matrixA.csv`` / ``_matrixB.csv``
* ``vif_result_matrixA.csv`` / ``vif_result_matrixB.csv``
* ``vif_redundant_removed_matrixA.csv`` / ``_matrixB.csv``
* ``cook_student_resid_matrixA.csv`` / ``_matrixB.csv``
* ``model_mse_compare_matrixA.csv`` / ``_matrixB.csv``
* ``significance_summary.csv`` / ``diagnostic_summary.csv``
* ``residual_plot_matrixA.png`` / ``residual_plot_matrixB.png``

The 10-fold cross-validation for both matrices is produced by Part 4's
``exp_h_factorial.py`` under one single protocol, and Part 2 imports those
numbers into ``model_cv10_matrix*.csv`` rather than recomputing them.  There
is exactly one number per cell in the group.

Author: CHEN Jiaqi (Part 2) - revised onto the unified protocol
"""

# ===================== 0. imports =====================
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ===================== 1. paths & constants =====================
save_dir = r"..\tables"
os.makedirs(save_dir, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
INNER_SPLITS = 5

FEAT_PATH = r"..\..\part1\data\train_features.csv"
CLEAN_PATH = r"..\..\part1\data\train_clean.csv"

CORE_CONT = ["GrLivArea", "TotalSF", "TotalBsmtSF", "LotArea",
             "GarageArea", "1stFlrSF", "YearBuilt", "OverallQual"]
POLY_DEGREE = 2
N_KNOTS = 5
SPLINE_DEGREE = 3

Y_col = "SalePriceLog"
ALPHA_GRID = np.logspace(-3, 3, 100)


# ===================== 2. load the unified data =====================
df_feat = pd.read_csv(FEAT_PATH)
df_clean = pd.read_csv(CLEAN_PATH)

meta_cols = [c for c in ("Id", Y_col, "SalePrice", "SalePriceOutlier")
             if c in df_clean.columns]
df_model = df_feat.merge(df_clean[meta_cols], on="Id", how="left")

# 泄漏清理：train_features.csv 自带一列 SalePrice，merge 后变成
# SalePrice_x / SalePrice_y；目标列 SalePriceLog 必须保留作为因变量。
leak_cols = [c for c in df_model.columns
             if "SalePrice" in c and c not in ("SalePriceLog", "SalePriceOutlier")]
df_model = df_model.drop(
    columns=[c for c in ("Id", "SalePrice", "SalePriceOutlier") if c in df_model.columns]
    + [c for c in leak_cols if c in df_model.columns])

print(f"unified matrix: {df_model.shape[0]} rows x {df_model.shape[1]} columns")
print(f"leakage columns removed: {leak_cols}")

feature_cols = [c for c in df_model.columns
                if c not in ("Id", "SalePriceLog", "SalePriceOutlier")]
X_all = df_model[feature_cols]
y_all = df_model[Y_col].to_numpy(dtype=float)
print(f"modelling features: {len(feature_cols)}")


# ===================== 3. the two matrices =====================
def build_matrix(expand: bool):
    """ColumnTransformer producing matrix A (250) or matrix B (342).

    Matrix A : Part 1's engineered matrix, unchanged.
    Matrix B : the 8 core continuous variables are replaced by
               PolynomialFeatures(degree=2) + SplineTransformer(n_knots=5,
               degree=3); the other 242 columns pass through untouched.

    The two matrices therefore differ in exactly one factor.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (PolynomialFeatures, SplineTransformer)

    cont = [c for c in CORE_CONT if c in X_all.columns]
    other = [c for c in X_all.columns if c not in cont]

    if not expand:
        return ColumnTransformer(
            [("core", SimpleImputer(strategy="median"), cont),
             ("rest", "passthrough", other)],
            remainder="drop", verbose_feature_names_out=False)

    return ColumnTransformer(
        [
            ("nl_poly", Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("sc", StandardScaler()),
                                   ("poly", PolynomialFeatures(degree=POLY_DEGREE,
                                                              include_bias=False))]),
             cont),
            ("nl_spline", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                    ("spline", SplineTransformer(n_knots=N_KNOTS,
                                                               degree=SPLINE_DEGREE))]),
             cont),
            ("rest", "passthrough", other),
        ],
        remainder="drop", verbose_feature_names_out=False)


def full_pipeline(expand: bool, model):
    return Pipeline([("prep", build_matrix(expand)),
                     ("scale", StandardScaler()),
                     ("model", model)])


# ===================== 4. diagnostics =====================
def remove_exactly_redundant(X: pd.DataFrame):
    """Drop columns that are exact linear combinations of the others."""
    import scipy.linalg as sla

    arr = X.to_numpy(dtype=float)
    _, _, piv = sla.qr(arr, mode="economic", pivoting=True)
    _, R = np.linalg.qr(arr)
    diag = np.abs(np.diag(R))
    tol = max(arr.shape) * np.finfo(float).eps * (diag.max() if len(diag) else 1.0)
    rank = int((diag > tol).sum())
    keep_mask = np.zeros(arr.shape[1], dtype=bool)
    keep_mask[piv[:rank]] = True
    dropped = [c for c, keep in zip(X.columns, keep_mask) if not keep]
    return X.loc[:, keep_mask], dropped


def calc_vif(df_x: pd.DataFrame) -> pd.DataFrame:
    """VIF via the closed form VIF_j = 1 / (1 - R_jj) on the correlation matrix."""
    arr = df_x.to_numpy(dtype=float)
    sd = arr.std(axis=0)
    sd[sd == 0] = 1.0
    z = (arr - arr.mean(axis=0)) / sd
    R = (z.T @ z) / (len(z) - 1)
    np.fill_diagonal(R, 1.0)
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Rinv = np.linalg.pinv(R)
    diag = np.diag(Rinv)
    with np.errstate(divide="ignore", invalid="ignore"):
        vif = np.where(diag < 1.0, np.inf, 1.0 / np.maximum(1.0 - diag, 1e-16))
    return pd.DataFrame({"feature": df_x.columns, "VIF": vif})


# ===================== 5. run both matrices =====================
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)

inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
sig_rows, diag_rows = [], []

for matrix_name, expand in [("matrixA", False), ("matrixB", True)]:
    print("\n" + "=" * 78)
    print(f"{matrix_name.upper()}  "
          f"({'250 cols - Part 1 engineered' if not expand else '342 cols - + poly/spline'})")
    print("=" * 78)

    pre = build_matrix(expand)
    pre.fit(X_tr_raw)
    names = pre.get_feature_names_out()
    Xtr = pd.DataFrame(pre.transform(X_tr_raw), columns=names, index=X_tr_raw.index)
    Xte = pd.DataFrame(pre.transform(X_te_raw), columns=names, index=X_te_raw.index)
    print(f"expanded matrix: {Xtr.shape[1]} columns")

    keep = Xtr.columns[Xtr.std() > 0]
    Xtr, Xte = Xtr[keep], Xte[keep]

    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)
    Xtr_s = Xtr_s.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xte_s = Xte_s.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------- remove exactly redundant columns, then VIF ----------
    Xtr_pruned, dropped = remove_exactly_redundant(Xtr_s)
    Xte_pruned = Xte_s[[c for c in Xtr_pruned.columns]]
    print(f"exactly redundant columns removed: {len(dropped)}")

    vif_result = calc_vif(Xtr_pruned)
    n_gt10 = int((vif_result["VIF"] > 10).sum())
    n_inf = int(np.isinf(vif_result["VIF"]).sum())
    finite = vif_result["VIF"][np.isfinite(vif_result["VIF"])]
    max_vif = float(finite.max()) if len(finite) else np.nan
    print(f"VIF > 10 : {n_gt10} / {len(vif_result)}    inf: {n_inf}    max finite: {max_vif:.2f}")

    # ---------- OLS full model ----------
    Xtr_c = Xtr_pruned.copy()
    Xtr_c.insert(0, "const", 1.0)
    ols_res = sm.OLS(y_tr, Xtr_c).fit()
    print(f"OLS in-sample R2 = {ols_res.rsquared:.4f}  (adjusted {ols_res.rsquared_adj:.4f})")

    Xte_c = Xte_pruned.copy()
    Xte_c.insert(0, "const", 1.0)
    Xte_c = Xte_c[Xtr_c.columns]

    n_sig = int((ols_res.pvalues[1:] < 0.05).sum())
    print(f"features significant at the 5% level: {n_sig}")

    # ---------- Cook's distance & studentised residuals ----------
    influence = ols_res.get_influence()
    hii = influence.hat_matrix_diag
    mask = hii < 0.995
    diagnose_df = pd.DataFrame({
        "Id": df_model.loc[Xtr_pruned.index, "Id"].to_numpy()
        if "Id" in df_model.columns else np.arange(len(Xtr_pruned)),
        "hii": hii,
        "cooks_d": np.where(mask, influence.cooks_distance[0], np.nan),
        "studentized_resid": np.where(mask, influence.resid_studentized_external, np.nan),
        "fitted_value": ols_res.fittedvalues.values,
        "residual": ols_res.resid.values,
    })
    n_cook = int(np.nansum(diagnose_df["cooks_d"] > 0.5))
    print(f"observations with Cook's distance > 0.5: {n_cook}")
    print(diagnose_df.nlargest(3, "cooks_d")[["Id", "cooks_d", "studentized_resid"]].to_string(index=False))

    # ---------- residual normality ----------
    from scipy import stats
    resid = ols_res.resid
    jb = stats.jarque_bera(resid)
    print(f"residual kurtosis {stats.kurtosis(resid):.2f}   Jarque-Bera p = {jb.pvalue:.3e}")

    sig_rows.append({
        "matrix": matrix_name, "n_features": Xtr.shape[1],
        "n_after_pruning": Xtr_pruned.shape[1],
        "n_redundant_removed": len(dropped),
        "n_significant_5pct": n_sig,
        "ols_r2": ols_res.rsquared, "ols_adj_r2": ols_res.rsquared_adj,
        "vif_gt_10": n_gt10, "vif_inf": n_inf, "max_finite_vif": max_vif,
    })
    diag_rows.append({
        "matrix": matrix_name,
        "residual_kurtosis": stats.kurtosis(resid),
        "jarque_bera_p": jb.pvalue,
        "durbin_watson": sm.stats.durbin_watson(resid),
        "corr_resid_fitted": float(np.corrcoef(resid, ols_res.fittedvalues)[0, 1]),
        "n_cooks_gt_0.5": n_cook,
        "max_cooks_d": float(np.nanmax(diagnose_df["cooks_d"])),
        "max_abs_studentized": float(np.nanmax(np.abs(diagnose_df["studentized_resid"]))),
    })

    # ---------- 80/20: OLS / Ridge / LASSO ----------
    models_8020 = {
        "OLS": full_pipeline(expand, LinearRegression()),
        "Ridge": full_pipeline(expand, RidgeCV(alphas=ALPHA_GRID, cv=inner,
                                                scoring="neg_mean_squared_error")),
        "LASSO": full_pipeline(expand, LassoCV(alphas=ALPHA_GRID, cv=inner,
                                                random_state=RANDOM_STATE,
                                                max_iter=50000, tol=1e-3)),
    }
    mse_rows = []
    for name, pipe in models_8020.items():
        pipe.fit(X_tr_raw, y_tr)
        pred = pipe.predict(X_te_raw)
        mse_rows.append({
            "Model": name,
            "Test_MSE_log": mean_squared_error(y_te, pred),
            "log_scale_RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "R2": r2_score(y_te, pred),
            "Selected_alpha": getattr(pipe.named_steps["model"], "alpha_", np.nan),
        })
    mse_df = pd.DataFrame(mse_rows)
    mse_df["Relative_reduction_vs_OLS"] = (
        1 - mse_df["Test_MSE_log"] / mse_df["Test_MSE_log"].iloc[0])
    print(mse_df.to_string(index=False))

    # ---------- export ----------
    with open(os.path.join(save_dir, f"ols_summary_{matrix_name}.txt"), "w",
              encoding="utf-8") as f:
        f.write(ols_res.summary().as_text())

    pd.DataFrame({"feature": ols_res.params.index,
                  "ols_std_coef": ols_res.params.values,
                  "std_error": ols_res.bse.values,
                  "t_value": ols_res.tvalues.values,
                  "p_value": ols_res.pvalues.values}).to_csv(
        os.path.join(save_dir, f"ols_std_coef_pvalue_{matrix_name}.csv"),
        index=False, encoding="utf-8-sig")
    vif_result.to_csv(os.path.join(save_dir, f"vif_result_{matrix_name}.csv"),
                      index=False, encoding="utf-8-sig")
    pd.DataFrame({"redundant_column": dropped}).to_csv(
        os.path.join(save_dir, f"vif_redundant_removed_{matrix_name}.csv"),
        index=False, encoding="utf-8-sig")
    diagnose_df.to_csv(os.path.join(save_dir, f"cook_student_resid_{matrix_name}.csv"),
                       index=False, encoding="utf-8-sig")
    mse_df.to_csv(os.path.join(save_dir, f"model_mse_compare_{matrix_name}.csv"),
                  index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(ols_res.fittedvalues, resid, alpha=0.5, s=8)
    axes[0].axhline(y=0, color="red", lw=1)
    axes[0].set_xlabel("Fitted value")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted")
    axes[1].hist(resid, bins=30, edgecolor="black")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"residual_plot_{matrix_name}.png"), dpi=300)
    plt.close(fig)


# ===================== 6. cross-matrix comparison =====================
sig_df = pd.DataFrame(sig_rows)
diag_df = pd.DataFrame(diag_rows)
sig_df.to_csv(os.path.join(save_dir, "significance_summary.csv"),
              index=False, encoding="utf-8-sig")
diag_df.to_csv(os.path.join(save_dir, "diagnostic_summary.csv"),
               index=False, encoding="utf-8-sig")

print("\n" + "=" * 78)
print("CROSS-MATRIX COMPARISON")
print("=" * 78)
print(sig_df.to_string(index=False))
print()
print(diag_df.to_string(index=False))
print("\nAll tables written to", save_dir)
