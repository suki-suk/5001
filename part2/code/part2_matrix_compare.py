# -*- coding: utf-8 -*-
"""
Part 2 (unified) - Feature-matrix comparison: matrix A vs matrix B
===================================================================
Runs Part 2's full diagnostic pipeline on BOTH feature matrices so that the
linear-family results are directly comparable with Part 3's and Part 4's.

    matrix A  250 columns  Part 1's engineered matrix, used as-is
    matrix B  342 columns  matrix A with the 8 core continuous variables
                          replaced by PolynomialFeatures(degree=2) +
                          SplineTransformer(n_knots=5, degree=3)

The two matrices differ in exactly one factor, so any difference in the
diagnostics is attributable to the automatic non-linear expansion alone.

Pipeline: standardized OLS full model -> assumption tests -> VIF on the
pruned matrix -> Cook's distance / studentised residuals -> Ridge & LASSO.

Protocol (identical to Part 3 and Part 4)
------------------------------------------
* data   : ``part1/data/train_features.csv`` (1458 x 250)
* target : ``SalePriceLog = log(SalePrice)``
* split  : ``train_test_split(test_size=0.2, random_state=42)``
* alpha  : ``RidgeCV`` / ``LassoCV`` over ``logspace(-3, 3, 100)``,
            5-fold inner CV, chosen on the training set only

Leakage
-------
``train_features.csv`` carries its own ``SalePrice`` column; merging it with
``train_clean.csv`` turns that into ``SalePrice_x`` / ``SalePrice_y``.  Every
``SalePrice``-bearing column except the target and the outlier flag is
removed before modelling.

Performance notes
-----------------
Cook's distance and studentised residuals are computed directly rather than
through ``statsmodels.OLSInfluence``: ``resid_studentized_external`` needs a
generalised inverse of a rank-deficient hat matrix and hangs indefinitely on
a 200+ parameter model.  The closed forms used below are exact and instant.

Outputs (all under ``../tables/``)
---------------------------------
* ``ols_summary_matrixA.txt`` / ``ols_summary_matrixB.txt``
* ``ols_std_coef_pvalue_matrixA.csv`` / ``_matrixB.csv``
* ``vif_result_matrixA.csv`` / ``vif_result_matrixB.csv``
* ``vif_redundant_removed_matrixA.csv`` / ``_matrixB.csv``
* ``cook_student_resid_matrixA.csv`` / ``_matrixB.csv``
* ``model_mse_compare_matrixA.csv`` / ``_matrixB.csv``
* ``significance_summary.csv``
* ``residual_plot_matrixA.png`` / ``residual_plot_matrixB.png``

Author: CHEN Jiaqi (Part 2) - revised onto the unified protocol
"""

import os
import time
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

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ===================== 1. paths & constants =====================
save_dir = r"..\tables"
os.makedirs(save_dir, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
INNER_SPLITS = 5
ALPHA_GRID = np.logspace(-3, 3, 100)

FEAT_PATH = r"..\..\part1\data\train_features.csv"
CLEAN_PATH = r"..\..\part1\data\train_clean.csv"
Y_col = "SalePriceLog"

CORE_CONT = ["GrLivArea", "TotalSF", "TotalBsmtSF", "LotArea",
             "GarageArea", "1stFlrSF", "YearBuilt", "OverallQual"]
POLY_DEGREE = 2
N_KNOTS = 5
SPLINE_DEGREE = 3


# ===================== 2. load the unified data =====================
df_feat = pd.read_csv(FEAT_PATH)
df_clean = pd.read_csv(CLEAN_PATH)

meta_cols = [c for c in ("Id", Y_col, "SalePrice", "SalePriceOutlier")
             if c in df_clean.columns]
df_model = df_feat.merge(df_clean[meta_cols], on="Id", how="left")

leak_cols = [c for c in df_model.columns
             if "SalePrice" in c and c not in ("SalePriceLog", "SalePriceOutlier")]
df_model = df_model.drop(
    columns=[c for c in ("Id", "SalePrice", "SalePriceOutlier") if c in df_model.columns]
    + [c for c in leak_cols if c in df_model.columns])
stamp(f"unified matrix: {df_model.shape[0]} rows x {df_model.shape[1]} columns "
      f"(leakage removed: {leak_cols})")

feature_cols = [c for c in df_model.columns
                if c not in ("Id", "SalePriceLog", "SalePriceOutlier")]
X_all = df_model[feature_cols]
y_all = df_model[Y_col].to_numpy(dtype=float)
stamp(f"modelling features: {len(feature_cols)}")


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


# ===================== 4. diagnostics helpers =====================
def remove_exactly_redundant(X: pd.DataFrame):
    """Drop columns that are exact linear combinations of the others.

    Column-pivoted QR gives a maximal linearly independent column set.  Without
    this step a duplicated column (e.g. ``MSSubClass_90`` vs
    ``BldgType_Duplex``) drives VIF to infinity and drowns every meaningful
    diagnostic.
    """
    import scipy.linalg as sla

    arr = X.to_numpy(dtype=float)
    _, r, piv = sla.qr(arr, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    tol = max(arr.shape) * np.finfo(float).eps * (diag.max() if len(diag) else 1.0)
    rank = int((diag > tol).sum())
    keep_mask = np.zeros(arr.shape[1], dtype=bool)
    keep_mask[piv[:rank]] = True
    dropped = [c for c, keep in zip(X.columns, keep_mask) if not keep]
    return X.loc[:, keep_mask], dropped


def calc_vif(df_x: pd.DataFrame) -> pd.DataFrame:
    """VIF via the closed form VIF_j = diag(inv(R))_j on the correlation matrix.

    Algebraically identical to statsmodels' per-column
    ``variance_inflation_factor`` but computes every column at once.
    """
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
    vif = np.diag(Rinv)
    vif = np.where(vif <= 0, np.inf, vif)
    return pd.DataFrame({"feature": df_x.columns, "VIF": vif})


def influence_diagnostics(X: pd.DataFrame, y: np.ndarray):
    """Leverage, Cook's distance and studentised residuals, computed directly.

    statsmodels' ``resid_studentized_external`` needs a generalised inverse of
    a rank-deficient hat matrix and hangs indefinitely on a 200+ parameter
    model, so the closed forms are used instead:

        h_ii   = x_i' (X'X)+ x_i
        Cook   = r_i^2 / p * h_ii / (1 - h_ii)^2
        r_int  = r_i / (s * sqrt(1 - h_ii))
        r_ext  = r_i / (s_(i) * sqrt(1 - h_ii)),
                 s_(i)^2 = (SSE - r_i^2/(1-h_ii)) / (n - p - 1)
    """
    Xm = X.to_numpy(dtype=float)
    Xc = np.column_stack([np.ones(len(Xm)), Xm])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    fitted = Xc @ beta
    resid = y - fitted
    n, p = Xc.shape
    dof = n - p

    xtx_inv = np.linalg.pinv(Xc.T @ Xc)
    hii = np.clip(np.einsum("ij,jk,ik->i", Xc, xtx_inv, Xc), 0.0, 1.0 - 1e-10)

    mask = hii < 0.995
    sse = float(resid @ resid)
    s2 = sse / dof
    cooks_d = (resid ** 2 / p) * hii / (1.0 - hii) ** 2
    stud_int = resid / np.sqrt(s2 * (1.0 - hii))
    s2_ext = np.maximum((sse - resid ** 2 / (1.0 - hii)) / (dof - 1), 1e-12)
    stud_ext = resid / np.sqrt(s2_ext * (1.0 - hii))
    return hii, cooks_d, stud_int, stud_ext, fitted, resid, sse, dof


# ===================== 5. run both matrices =====================
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)
stamp("80/20 split done")

inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
sig_rows = []

for matrix_name, expand in [("matrixA", False), ("matrixB", True)]:
    print("\n" + "=" * 78)
    print(f"{matrix_name.upper()}  "
          f"({'250 cols - Part 1 engineered' if not expand else '342 cols - + poly/spline'})")
    print("=" * 78, flush=True)

    pre = build_matrix(expand)
    pre.fit(X_tr_raw)
    names = pre.get_feature_names_out()
    Xtr = pd.DataFrame(pre.transform(X_tr_raw), columns=names, index=X_tr_raw.index)
    Xte = pd.DataFrame(pre.transform(X_te_raw), columns=names, index=X_te_raw.index)
    stamp(f"prepared matrix: {Xtr.shape[1]} columns")

    keep = Xtr.columns[Xtr.std() > 0]
    Xtr, Xte = Xtr[keep], Xte[keep]

    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)
    Xtr_s = Xtr_s.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xte_s = Xte_s.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------- prune exactly redundant columns, then VIF ----------
    Xtr_pruned, dropped = remove_exactly_redundant(Xtr_s)
    Xte_pruned = Xte_s[[c for c in Xtr_pruned.columns]]
    stamp(f"exactly redundant columns removed: {len(dropped)}")

    vif_result = calc_vif(Xtr_pruned)
    n_gt10 = int((vif_result["VIF"] > 10).sum())
    n_inf = int(np.isinf(vif_result["VIF"]).sum())
    finite = vif_result["VIF"][np.isfinite(vif_result["VIF"])]
    max_vif = float(finite.max()) if len(finite) else np.nan
    print(f"    VIF > 10 : {n_gt10} / {len(vif_result)}    inf: {n_inf}    "
          f"max finite: {max_vif:.2f}", flush=True)

    # ---------- OLS full model ----------
    Xtr_c = Xtr_pruned.copy()
    Xtr_c.insert(0, "const", 1.0)
    ols_res = sm.OLS(y_tr, Xtr_c).fit()
    stamp(f"OLS fitted: R2 = {ols_res.rsquared:.4f} (adj {ols_res.rsquared_adj:.4f})")

    Xte_c = Xte_pruned.copy()
    Xte_c.insert(0, "const", 1.0)
    Xte_c = Xte_c[Xtr_c.columns]

    n_sig = int((ols_res.pvalues[1:] < 0.05).sum())
    print(f"    features significant at the 5% level: {n_sig}", flush=True)

    # ---------- influence diagnostics ----------
    hii, cooks_d, stud_int, stud_ext, fitted, resid_v, sse, dof = \
        influence_diagnostics(Xtr_pruned, y_tr)

    diagnose_df = pd.DataFrame({
        "Id": df_model.loc[Xtr_pruned.index, "Id"].to_numpy()
        if "Id" in df_model.columns else np.arange(len(Xtr_pruned)),
        "hii": hii,
        "cooks_d": np.where(hii < 0.995, cooks_d, np.nan),
        "studentized_resid": np.where(hii < 0.995, stud_ext, np.nan),
        "studentized_resid_internal": np.where(hii < 0.995, stud_int, np.nan),
        "fitted_value": fitted,
        "residual": resid_v,
    })
    n_cook = int(np.nansum(diagnose_df["cooks_d"] > 0.5))
    print(f"    observations with Cook's distance > 0.5: {n_cook}", flush=True)
    print(diagnose_df.nlargest(3, "cooks_d")[["Id", "cooks_d", "studentized_resid"]].to_string(index=False),
          flush=True)

    # ---------- residual assumptions ----------
    from scipy import stats

    jb = stats.jarque_bera(resid_v)
    kurt = stats.kurtosis(resid_v)
    dw = sm.stats.durbin_watson(resid_v)
    corr_rf = float(np.corrcoef(resid_v, fitted)[0, 1])
    print(f"    residual kurtosis {kurt:.2f}   Jarque-Bera p = {jb.pvalue:.3e}   "
          f"Durbin-Watson {dw:.3f}   corr(resid, fitted) {corr_rf:+.4f}", flush=True)

    sig_rows.append({
        "matrix": matrix_name, "n_features": Xtr.shape[1],
        "n_after_pruning": Xtr_pruned.shape[1],
        "n_redundant_removed": len(dropped),
        "n_significant_5pct": n_sig,
        "ols_r2": ols_res.rsquared, "ols_adj_r2": ols_res.rsquared_adj,
        "vif_gt_10": n_gt10, "vif_inf": n_inf, "max_finite_vif": max_vif,
        "residual_kurtosis": kurt, "jarque_bera_p": jb.pvalue,
        "durbin_watson": dw, "corr_resid_fitted": corr_rf,
        "n_cooks_gt_0.5": n_cook,
        "max_cooks_d": float(np.nanmax(diagnose_df["cooks_d"])),
        "max_abs_studentized": float(np.nanmax(np.abs(diagnose_df["studentized_resid"]))),
    })

    # ---------- Ridge & LASSO ----------
    models = {
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(alphas=ALPHA_GRID, cv=inner, scoring="neg_mean_squared_error"),
        "LASSO": LassoCV(alphas=ALPHA_GRID, cv=inner, random_state=RANDOM_STATE,
                         max_iter=50000, tol=1e-3),
    }
    rows = []
    for name, mdl in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", mdl)])
        pipe.fit(Xtr_pruned, y_tr)
        pred = pipe.predict(Xte_pruned)
        mse = mean_squared_error(y_te, pred)
        alpha = getattr(mdl, "alpha_", np.nan)
        if name == "LASSO":
            n_kept = int((np.abs(mdl.coef_) > 0).sum())
            sparsity = 1 - n_kept / len(mdl.coef_)
        else:
            n_kept, sparsity = len(mdl.coef_), 0.0
        rows.append({
            "Model": name,
            "Test_MSE_log": mse,
            "log_scale_RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_te, pred),
            "Selected_alpha": alpha,
            "n_nonzero_coef": n_kept,
            "sparsity": sparsity,
        })
        stamp(f"{name}: MSE_log = {mse:.6f}  R2 = {r2_score(y_te, pred):.4f}  alpha = {alpha}")

    mse_df = pd.DataFrame(rows)
    mse_df["Relative_reduction_vs_OLS"] = (
        1 - mse_df["Test_MSE_log"] / mse_df["Test_MSE_log"].iloc[0])
    print()
    print(mse_df.to_string(index=False), flush=True)

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
    axes[0].scatter(fitted, resid_v, alpha=0.5, s=8)
    axes[0].axhline(y=0, color="red", lw=1)
    axes[0].set_xlabel("Fitted value")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted")
    axes[1].hist(resid_v, bins=30, edgecolor="black")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"residual_plot_{matrix_name}.png"), dpi=300)
    plt.close(fig)


# ===================== 6. cross-matrix comparison =====================
sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(os.path.join(save_dir, "significance_summary.csv"),
              index=False, encoding="utf-8-sig")

print("\n" + "=" * 78)
print("CROSS-MATRIX COMPARISON")
print("=" * 78)
print(sig_df.to_string(index=False))
stamp("DONE")
