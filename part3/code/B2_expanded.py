# -*- coding: utf-8 -*-
"""
Part 3 - Experiment B2: Tree ensembles on the NON-LINEAR EXPANDED matrix
=========================================================================
Why this file exists
--------------------
Part 3's original experiments covered:

    A.py  : linear models on the poly/spline expanded matrix (342 columns)
    B.py  : tree ensembles on the raw engineered matrix (250 columns)

so the 2 x 2 design had one hole: **tree ensembles on the expanded matrix**.
This file fills it, and it is the only new cell Part 3 contributes to the
group's unified comparison.

    cell A2  RF / GBR on 250 columns   -> B.py        (existing)
    cell B2  RF / GBR on 342 columns   -> this file   (new)

Protocol (identical to every other part)
----------------------------------------
* data   : Part 1's ``train_features.csv``  (1458 rows x 250 columns)
* target : ``SalePriceLog = log(SalePrice)``
* split  : ``train_test_split(test_size=0.2, random_state=42)``
* CV     : ``KFold(n_splits=10, shuffle=True, random_state=42)``
* leakage: every ``SalePrice``-bearing column is removed before modelling

Outputs (all under ``../tables/``)
---------------------------------
* ``member3_B2_expanded_80_20_mse.csv``
* ``member3_B2_expanded_10fold.csv``
* ``member3_B2_expanded_gbr_feature_importance.csv``
* ``member3_B2_expanded_gbr_permutation_importance.csv``

Author: HUANG Ziyou (Part 3)
"""

import os
import warnings

# joblib 子进程管道需要纯 ASCII 临时目录（本机 Windows 用户名含中文）
_ASCII_TMP = os.path.join(os.environ.get("SystemDrive", "C:") + "\\", "sdsc5001_tmp")
try:
    os.makedirs(_ASCII_TMP, exist_ok=True)
    _ASCII_TMP.encode("ascii")
except Exception:
    _ASCII_TMP = r"C:\sdsc5001_tmp"
    os.makedirs(_ASCII_TMP, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = _ASCII_TMP
os.environ["TMPDIR"] = _ASCII_TMP
os.environ["TEMP"] = _ASCII_TMP
os.environ["TMP"] = _ASCII_TMP

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tables")
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_PATH = os.path.join(ROOT, "..", "part1", "data", "train_features.csv")
CLEAN_PATH = os.path.join(ROOT, "..", "part1", "data", "train_clean.csv")

RANDOM_STATE = 42

# ========== 1. 读取统一数据 ==========
feat = pd.read_csv(FEAT_PATH)
clean = pd.read_csv(CLEAN_PATH)
target = clean[["Id", "SalePriceLog", "SalePrice", "SalePriceOutlier"]]
df = feat.merge(target, on="Id", how="left")

# 【泄漏修复】train_features.csv 自带一列 SalePrice，merge 后变成
# SalePrice_x / SalePrice_y；按列名删 "SalePrice" 会失效，目标会被当特征用。
leak_cols = [
    c for c in df.columns
    if "SalePrice" in c and c not in ["SalePriceLog", "SalePriceOutlier"]
]
print("Leakage columns to drop:", leak_cols)

DROP_OUTLIER = False
if DROP_OUTLIER:
    df = df[df["SalePriceOutlier"] == 0].reset_index(drop=True)

y_log = df["SalePriceLog"].values
print("Number of modeling samples:", len(df))

# ========== 2. 特征分组 ==========
candidate_cont = [
    "GrLivArea", "TotalSF", "TotalBsmtSF", "LotArea",
    "GarageArea", "1stFlrSF", "YearBuilt", "OverallQual",
]
cont_cols = [c for c in candidate_cont if c in df.columns]

exclude = set(cont_cols) | set(leak_cols) | {"Id", "SalePriceLog", "SalePriceOutlier"}
other_cols = [c for c in df.columns if c not in exclude]

drop_for_X = ["Id", "SalePriceLog", "SalePriceOutlier"] + leak_cols
drop_for_X = [c for c in drop_for_X if c in df.columns]
X = df.drop(columns=drop_for_X)

assert not any("SalePrice" in c for c in X.columns), \
    f"Leakage still in X: {[c for c in X.columns if 'SalePrice' in c]}"

nan_in_rest = X[other_cols].isna().sum()
nan_in_rest = nan_in_rest[nan_in_rest > 0]
for c in nan_in_rest.index:
    X[c] = X[c].fillna(X[c].median())


# ========== 3. 与 A.py 完全一致的非线性扩展管线 ==========
def build_preprocessor_expanded(other_cols_use):
    """8 个核心连续变量 -> Polynomial(deg=2) + Spline(n_knots=5, deg=3)，
    其余列原样保留。与 A.py 的 build_preprocessor 参数完全相同。"""
    return ColumnTransformer(
        transformers=[
            ("nl_poly", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ]), cont_cols),
            ("nl_spline", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ]), cont_cols),
            ("rest", "passthrough", other_cols_use),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ========== 4. 80/20 划分 ==========
print("\n===== 80/20 split (aligned with Experiment A / Member 2) =====")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_log, test_size=0.2, random_state=RANDOM_STATE)
print(f"train: {X_tr.shape}, test: {X_te.shape}")

models_B2 = {
    "B2_RF_expanded": Pipeline([
        ("prep", build_preprocessor_expanded(other_cols)),
        ("model", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "B2_GBR_expanded": Pipeline([
        ("prep", build_preprocessor_expanded(other_cols)),
        ("model", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=3, random_state=RANDOM_STATE)),
    ]),
}

# 确认扩展后的特征数
_probe = build_preprocessor_expanded(other_cols).fit_transform(X_tr)
print(f"expanded feature count: {_probe.shape[1]}")

# ========== 5. 80/20 结果 ==========
mse_records = []
for name, pipe in models_B2.items():
    print(f"\n>>> Fitting {name} on 80/20 split ...")
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mse = mean_squared_error(y_te, pred)
    mse_records.append({"model": name, "Test_MSE_log": mse})
    print(f"{name} Test MSE_log = {mse:.6f}")

mse_df = pd.DataFrame(mse_records).sort_values("Test_MSE_log")
print("\n===== Experiment B2: 80/20 Test MSE (342 cols) =====")
print(mse_df.to_string(index=False))
mse_df.to_csv(os.path.join(OUT_DIR, "member3_B2_expanded_80_20_mse.csv"), index=False)

# ========== 6. 10 折 CV ==========
print("\n===== 10-fold Cross-Validation =====")
kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

cv_records = []
for name, pipe in models_B2.items():
    print(f"\n>>> 10-fold CV for {name} ...")
    scores = cross_validate(
        pipe, X, y_log, cv=kf,
        scoring={"rmse": "neg_root_mean_squared_error",
                 "mae": "neg_mean_absolute_error",
                 "r2": "r2"},
        n_jobs=-1, error_score="raise",
    )
    cv_records.append({
        "model": name,
        "RMSE_log": -scores["test_rmse"].mean(),
        "RMSE_log_std": scores["test_rmse"].std(),
        "MAE_log": -scores["test_mae"].mean(),
        "R2": scores["test_r2"].mean(),
        "R2_std": scores["test_r2"].std(),
    })

res_B2 = pd.DataFrame(cv_records).sort_values("RMSE_log")
print("\n===== Experiment B2: 10-fold CV =====")
print(res_B2.to_string(index=False))
res_B2.to_csv(os.path.join(OUT_DIR, "member3_B2_expanded_10fold.csv"), index=False)

# ========== 7. GBR 特征重要性（扩展矩阵） ==========
print("\n>>> Fitting GBR on the expanded matrix for feature importance ...")
gbr_pipe = Pipeline([
    ("prep", build_preprocessor_expanded(other_cols)),
    ("model", GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=3, random_state=RANDOM_STATE)),
])
gbr_pipe.fit(X, y_log)

feature_names = gbr_pipe.named_steps["prep"].get_feature_names_out()
importances = gbr_pipe.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
    .sort_values("importance", ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, "member3_B2_expanded_gbr_feature_importance.csv"),
               index=False)
print("\nGBR Top 20 (expanded matrix):")
print(imp_df.head(20).to_string(index=False))

# ========== 8. 置换重要性 ==========
print("\n>>> Permutation importance on the expanded matrix ...")
# 置换重要性必须作用在【展开后】的矩阵上：permutation_importance 返回的是
# 输入列的个数，若直接喂原始 X 会与展开后的特征名对不上。
_prep_fitted = build_preprocessor_expanded(other_cols).fit(X)
Xte_exp = _prep_fitted.transform(X_te)
gbr_fitted = gbr_pipe.named_steps["model"]
perm = permutation_importance(
    gbr_fitted, Xte_exp, y_te, n_repeats=10, random_state=RANDOM_STATE,
    scoring="neg_mean_squared_error", n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": feature_names,
    "perm_importance_mean": perm.importances_mean,
    "perm_importance_std": perm.importances_std,
}).sort_values("perm_importance_mean", ascending=False)
perm_df.to_csv(
    os.path.join(OUT_DIR, "member3_B2_expanded_gbr_permutation_importance.csv"),
    index=False)
print("\nTop 20 permutation importance (expanded matrix):")
print(perm_df.head(20).to_string(index=False))

print("\nDone. Tables written to", OUT_DIR)
