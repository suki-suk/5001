# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 23:56:48 2026

@author: Dell
"""

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 1. 读数据 ==========
FEAT_PATH  = os.path.join(ROOT, "..", "part1", "data", "train_features.csv")
CLEAN_PATH = os.path.join(ROOT, "..", "part1", "data", "train_clean.csv")

feat  = pd.read_csv(FEAT_PATH)
clean = pd.read_csv(CLEAN_PATH)
target = clean[["Id", "SalePriceLog", "SalePrice", "SalePriceOutlier"]]
df = feat.merge(target, on="Id", how="left")

# 【一致原则 1】与 Member 2 一致：不额外剔除异常点
DROP_OUTLIER = False
if DROP_OUTLIER:
    df = df[df["SalePriceOutlier"] == 0].reset_index(drop=True)

y_log = df["SalePriceLog"].values
print("建模样本数:", len(df))   # 应为 1458，与 feature 文件一致

# ========== 2. 列分组 ==========
candidate_cont = [
    "GrLivArea", "TotalSF", "TotalBsmtSF", "LotArea",
    "GarageArea", "1stFlrSF", "YearBuilt", "OverallQual",
]
cont_cols = [c for c in candidate_cont if c in df.columns]

# ========== 3b. 泄漏修复：排除所有 SalePrice 相关列 ==========
# train_features.csv 自带一列 SalePrice，与 train_clean.csv 合并后该列会变成
# SalePrice_x / SalePrice_y，导致下面按列名删除 "SalePrice" 时找不到它，
# 目标变量会被直接当作特征使用（与 SalePriceLog 相关 0.95），使测试误差虚高约 3 倍。
# 这里的做法与 B.py 一致：凡是含 "SalePrice" 的列，除目标列与标记列外一律剔除。
leak_cols = [
    c for c in df.columns
    if "SalePrice" in c and c not in ["SalePriceLog", "SalePriceOutlier"]
]
print("Leakage columns to drop:", leak_cols)

exclude = set(cont_cols) | set(leak_cols) | {"Id", "SalePriceLog", "SalePriceOutlier"}
other_cols = [c for c in df.columns if c not in exclude]

# ========== 3c. 特征管线 ==========
def build_preprocessor(other_cols_use):
    return ColumnTransformer(
        transformers=[
            ("nl_poly", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ]), cont_cols),
            ("nl_spline", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ]), cont_cols),
            ("rest", "passthrough", other_cols_use),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

drop_for_X = ["Id", "SalePriceLog", "SalePriceOutlier"] + leak_cols
drop_for_X = [c for c in drop_for_X if c in df.columns]
X = df.drop(columns=drop_for_X)

# 双重保险：X 里不允许出现任何 SalePrice 相关列
assert not any("SalePrice" in c for c in X.columns), \
    f"Leakage still in X: {[c for c in X.columns if 'SalePrice' in c]}"

# NaN 处理（passthrough 列）
nan_in_rest = X[other_cols].isna().sum()
nan_in_rest = nan_in_rest[nan_in_rest > 0]
for c in nan_in_rest.index:
    X[c] = X[c].fillna(X[c].median())

# ========== 4. 与 Member 2 对齐的 80/20 split ==========
print("\n===== 与 Member 2 对齐：80/20 split =====")
X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"train: {X_tr.shape}, test: {X_te.shape}")

# 【一致原则 4】用 RidgeCV/LassoCV 在训练集内部选 alpha（与 Member 2 相同网格）
prep_for_cv = build_preprocessor(other_cols)

# 先 transform 训练集（注意：这里为了用 RidgeCV 直接选 alpha，需先 fit_transform）
X_tr_prep = prep_for_cv.fit_transform(X_tr)
X_te_prep = prep_for_cv.transform(X_te)

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), scoring="neg_mean_squared_error")
ridge_cv.fit(X_tr_prep, y_tr)
ridge_pred = ridge_cv.predict(X_te_prep)

lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 100), random_state=42, max_iter=10000)
lasso_cv.fit(X_tr_prep, y_tr)
lasso_pred = lasso_cv.predict(X_te_prep)

ols = LinearRegression().fit(X_tr_prep, y_tr)
ols_pred = ols.predict(X_te_prep)

# 【一致原则 5】主指标：log 尺度 MSE（与 Member 2 一致）
mse_ols   = mean_squared_error(y_te, ols_pred)
mse_ridge = mean_squared_error(y_te, ridge_pred)
mse_lasso = mean_squared_error(y_te, lasso_pred)

print(f"\nRidge 最优 alpha: {ridge_cv.alpha_:.6f}")
print(f"Lasso 最优 alpha: {lasso_cv.alpha_:.6f}")
print(f"OLS   MSE_log: {mse_ols:.6f}")
print(f"Ridge MSE_log: {mse_ridge:.6f}")
print(f"Lasso MSE_log: {mse_lasso:.6f}")

mse_df = pd.DataFrame({
    "Model": ["OLS", "Ridge", "Lasso"],
    "Test_MSE_log": [mse_ols, mse_ridge, mse_lasso],
    "Selected_alpha": [np.nan, ridge_cv.alpha_, lasso_cv.alpha_],
})
mse_df.to_csv(os.path.join(OUT_DIR, "member3_A_80_20_mse.csv"), index=False)

# ========== 5. 保留你的 10 折 CV（补充，更稳健） ==========
print("\n===== 补充：10 折 CV =====")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

models_A = {
    "A1_OLS": Pipeline([("prep", build_preprocessor(other_cols)), ("model", LinearRegression())]),
    "A2_Ridge": Pipeline([("prep", build_preprocessor(other_cols)), ("model", Ridge(alpha=ridge_cv.alpha_, random_state=42))]),
    "A3_Lasso": Pipeline([("prep", build_preprocessor(other_cols)), ("model", Lasso(alpha=lasso_cv.alpha_, max_iter=50000, tol=1e-3, random_state=42))]),
}

results = []
for name, pipe in models_A.items():
    scores = cross_validate(
        pipe, X, y_log, cv=kf,
        scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=1, error_score="raise",
    )
    results.append({
        "model": name,
        "RMSE_log": -scores["test_rmse"].mean(),
        "MAE_log": -scores["test_mae"].mean(),
        "R2": scores["test_r2"].mean(),
    })

res_A = pd.DataFrame(results).sort_values("RMSE_log")
print(res_A.to_string(index=False))
res_A.to_csv(os.path.join(OUT_DIR, "member3_results_A_10fold.csv"), index=False)

# ========== 6. Lasso 变量选择（用 80/20 训练集上选出的 alpha） ==========
lasso_final = Pipeline([
    ("prep", build_preprocessor(other_cols)),
    ("model", Lasso(alpha=lasso_cv.alpha_, max_iter=50000, tol=1e-3, random_state=42)),
])
lasso_final.fit(X, y_log)
feature_names = lasso_final.named_steps["prep"].get_feature_names_out()
coefs = lasso_final.named_steps["model"].coef_
n_nonzero = int(np.sum(coefs != 0))
print(f"\nLasso Total Lasso features: {len(coefs)}, Non-zero: {n_nonzero}, Sparsity: {1-n_nonzero/len(coefs):.2%}")

coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df[coef_df["coef"] != 0].sort_values("abs_coef", ascending=False)
coef_df.to_csv(os.path.join(OUT_DIR, "member3_lasso_selected_features.csv"), index=False)
print("\nTop 20 selected features by |coef|:")
print(coef_df.head(20).to_string(index=False))
print(f"\n结果已保存到 {OUT_DIR}")