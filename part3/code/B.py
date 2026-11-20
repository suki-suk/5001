import os
import numpy as np
import pandas as pd
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

# 只 merge 目标列和标记列，不 merge 原始 SalePrice，避免产生 SalePrice_x/_y
target = clean[["Id", "SalePriceLog", "SalePriceOutlier"]]
df = feat.merge(target, on="Id", how="left")

# 如果 feature 文件自带 SalePrice，这里会暴露出来
print("Columns containing 'SalePrice':", [c for c in df.columns if "SalePrice" in c])

# 与实验 A / Member 2 一致：不额外剔除异常点
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

# ========== 3. 泄漏修复：排除所有 SalePrice 相关列 ==========
# 目标列 SalePriceLog、标记列 SalePriceOutlier 保留在 df 中用于评估，但不进 X
# 其余任何含 SalePrice 的列（SalePrice、SalePrice_x、SalePrice_y 等）全部视为泄漏，排除
leak_cols = [
    c for c in df.columns
    if "SalePrice" in c and c not in ["SalePriceLog", "SalePriceOutlier"]
]
print("Leakage columns to drop:", leak_cols)

exclude = set(cont_cols) | set(leak_cols) | {"Id", "SalePriceLog", "SalePriceOutlier"}
other_cols = [c for c in df.columns if c not in exclude]

drop_for_X = ["Id", "SalePriceLog", "SalePriceOutlier"] + leak_cols
drop_for_X = [c for c in drop_for_X if c in df.columns]
X = df.drop(columns=drop_for_X)

# 双重保险：X 里不允许出现任何 SalePrice 相关列
assert not any("SalePrice" in c for c in X.columns), \
    f"Leakage still in X: {[c for c in X.columns if 'SalePrice' in c]}"

# NaN 处理
nan_cols = X.isna().sum()
nan_cols = nan_cols[nan_cols > 0]
for c in nan_cols.index:
    X[c] = X[c].fillna(X[c].median())

# ========== 4. 特征管线（只保留树模型原特征） ==========
def build_preprocessor_tree(other_cols_use):
    return ColumnTransformer(
        transformers=[
            ("cont", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), cont_cols),
            ("rest", "passthrough", other_cols_use),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

# ========== 5. 80/20 划分（与实验 A、Member 2 一致） ==========
print("\n===== 80/20 split (aligned with Experiment A / Member 2) =====")
X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"train: {X_tr.shape}, test: {X_te.shape}")

models_B = {
    "B1_RF": Pipeline([
        ("prep", build_preprocessor_tree(other_cols)),
        ("model", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2,
            random_state=42, n_jobs=-1)),
    ]),
    "B2_GBR": Pipeline([
        ("prep", build_preprocessor_tree(other_cols)),
        ("model", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=3, random_state=42)),
    ]),
}

mse_records = []
for name, pipe in models_B.items():
    print(f"\n>>> Fitting {name} on 80/20 split ...")
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mse = mean_squared_error(y_te, pred)
    mse_records.append({"model": name, "Test_MSE_log": mse})
    print(f"{name} Test MSE_log = {mse:.6f}")

mse_df = pd.DataFrame(mse_records).sort_values("Test_MSE_log")
print("\n===== Experiment B: 80/20 Test MSE =====")
print(mse_df.to_string(index=False))
mse_df.to_csv(os.path.join(OUT_DIR, "member3_B_80_20_mse.csv"), index=False)

# ========== 6. 10 折 CV ==========
print("\n===== 10-fold Cross-Validation =====")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_records = []
for name, pipe in models_B.items():
    print(f"\n>>> 10-fold CV for {name} ...")
    scores = cross_validate(
        pipe, X, y_log, cv=kf,
        scoring={"rmse": "neg_root_mean_squared_error",
                 "mae": "neg_mean_absolute_error",
                 "r2": "r2"},
        n_jobs=1, verbose=1, error_score="raise",
    )
    cv_records.append({
        "model": name,
        "RMSE_log": -scores["test_rmse"].mean(),
        "RMSE_log_std": scores["test_rmse"].std(),
        "MAE_log": -scores["test_mae"].mean(),
        "R2": scores["test_r2"].mean(),
        "R2_std": scores["test_r2"].std(),
    })

res_B = pd.DataFrame(cv_records).sort_values("RMSE_log")
print("\n===== Experiment B: 10-fold CV =====")
print(res_B.to_string(index=False))
res_B.to_csv(os.path.join(OUT_DIR, "member3_results_B_10fold.csv"), index=False)

# ========== 7. 特征重要性（GBR） ==========
print("\n>>> Fitting GBR to extract feature importance ...")
gbr_pipe = Pipeline([
    ("prep", build_preprocessor_tree(other_cols)),
    ("model", GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=3, random_state=42)),
])
gbr_pipe.fit(X, y_log)

feature_names = gbr_pipe.named_steps["prep"].get_feature_names_out()
importances = gbr_pipe.named_steps["model"].feature_importances_

imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

imp_df.to_csv(os.path.join(OUT_DIR, "member3_B_gbr_feature_importance.csv"), index=False)
print("\nGBR Top 20 Feature Importance:")
print(imp_df.head(20).to_string(index=False))

# ========== 8. 置换重要性 ==========
from sklearn.inspection import permutation_importance

print("\n>>> Computing permutation importance on test set (B2 GBR) ...")
gbr_fitted = models_B["B2_GBR"]
gbr_fitted.fit(X_tr, y_tr)
perm = permutation_importance(
    gbr_fitted, X_te, y_te,
    n_repeats=10, random_state=42, n_jobs=1,
    scoring="neg_mean_squared_error",
)
perm_df = pd.DataFrame({
    "feature": X_te.columns,
    "perm_importance_mean": perm.importances_mean,
    "perm_importance_std": perm.importances_std,
}).sort_values("perm_importance_mean", ascending=False)

perm_df.to_csv(os.path.join(OUT_DIR, "member3_B_gbr_permutation_importance.csv"), index=False)
print("\nGBR Top 20 Permutation Importance:")
print(perm_df.head(20).to_string(index=False))

print(f"\nExperiment B results saved to: {OUT_DIR}")