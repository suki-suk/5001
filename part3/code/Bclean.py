#B clean data
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 1. 读 clean 数据 ==========
CLEAN_PATH = os.path.join(ROOT, "..", "part1", "data", "train_clean.csv")
df = pd.read_csv(CLEAN_PATH)

# 与实验 A/B、Member 2 一致：不额外剔除异常点
DROP_OUTLIER = False
if DROP_OUTLIER:
    df = df[df["SalePriceOutlier"] == 0].reset_index(drop=True)

y_log = df["SalePriceLog"].values
print("Number of modeling samples:", len(df))

# ========== 2. 特征分组（基于 clean 原始列） ==========
# 要排除的目标/标记/ID 列
drop_cols = ["Id", "SalePrice", "SalePriceLog", "SalePriceOutlier"]
drop_cols = [c for c in drop_cols if c in df.columns]

# 分类变量（clean 里是字符串）
cat_cols = [
    "MSZoning", "Neighborhood", "BldgType", "HouseStyle", "RoofStyle",
    "Foundation", "GarageType", "SaleType", "SaleCondition", "CentralAir",
    "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",
    "LandSlope", "Condition1", "Condition2", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "BsmtQual",
    "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
    "HeatingQC", "Electrical", "KitchenQual", "Functional", "FireplaceQu",
    "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC",
    "Fence", "MiscFeature",
]
cat_cols = [c for c in cat_cols if c in df.columns]

# 数值变量（其余全部）
num_cols = [
    c for c in df.columns
    if c not in cat_cols + drop_cols
]
print(f"cat_cols: {len(cat_cols)}, num_cols: {len(num_cols)}")

X = df.drop(columns=drop_cols)

# ========== 3. 管线：分类用 OrdinalEncoder，数值中位数填补 ==========
def build_preprocessor_clean():
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

# ========== 4. 80/20 划分（与实验 A/B、Member 2 一致） ==========
print("\n===== 80/20 split (aligned with Experiment A/B / Member 2) =====")
X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"train: {X_tr.shape}, test: {X_te.shape}")

models_B2 = {
    "B2a_RF_clean": Pipeline([
        ("prep", build_preprocessor_clean()),
        ("model", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2,
            random_state=42, n_jobs=-1)),
    ]),
    "B2b_GBR_clean": Pipeline([
        ("prep", build_preprocessor_clean()),
        ("model", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=3, random_state=42)),
    ]),
}

mse_records = []
for name, pipe in models_B2.items():
    print(f"\n>>> Fitting {name} on 80/20 split ...")
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mse = mean_squared_error(y_te, pred)
    mse_records.append({"model": name, "Test_MSE_log": mse})
    print(f"{name} Test MSE_log = {mse:.6f}")

mse_df = pd.DataFrame(mse_records).sort_values("Test_MSE_log")
print("\n===== Experiment B2 (clean data): 80/20 Test MSE =====")
print(mse_df.to_string(index=False))
mse_df.to_csv(os.path.join(OUT_DIR, "member3_B2_clean_80_20_mse.csv"), index=False)

# ========== 5. 10 折 CV ==========
print("\n===== 10-fold Cross-Validation =====")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_records = []
for name, pipe in models_B2.items():
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

res_B2 = pd.DataFrame(cv_records).sort_values("RMSE_log")
print("\n===== Experiment B2 (clean data): 10-fold CV =====")
print(res_B2.to_string(index=False))
res_B2.to_csv(os.path.join(OUT_DIR, "member3_results_B2_clean_10fold.csv"), index=False)

# ========== 6. GBR 特征重要性 ==========
print("\n>>> Fitting GBR to extract feature importance ...")
gbr_pipe = Pipeline([
    ("prep", build_preprocessor_clean()),
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

imp_df.to_csv(os.path.join(OUT_DIR, "member3_B2_gbr_clean_feature_importance.csv"), index=False)
print("\nGBR Top 20 Feature Importance (clean data):")
print(imp_df.head(20).to_string(index=False))

print(f"\nExperiment B2 results saved to: {OUT_DIR}")

'''result
Number of modeling samples: 1460
cat_cols: 43, num_cols: 36

===== 80/20 split (aligned with Experiment A/B / Member 2) =====
train: (1168, 79), test: (292, 79)

Fitting B2a_RF_clean on 80/20 split ...
B2a_RF_clean Test MSE_log = 0.021152

Fitting B2b_GBR_clean on 80/20 split ...
B2b_GBR_clean Test MSE_log = 0.018851

===== Experiment B2 (clean data): 80/20 Test MSE =====
        model  Test_MSE_log
B2b_GBR_clean      0.018851
 B2a_RF_clean      0.021152

===== 10-fold Cross-Validation =====

10-fold CV for B2a_RF_clean ...
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   40.5s finished
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.

10-fold CV for B2b_GBR_clean ...
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   57.8s finished

===== Experiment B2 (clean data): 10-fold CV =====
        model  RMSE_log  RMSE_log_std  MAE_log       R2   R2_std
B2b_GBR_clean  0.125307      0.020233 0.085092 0.897559 0.031327
 B2a_RF_clean  0.139102      0.023937 0.093937 0.873518 0.041384

Fitting GBR to extract feature importance ...

GBR Top 20 Feature Importance (clean data):
      feature  importance
  OverallQual    0.442715
    GrLivArea    0.174648
  TotalBsmtSF    0.051758
   GarageCars    0.042806
    YearBuilt    0.035025
   BsmtFinSF1    0.025628
 YearRemodAdd    0.024941
 GarageFinish    0.023177
      LotArea    0.019635
  OverallCond    0.017186
   GarageArea    0.015742
     1stFlrSF    0.015379
   CentralAir    0.014497
   Fireplaces    0.014232
     MSZoning    0.010908
  KitchenQual    0.006527
     2ndFlrSF    0.005504
SaleCondition    0.004025
     BsmtQual    0.003898
  GarageYrBlt    0.003732

Experiment B2 results saved to: <part3>/tables
'''