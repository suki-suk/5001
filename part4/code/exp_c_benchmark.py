# -*- coding: utf-8 -*-
"""
Part 4 · 实验 C：统一口径的模型对比（Unified Benchmark）
==========================================================
为什么需要这一步？
------------------
Part2 与 Part3 各自报了数字，但两者不完全可比：

* Part2 在 **原始 train_clean.csv** 上用 patsy 做 one-hot + Treatment 参照组，
  再标准化，拟合 223 个特征的全模型，评价用的是 Kaggle 官方 test 集；
* Part3 在 **Part1 的 252 列特征矩阵** 上做多项式/样条扩展，
  评价用的是 80/20 划分与 10 折 CV。

两边的样本数（1,168 vs 1,166）、特征数（223 vs 336）、评价集（官方 test vs 内部划分）
都不同，所以"LASSO 测试 MSE 0.0179"和"LASSO 测试 MSE 0.0043"放在一起比较是不成立的。

本脚本做的是 **把所有人拉到同一张桌子上**：
同一份数据（Part1 特征矩阵）、同一个划分（80/20, random_state=42）、
同一套指标（RMSE / MAE / R² / RMSLE）、同一套交叉验证（10 折 KFold）。

产出
----
* ``output/unified_80_20.csv``     —— 五模型 80/20 结果
* ``output/unified_cv10.csv``      —— 五模型 10 折 CV 结果（均值 ± 标准差）
* ``output/unified_benchmark.csv`` —— 合并总表
* ``figures/fig01_unified_benchmark.png``

作者：Yifei Lv（Part4）
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C


# ---------------------------------------------------------------- 模型定义
def build_models(X: pd.DataFrame, y: np.ndarray):
    """构造五个候选模型，全部使用同一份特征矩阵、同一套协议。

    * OLS / Ridge / Lasso  —— 线性族（线性回归 + 两种收缩惩罚）
    * Random Forest / GBR —— 树集成族

    正则化强度 alpha 由 **训练集内部** 的 RidgeCV / LassoCV 选出，
    绝不接触验证集或测试集。
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # ---- alpha 只在训练集内部用 CV 选，网格与 Part3 保持一致 ----
    from sklearn.model_selection import GridSearchCV, KFold

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    alpha_grid = np.logspace(-3, 3, 100)

    ridge = GridSearchCV(
        Ridge(random_state=C.RANDOM_STATE),
        {"alpha": alpha_grid},
        cv=inner_cv, scoring="neg_mean_squared_error", n_jobs=-1,
    )
    lasso = GridSearchCV(
        Lasso(random_state=C.RANDOM_STATE, max_iter=50000),
        {"alpha": alpha_grid},
        cv=inner_cv, scoring="neg_mean_squared_error", n_jobs=-1,
    )

    models = {
        "OLS": Pipeline([("scale", scaler), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scale", scaler), ("model", ridge)]),
        "Lasso": Pipeline([("scale", scaler), ("model", lasso)]),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=C.RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=C.RANDOM_STATE
        ),
    }
    return models


# ---------------------------------------------------------------- 指标
def metrics(y_true: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """统一指标：全部在 log 尺度计算，RMSLE 额外给美元尺度的 RMSE。

    * RMSE_log —— 主指标，与 Part2 / Part3 一致
    * MAE_log  —— 绝对误差，对极端点更稳健
    * R2       —— 解释力
    * RMSLE    —— 课程要求的指标，等价于美元尺度的相对误差
    * RMSE_USD —— 反变换回美元后的 RMSE，方便直观理解
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    resid = y_true - y_pred_log
    pred_usd = np.exp(y_pred_log)
    true_usd = np.exp(y_true)

    return {
        "MSE_log": mean_squared_error(y_true, y_pred_log),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_true, y_pred_log))),
        "MAE_log": mean_absolute_error(y_true, y_pred_log),
        "R2": r2_score(y_true, y_pred_log),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(true_usd) - np.log1p(pred_usd)) ** 2))),
        "RMSE_USD": float(np.sqrt(mean_squared_error(true_usd, pred_usd))),
    }


# ---------------------------------------------------------------- 主流程
def run() -> pd.DataFrame:
    C.apply_style()
    ds = C.load_dataset()
    print(ds.summary())

    X, y = ds.X, ds.y
    X_tr, X_te, y_tr, y_te = C.split_80_20(ds)
    print(f"80/20 split -> train {X_tr.shape}, test {X_te.shape}")

    models = build_models(X, y)

    # =============== 1. 80/20 划分 ===============
    rows_80 = []
    preds_80 = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        preds_80[name] = pred
        row = {"model": name}
        row.update(metrics(y_te, pred))
        if hasattr(model, "named_steps"):
            est = model.named_steps["model"]
            if hasattr(est, "best_params_"):
                row["alpha"] = est.best_params_["alpha"]
        rows_80.append(row)
        print(f"  {name:20s} RMSE_log={row['RMSE_log']:.5f}  R2={row['R2']:.4f}")

    df_80 = pd.DataFrame(rows_80).sort_values("RMSE_log").reset_index(drop=True)
    df_80.to_csv(os.path.join(C.OUT_DIR, "unified_80_20.csv"), index=False)

    # =============== 2. 10 折交叉验证 ===============
    from sklearn.model_selection import cross_validate

    kf = C.kfold_splits()
    rows_cv = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=kf, n_jobs=-1, error_score="raise",
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
        )
        rows_cv.append({
            "model": name,
            "RMSE_log": -scores["test_rmse"].mean(),
            "RMSE_log_std": scores["test_rmse"].std(),
            "MAE_log": -scores["test_mae"].mean(),
            "MAE_log_std": scores["test_mae"].std(),
            "R2": scores["test_r2"].mean(),
            "R2_std": scores["test_r2"].std(),
        })
        print(f"  [CV10] {name:20s} RMSE={rows_cv[-1]['RMSE_log']:.5f} "
              f"(±{rows_cv[-1]['RMSE_log_std']:.5f})  R2={rows_cv[-1]['R2']:.4f}")

    df_cv = pd.DataFrame(rows_cv).sort_values("RMSE_log").reset_index(drop=True)
    df_cv.to_csv(os.path.join(C.OUT_DIR, "unified_cv10.csv"), index=False)

    # =============== 3. 合并总表 ===============
    merged = df_80.merge(df_cv, on="model", how="outer", suffixes=("_80_20", "_cv10"))
    merged["rank_80_20"] = merged["RMSE_log_80_20"].rank(method="min")
    merged["rank_cv10"] = merged["RMSE_log_cv10"].rank(method="min")
    merged = merged.sort_values("RMSE_log_cv10").reset_index(drop=True)
    merged.to_csv(os.path.join(C.OUT_DIR, "unified_benchmark.csv"), index=False)

    # 保存各模型 80/20 的预测，供残差/诊断图复用
    np.save(os.path.join(C.OUT_DIR, "cv_folds.npy"), np.array([]))
    pd.DataFrame({**{f"{k}_pred": v for k, v in preds_80.items()},
                  "y_true": y_te}).to_csv(
        os.path.join(C.OUT_DIR, "unified_80_20_predictions.csv"), index=False)
    ds.train[["Id"]].assign(**{
        "split": np.where(ds.train["Id"].isin(X_te.index), "test", "train")
    }).to_csv(os.path.join(C.OUT_DIR, "_split_marker.csv"), index=False)

    print("\n=== 统一口径对比 ===")
    print(merged[["model", "RMSE_log_80_20", "R2_80_20",
                   "RMSE_log_cv10", "RMSE_log_std", "R2_cv10"]].to_string(index=False))

    _plot(df_80, df_cv, preds_80, y_te)
    return merged


# ---------------------------------------------------------------- 作图
def _plot(df_80, df_cv, preds_80, y_te) -> None:
    import matplotlib.pyplot as plt

    C.apply_style()
    order = df_cv["model"].tolist()          # 按 CV 表现排序
    colors = [C.MODEL_COLORS.get(m, C.COLORS["grey"]) for m in order]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # --- (a) 10 折 CV RMSE ---
    ax = axes[0]
    sub = df_cv.set_index("model").loc[order]
    ypos = np.arange(len(order))
    ax.barh(ypos, sub["RMSE_log"], xerr=sub["RMSE_log_std"], color=colors,
            height=0.6, capsize=4, error_kw={"ecolor": "#444444", "lw": 1})
    ax.set_yticks(ypos, order)
    ax.invert_yaxis()
    ax.set_xlabel("10-fold CV RMSE (log scale)")
    ax.set_title("(a) Cross-validated generalisation error", loc="left")
    for i, v in enumerate(sub["RMSE_log"]):
        ax.text(v + sub["RMSE_log_std"].iloc[i] + 0.004, i, f"{v:.4f}",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, sub["RMSE_log"].max() * 1.28)

    # --- (b) 80/20 vs CV 对照 ---
    ax = axes[1]
    sub80 = df_80.set_index("model").loc[order]
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, sub80["RMSE_log"], w, label="80/20 split", color=colors, alpha=0.95)
    ax.bar(x + w / 2, sub["RMSE_log"], w, label="10-fold CV", color=colors,
           alpha=0.45, edgecolor="#333333", hatch="//")
    ax.set_xticks(x, order, rotation=20, ha="right")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("(b) Same data, two protocols", loc="left")
    ax.legend()
    ax.set_yscale("log")
    ax.axhline(0.06, ls="--", c="grey", lw=1)
    ax.text(len(order) - 0.5, 0.062, "OLS CV collapse", fontsize=8,
            ha="right", color="#555555")

    # --- (c) 预测 vs 真实 ---
    ax = axes[2]
    for m in order:
        ax.scatter(np.exp(y_te), np.exp(preds_80[m]), s=14, alpha=0.55,
                   color=C.MODEL_COLORS.get(m, C.COLORS["grey"]), label=m)
    lim = [np.exp(y_te).min() * 0.9, np.exp(y_te).max() * 1.05]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual sale price (USD)")
    ax.set_ylabel("Predicted sale price (USD)")
    ax.set_title("(c) 80/20 predictions vs truth", loc="left")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Unified benchmark on one protocol: Part 1 feature matrix, 80/20 split + 10-fold CV",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(C.FIG_DIR, "fig01_unified_benchmark.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    run()
