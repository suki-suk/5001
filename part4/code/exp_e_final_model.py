# -*- coding: utf-8 -*-
"""
Part 4 · 实验 E：最终模型与测试集预测（Final Model & Submission）
===================================================================
流程
----
1. 在全部训练数据（1,458 条）上训练最终模型；
2. 在 Kaggle 官方 test 集（1,459 条）上预测 SalePrice；
3. 按 ``sample_submission.csv`` 的格式输出 ``submission.csv``。

模型选择依据
------------
Part4 实验 C 的统一口径对比（同一份特征矩阵、同一 80/20 划分、同一 10 折 CV）
给出的排序是 **Lasso > Ridge > GBR > OLS > Random Forest**。
Lasso 与 Ridge 的差距很小（CV RMSE 0.1069 vs 0.1092，约 2%），但 Lasso：

* 10 折 CV 标准差更小（0.0129 vs 0.0136），表现更稳定；
* 自动把不重要的系数压到 0，从 250 个特征里保留不到 70 个，
  可解释性远好于 Ridge（课程 RQ3 明确要求 Lasso 完成变量选择）；
* 岭回归一个系数都不会真正归零，无法做变量选择。

因此 **最终模型采用 Lasso**，同时提交 Ridge 作为备选模型的结果文件。

输出
----
* ``output/submission_lasso.csv``     —— 主交付（推荐）
* ``output/submission_ridge.csv``     —— 备选
* ``output/submission_gbr.csv``       —— 备选（树集成）
* ``output/final_model_coefficients.csv``
* ``output/test_prediction_summary.csv``
* ``figures/fig09_test_predictions.png``
* ``figures/fig10_calibration.png``

作者：Yifei Lv（Part4）
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C


def build_models(ds):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = ds.X, ds.y
    inner = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    grid = np.logspace(-3, 3, 100)

    lasso = Pipeline([("scale", StandardScaler()),
                      ("model", LassoCV(alphas=grid, cv=inner, max_iter=50000,
                                        random_state=C.RANDOM_STATE))])
    ridge = Pipeline([("scale", StandardScaler()),
                      ("model", RidgeCV(alphas=grid, cv=inner))])
    gbr = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=C.RANDOM_STATE)
    return {"Lasso": lasso, "Ridge": ridge, "Gradient Boosting": gbr}


def run():
    C.apply_style()
    ds = C.load_dataset()
    print(ds.summary())

    models = build_models(ds)
    X_test = ds.test[ds.feature_cols]

    summary_rows = []
    for name, model in models.items():
        model.fit(ds.X, ds.y)
        pred_log = model.predict(X_test)
        pred_usd = np.exp(pred_log)

        # ---------------- 写 submission ----------------
        sub = pd.DataFrame({
            "Id": ds.test["Id"].astype(int).to_numpy(),
            "SalePrice": np.round(pred_usd, 0).astype(int),
        }).sort_values("Id").reset_index(drop=True)
        fname = {"Lasso": "submission_lasso.csv",
                 "Ridge": "submission_ridge.csv",
                 "Gradient Boosting": "submission_gbr.csv"}[name]
        sub.to_csv(os.path.join(C.OUT_DIR, fname), index=False)

        # ---------------- 预测分布摘要 ----------------
        summary_rows.append({
            "model": name,
            "n_test": len(sub),
            "min": pred_usd.min(),
            "p05": np.percentile(pred_usd, 5),
            "median": np.median(pred_usd),
            "mean": pred_usd.mean(),
            "p95": np.percentile(pred_usd, 95),
            "max": pred_usd.max(),
            "std": pred_usd.std(),
            "skew": float(pd.Series(pred_usd).skew()),
        })

        # ---------------- 最终系数 ----------------
        if hasattr(model, "named_steps"):
            coef = model.named_steps["model"].coef_
            pd.DataFrame({
                "model": name,
                "feature": ds.feature_cols,
                "coef": coef,
                "abs_coef": np.abs(coef),
            }).sort_values("abs_coef", ascending=False).to_csv(
                os.path.join(C.OUT_DIR, f"final_coefficients_{name.split()[0].lower()}.csv"),
                index=False)
            print(f"  {name}: alpha = {getattr(model.named_steps['model'], 'alpha_', np.nan):.4f}"
                  f", non-zero coefs = {int((coef != 0).sum())}/{len(coef)}")

        print(f"  {name}: saved {fname}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(C.OUT_DIR, "test_prediction_summary.csv"), index=False)
    print("\n=== 测试集预测分布摘要（美元）===")
    print(summary.to_string(index=False))

    _plot(ds, models, summary)
    return summary


def _plot(ds, models, summary):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error

    C.apply_style()
    train_price = ds.train[C.PRICE_COL].to_numpy(dtype=float)
    train_log = ds.train[C.TARGET].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

    # (a) 预测分布 vs 训练集真实分布
    ax = axes[0]
    ax.hist(train_price, bins=60, color=C.COLORS["light"], edgecolor="#BBBBBB",
            label="Training actual (1,458)", density=True)
    for name, color in (("Lasso", C.COLORS["green"]), ("Ridge", C.COLORS["accent"]),
                        ("Gradient Boosting", C.COLORS["red"])):
        model = models[name]
        pred = np.exp(model.predict(ds.test[ds.feature_cols]))
        ax.hist(pred, bins=60, histtype="step", lw=2.0, color=color,
                density=True, label=f"{name} prediction (1,459)")
    ax.set_xlabel("Sale price (USD)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Predicted test prices track the training distribution", loc="left")
    ax.legend(fontsize=8)

    # (b) 训练集拟合优度（log 尺度）
    ax = axes[1]
    ax.scatter(train_log, train_log, s=10, color=C.COLORS["light"],
               edgecolor="none", label="Training data (log scale)")
    for name, color in (("Lasso", C.COLORS["green"]), ("Ridge", C.COLORS["accent"]),
                        ("Gradient Boosting", C.COLORS["red"])):
        model = models[name]
        pred = model.predict(ds.X)
        ax.scatter(train_log, pred, s=8, alpha=0.30, color=color,
                   label=f"{name}  (train RMSE={np.sqrt(mean_squared_error(train_log, pred)):.4f})")
    lim = [train_log.min() - 0.1, train_log.max() + 0.1]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.65)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual log(SalePrice)")
    ax.set_ylabel("Predicted log(SalePrice)")
    ax.set_title("(b) Final model fitted on all training data", loc="left")
    ax.legend(fontsize=8)

    fig.suptitle("Final model and test-set predictions", fontsize=13,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(C.FIG_DIR, "fig09_test_predictions.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    run()
