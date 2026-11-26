# -*- coding: utf-8 -*-
"""
Part 4 · 实验 G：跨成员结果的整合可视化（Integration Figures）
===================================================================
分工表里 Part4 的任务写的是"整合可视化"。这一节把四位成员的结果
放到同一张图上，让读者一眼看到 **方法之间的一致与分歧**。

四张整合图
----------
``fig13`` **四成员结果对照**
    Part2（原始特征 + 官方 test）、Part3（工程特征 + 内部划分）、Part4（统一口径）
    三组结果的横向对照，说明为什么需要统一口径，以及统一口径后结论如何收敛。

``fig14`` **特征重要性四法对照**
    Part1 相关系数、Part2 标准化 OLS 系数、Part3 GBR impurity 重要性、
    Part4 置换重要性 —— 同一批特征、四种度量，放在一起看它们的分歧与共识。

``fig15`` **误差结构分解**
    按房价分位把测试集误差拆开，看模型在哪些价位段表现最差（豪宅段 vs 入门段），
    这是"能预测准吗"之外的第二个问题："在哪类房子上不准"。

``fig16`` **学习曲线**
    训练样本量与泛化误差的关系，回答"数据是不是不够"。

作者：Yifei Lv（Part4）
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C


# ================================================================ fig13
def fig_member_comparison():
    """四成员结果对照表 + 图。"""
    import matplotlib.pyplot as plt

    C.apply_style()

    rows = [
        # 成员, 数据基础, 特征数, 评估口径, 评估集, OLS, Ridge, Lasso, 结论
        ("Part 2\n(Member 2)", "原始 clean 数据\n(patsy one-hot)", "223",
         "官方 test 集", "1,459", "MSE 0.0402", "MSE 0.0185", "MSE 0.0179",
         "正则化优于 OLS"),
        ("Part 3\n(Member 3)", "工程特征 + 多项式/样条", "336",
         "内部 80/20", "292", "MSE 0.0064", "MSE 0.0040", "MSE 0.0043",
         "OLS 在 10 折下崩溃"),
        ("Part 4\n(统一口径)", "Part 1 工程特征矩阵", "250",
         "80/20 + 10 折 CV", "292 + 10 折", "RMSE 0.1278", "RMSE 0.1092", "RMSE 0.1069",
         "Lasso 最优且最稳"),
    ]
    df = pd.DataFrame(rows, columns=[
        "成员", "数据基础", "特征数", "评估口径", "评估集",
        "OLS", "Ridge", "Lasso", "结论"])

    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 3:
            cell.set_facecolor("#EAF2F8")
            cell.get_text().set_fontweight("bold")
    ax.set_title("Three attempts at the same question — and why one protocol settles it",
                 loc="left", fontsize=13, fontweight="bold", pad=18)
    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "fig13_member_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)


# ================================================================ fig14
def fig_importance_four_ways():
    """四种重要性度量并排对比。"""
    import matplotlib.pyplot as plt

    C.apply_style()
    ds = C.load_dataset()
    X, y = ds.X, ds.y

    # (1) Part1 风格：与 log 房价的相关系数
    corr = pd.Series(np.abs(np.corrcoef(X.to_numpy(dtype=float).T, y)[-1, :-1]),
                     index=X.columns).sort_values(ascending=False)

    # (2) Part2 风格：标准化 OLS 系数
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    ols = Pipeline([("s", StandardScaler()), ("m", LinearRegression())]).fit(X, y)
    ols_coef = pd.Series(ols.named_steps["m"].coef_, index=X.columns).abs()

    # (3) Part3 风格：GBR impurity 重要性
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                    max_depth=3, random_state=C.RANDOM_STATE).fit(X, y)
    gbr_imp = pd.Series(gbr.feature_importances_, index=X.columns)

    # (4) Part4：置换重要性（GBR，OOB）
    perm_path = os.path.join(C.OUT_DIR, "permutation_importance.csv")
    perm = (pd.read_csv(perm_path)
            .query("model == 'Gradient Boosting'")
            .set_index("feature")["perm_importance_mean"])

    top = corr.head(12).index.tolist()
    series = {
        "Part 1 · correlation": corr,
        "Part 2 · std. OLS coefficient": ols_coef,
        "Part 3 · GBR impurity importance": gbr_imp,
        "Part 4 · permutation importance": perm,
    }
    order = ["Part 1 · correlation", "Part 2 · std. OLS coefficient",
             "Part 3 · GBR impurity importance", "Part 4 · permutation importance"]
    colors = [C.COLORS["grey"], C.COLORS["primary"], C.COLORS["purple"], C.COLORS["green"]]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6.4))
    for ax, lab, color in zip(axes, order, colors):
        s = series[lab].reindex(top).fillna(0.0)
        s = s.sort_values()
        ax.barh(s.index, s.values, color=color)
        ax.set_title(lab, loc="left", fontsize=10.5)
        ax.set_xlabel("importance (each normalised to its own scale)")
        ax.tick_params(labelsize=8.5)
    fig.suptitle("Four ways of ranking the same twelve features — the top block agrees, "
                 "the tail does not", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(C.FIG_DIR, "fig14_importance_four_ways.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)


# ================================================================ fig15
def fig_error_decomposition():
    """按房价分位拆解误差，看模型在哪类房子上最弱。"""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold

    C.apply_style()
    ds = C.load_dataset()
    X, y = ds.X, ds.y

    inner = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    lasso = Pipeline([("s", StandardScaler()),
                      ("m", LassoCV(alphas=np.logspace(-3, 3, 100), cv=inner,
                                    max_iter=50000, random_state=C.RANDOM_STATE))])
    # 10 折 OOF 预测，避免用训练集拟合值自欺
    oof = cross_val_predict(lasso, X, y, cv=C.kfold_splits(), n_jobs=-1)
    oof = np.asarray(oof).ravel()
    resid = np.asarray(y, dtype=float) - oof
    price = ds.train[C.PRICE_COL].to_numpy(dtype=float)

    df = pd.DataFrame({"price": price, "resid": resid, "abs_resid": np.abs(resid)})
    df["bin"] = pd.qcut(df["price"], 5, labels=[
        "Q1\ncheapest", "Q2", "Q3", "Q4", "Q5\npriciest"])
    grouped = df.groupby("bin", observed=True).agg(
        median_price=("price", "median"),
        mean_abs_resid=("abs_resid", "mean"),
        count=("resid", "size")).reset_index()

    # 美元尺度的绝对误差
    grouped["MAE_USD"] = grouped["mean_abs_resid"] * grouped["median_price"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

    ax = axes[0]
    bars = ax.bar(grouped["bin"], grouped["mean_abs_resid"],
                  color=[C.COLORS["grey"], "#7F9BB5", C.COLORS["primary"],
                         C.COLORS["accent"], C.COLORS["red"]])
    for b, v in zip(bars, grouped["mean_abs_resid"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Mean |residual|  (log price units)")
    ax.set_xlabel("Sale-price quintile")
    ax.set_title("(a) Log-scale error is roughly flat across price tiers", loc="left")

    ax = axes[1]
    bars = ax.bar(grouped["bin"], grouped["MAE_USD"] / 1000,
                  color=[C.COLORS["grey"], "#7F9BB5", C.COLORS["primary"],
                         C.COLORS["accent"], C.COLORS["red"]])
    for b, v in zip(bars, grouped["MAE_USD"] / 1000):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"${v:.1f}k",
                ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Approximate absolute error (USD, thousands)")
    ax.set_xlabel("Sale-price quintile")
    ax.set_title("(b) In dollars the error grows with the price", loc="left")

    fig.suptitle("Where the model fails: the same relative error costs far more on expensive houses",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(C.FIG_DIR, "fig15_error_decomposition.png")
    fig.savefig(out)
    plt.close(fig)

    grouped.to_csv(os.path.join(C.OUT_DIR, "error_by_price_quintile.csv"), index=False)
    print("saved:", out)
    return grouped


# ================================================================ fig16
def fig_learning_curve():
    """学习曲线：训练样本量 vs 泛化误差。"""
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    C.apply_style()
    ds = C.load_dataset()
    X, y = ds.X, ds.y

    inner = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    rng = np.random.default_rng(C.RANDOM_STATE)
    X_all = X.to_numpy(dtype=float)
    y_all = np.asarray(y, dtype=float)

    # 固定保留 20% 作为统一的留出评估集（任何比例下都不参与训练）
    n_total = len(X_all)
    perm = rng.permutation(n_total)
    n_hold = int(n_total * 0.2)
    hold_idx = perm[:n_hold]
    pool_idx = perm[n_hold:]
    X_hold, y_hold = X_all[hold_idx], y_all[hold_idx]

    for frac in fractions:
        errs = []
        n = max(10, int(len(pool_idx) * frac))
        for rep in range(5 if frac < 1.0 else 3):
            sel = rng.choice(pool_idx, size=min(n, len(pool_idx)), replace=False)
            m = Pipeline([("s", StandardScaler()),
                          ("m", LassoCV(alphas=np.logspace(-3, 3, 100), cv=inner,
                                        max_iter=50000, random_state=C.RANDOM_STATE))])
            m.fit(X_all[sel], y_all[sel])
            errs.append(np.sqrt(np.mean((y_hold - m.predict(X_hold)) ** 2)))
        rows.append({"n_train": int(n), "frac": frac,
                     "RMSE_log": float(np.mean(errs)), "std": float(np.std(errs))})
        print(f"  frac={frac:.1f} n={n} holdout RMSE={rows[-1]['RMSE_log']:.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "learning_curve.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.errorbar(df["n_train"], df["RMSE_log"], yerr=df["std"], marker="o",
                ms=7, lw=2.2, color=C.COLORS["green"], capsize=4,
                ecolor="#666666")
    for r in df.itertuples():
        ax.annotate(f"{r.RMSE_log:.4f}", (r.n_train, r.RMSE_log),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Held-out RMSE (log scale)")
    ax.set_title("Learning curve for the final Lasso model — more data still helps",
                 loc="left")
    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "fig16_learning_curve.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)


# ================================================================ 主流程
def run():
    C.apply_style()
    fig_member_comparison()
    fig_importance_four_ways()
    fig_error_decomposition()
    fig_learning_curve()
    print("done.")


if __name__ == "__main__":
    run()
