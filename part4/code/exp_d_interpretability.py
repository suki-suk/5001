# -*- coding: utf-8 -*-
"""
Part 4 · 实验 D：特征重要性与可解释性分析（Interpretability）
==================================================================
这是 Part4 的核心交付，对应分工表里"特征重要性与可解释性分析 + 整合可视化"。

问题
----
四个成员给出了四组"重要性"，但彼此口径不同：

* Part2：标准化 OLS 系数（受共线性污染，屋顶材料等虚拟变量系数虚高）
* Part3：GBR 的 impurity-based 重要性（偏向高基数 / 连续变量）
* Part1：与房价的相关系数（只看单变量，不控制其他变量）

本模块给出 **一套统一的、与 Part1 的 252 列特征矩阵对齐的重要性体系**：

1. **Permutation Importance（置换重要性）**
   打乱某个特征、观察误差上升多少。它是唯一不依赖模型内部尺度的度量，
   可以直接跨 OLS / Ridge / Lasso / GBR 比较。在 10 折 OOB 验证集上计算，
   衡量的是"泛化贡献"而非"拟合贡献"。

2. **标准化线性系数（Ridge / Lasso）**
   特征全部标准化后，系数就是"该特征变动 1 个标准差带来的 log(价格) 变化"，
   跨特征可直接比较。Lasso 的稀疏结构另外输出被选中的变量清单。

3. **SHAP 值（Linear SHAP / TreeSHAP）**
   对单条预测给出"每个特征贡献了多少"的加性分解。输出全局重要性、
   单条样本瀑布分解、特征间交互强度。

4. **部分依赖图（PDP）与 ICE 曲线**
   回答"当居住面积从 1,000 涨到 3,000 平方英尺时，模型预测的房价怎么变"。

5. **H-statistic 交互强度**
   量化特征两两交互的强度，验证"质量 × 面积"这一交互是否真的重要。

产出
----
* ``output/permutation_importance.csv``
* ``output/linear_coefficients.csv`` / ``lasso_selected_features.csv``
* ``output/shap_global_importance.csv``
* ``output/shap_waterfall_example.csv``
* ``output/interaction_strength.csv``
* ``output/partial_dependence.csv`` / ``ice_curves.csv``
* ``figures/fig02…fig08``

作者：Yifei Lv（Part4）
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C


# ---------------------------------------------------------------- 工具
def _fit_final_models(ds):
    """在全部训练数据上拟合 Ridge / Lasso / GBR（各自用内部 CV 选超参），供解释分析使用。"""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = ds.X, ds.y
    inner = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    grid = np.logspace(-3, 3, 100)

    ridge = Pipeline([("scale", StandardScaler()),
                      ("model", RidgeCV(alphas=grid, cv=inner))])
    lasso = Pipeline([("scale", StandardScaler()),
                      ("model", LassoCV(alphas=grid, cv=inner, max_iter=50000,
                                        random_state=C.RANDOM_STATE))])
    gbr = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=C.RANDOM_STATE)

    for m in (ridge, lasso, gbr):
        m.fit(X, y)
    return ridge, lasso, gbr


# ================================================================ 1. 置换重要性
def permutation_importance(ds, models: dict, n_repeats: int = 10) -> pd.DataFrame:
    """在 10 折 OOB 验证集上计算置换重要性（衡量泛化贡献）。

    对每个模型的每一折：在训练折上拟合 → 在验证折上计算基线误差 →
    逐个打乱特征重算误差 → 差值即为该特征的置换重要性。
    """
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error

    X, y = ds.X, ds.y
    kf = C.kfold_splits()
    rows = []

    for name, model in models.items():
        fold_imps = []
        for tr_idx, te_idx in kf.split(X):
            m = clone(model)
            m.fit(X.iloc[tr_idx], y[tr_idx])

            Xte = X.iloc[te_idx]
            yte = y[te_idx]
            base_mse = mean_squared_error(yte, m.predict(Xte))

            rng = np.random.default_rng(C.RANDOM_STATE)
            per_feature = np.zeros(len(Xte.columns))
            for j in range(len(Xte.columns)):
                deltas = []
                for _ in range(n_repeats):
                    Xp = Xte.copy()
                    Xp.iloc[:, j] = rng.permutation(Xp.iloc[:, j].to_numpy())
                    mse = mean_squared_error(yte, m.predict(Xp))
                    deltas.append(mse - base_mse)
                per_feature[j] = float(np.mean(deltas))
            fold_imps.append(per_feature)

        imp = np.mean(fold_imps, axis=0)
        rows.append(pd.DataFrame({
            "model": name,
            "feature": X.columns,
            "perm_importance_mean": imp,
            "perm_importance_std": np.std(fold_imps, axis=0),
        }))
        print(f"  permutation importance done: {name}")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(C.OUT_DIR, "permutation_importance.csv"), index=False)
    return out


# ================================================================ 2. 线性系数
def linear_coefficients(ds, ridge, lasso) -> tuple:
    """标准化 Ridge / Lasso 系数（跨特征可直接比较）。"""
    X = ds.X
    rows = []
    for name, pipe in (("Ridge", ridge), ("Lasso", lasso)):
        coef = pipe.named_steps["model"].coef_
        rows.append(pd.DataFrame({
            "model": name,
            "feature": X.columns,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }))
    allc = pd.concat(rows, ignore_index=True)
    allc.to_csv(os.path.join(C.OUT_DIR, "linear_coefficients.csv"), index=False)

    lasso_sel = (allc[(allc["model"] == "Lasso") & (allc["coef"] != 0)]
                 .sort_values("abs_coef", ascending=False)
                 .reset_index(drop=True))
    lasso_sel.to_csv(os.path.join(C.OUT_DIR, "lasso_selected_features.csv"), index=False)
    return allc, lasso_sel


# ================================================================ 3. SHAP
def shap_analysis(ds, ridge, gbr, sample_size: int = 600):
    """SHAP 全局重要性 + 单样本瀑布分解 + 交互强度。"""
    import shap

    X = ds.X
    sample = X.sample(n=min(sample_size, len(X)), random_state=C.RANDOM_STATE)

    # --- 线性模型用 Linear SHAP（快速且精确）---
    lin_explainer = shap.LinearExplainer(
        ridge.named_steps["model"], ridge.named_steps["scale"].transform(X))
    lin_sv = lin_explainer.shap_values(sample)

    # --- 树模型用 TreeSHAP ---
    tree_explainer = shap.TreeExplainer(gbr)
    tree_sv = tree_explainer.shap_values(sample)

    rows = []
    for m, sv in (("Ridge (Linear SHAP)", lin_sv), ("GBR (TreeSHAP)", tree_sv)):
        rows.append(pd.DataFrame({
            "model": m, "feature": X.columns,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }))
    shap_imp = pd.concat(rows, ignore_index=True)
    shap_imp.to_csv(os.path.join(C.OUT_DIR, "shap_global_importance.csv"), index=False)

    # --- 单条样本瀑布分解 ---
    ex = tree_explainer(sample)
    waterfall = pd.DataFrame({
        "feature": X.columns,
        "shap_value": ex[0].values,
        "feature_value": sample.iloc[0].values,
    }).sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True)
    waterfall.to_csv(os.path.join(C.OUT_DIR, "shap_waterfall_example.csv"), index=False)

    return shap_imp, waterfall, tree_explainer, sample


def interaction_strength(model, sample, top_k: int = 12) -> pd.DataFrame:
    """H-statistic：衡量特征对之间的交互强度。

    H = 0 表示纯加性贡献；H > 0 表示该特征对的效应依赖于对方取值。
    完整的两两组合在 250 维上是 3 万次预测，代价过高；
    这里先按置换重要性挑出 top_k 个关键特征，只在它们之间两两计算——
    这也符合"重要特征之间才可能有值得关注的交互"的分析逻辑。
    """
    from itertools import combinations

    X = sample
    base = model.predict(X)
    rng = np.random.default_rng(C.RANDOM_STATE)

    # 先用一次置换重要性挑出候选特征
    from sklearn.base import clone
    m = clone(model)
    m.fit(X, X.index.map(lambda i: 0) * 0 + sample["OverallQual"].iloc[0] * 0) \
        if False else m  # 占位，真正的候选来自外部传入的 importance

    candidates = top_k if isinstance(top_k, list) else []
    cols = [c for c in candidates if c in X.columns]

    rows = []
    for a, b in combinations(cols, 2):
        Xa = X.copy(); Xb = X.copy()
        Xa[a] = rng.permutation(Xa[a].to_numpy())
        Xb[b] = rng.permutation(Xb[b].to_numpy())
        pred_a = model.predict(Xa)
        pred_b = model.predict(Xb)
        Xboth = X.copy()
        Xboth[a] = Xa[a].to_numpy()
        Xboth[b] = Xb[b].to_numpy()
        pred_both = model.predict(Xboth)
        add = base + (pred_a - base) + (pred_b - base)
        h = float(np.mean((pred_both - add) ** 2))
        rows.append({"feature_a": a, "feature_b": b, "h_stat": h})

    out = pd.DataFrame(rows).sort_values("h_stat", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(C.OUT_DIR, "interaction_strength.csv"), index=False)
    return out


# ================================================================ 4. PDP / ICE
def partial_dependence(model, X: pd.DataFrame, features, n_ice: int = 50) -> tuple:
    """部分依赖图（PDP）与个体条件期望（ICE）。

    PDP：对网格上的每个取值，把该特征固定在该值，其余特征按数据集边缘分布采样，
          取平均预测。ICE：对单条样本，只改变该特征，看预测怎么动。
    两者一起看能区分"整体趋势"与"个体差异"。
    """
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows, ice_rows = [], []

    for feat in features:
        vals = X[feat].to_numpy()
        grid = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 30)

        # ---- PDP：手工实现，避免 recursion 在高维下的巨大开销 ----
        n_boot = 30
        acc = np.zeros(len(grid))
        for g_i, g in enumerate(grid):
            preds = []
            for b in range(n_boot):
                Xb = X.sample(n=min(400, len(X)), random_state=int(rng.integers(1e9)))
                Xb = Xb.copy()
                Xb[feat] = g
                preds.append(model.predict(Xb))
            acc[g_i] = float(np.mean(preds))
        for g, a in zip(grid, acc):
            rows.append({"feature": feat, "value": float(g), "pd": float(a)})

        # ---- ICE：固定样本，只变这一个特征 ----
        idx = rng.choice(len(X), size=min(n_ice, len(X)), replace=False)
        for si, i in enumerate(idx):
            base_row = X.iloc[[i]]
            for g in grid:
                Xb = base_row.copy()
                Xb[feat] = g
                ice_rows.append({
                    "feature": feat, "sample": int(si),
                    "value": float(g),
                    "ice": float(model.predict(Xb)[0]),
                })

    pd_df = pd.DataFrame(rows)
    ice_df = pd.DataFrame(ice_rows)
    pd_df.to_csv(os.path.join(C.OUT_DIR, "partial_dependence.csv"), index=False)
    ice_df.to_csv(os.path.join(C.OUT_DIR, "ice_curves.csv"), index=False)
    return pd_df, ice_df


# ================================================================ 5. 作图
def plot_all(perm_imp, allcoef, lasso_sel, shap_imp, waterfall,
             inter, pd_df, ice_df):
    import matplotlib.pyplot as plt

    C.apply_style()
    fig_dir = C.FIG_DIR

    # ---------- fig02: 置换重要性对比 ----------
    models_list = perm_imp["model"].unique().tolist()
    top_n = 18
    chosen = (perm_imp[perm_imp["model"] == models_list[0]]
              .nlargest(top_n, "perm_importance_mean")["feature"].tolist())
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    y = np.arange(len(chosen))
    h = 0.8 / len(models_list)
    pal = [C.COLORS["primary"], C.COLORS["accent"], C.COLORS["green"], C.COLORS["purple"]]
    for i, m in enumerate(models_list):
        sub = (perm_imp[perm_imp["model"] == m]
               .set_index("feature").reindex(chosen)
               .fillna(0.0))
        ax.barh(y + i * h - 0.4 + h / 2, sub["perm_importance_mean"], h,
                label=m, color=pal[i % len(pal)])
    ax.set_yticks(y, chosen)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance  (increase in OOB MSE when the feature is shuffled)")
    ax.set_title("Top features by permutation importance — stable across model families",
                 loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig02_permutation_importance.png"))
    plt.close(fig)

    # ---------- fig03: Ridge vs Lasso 标准化系数 ----------
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    for ax, mname in zip(axes, ["Ridge", "Lasso"]):
        sub = allcoef[allcoef["model"] == mname].nlargest(15, "abs_coef")
        sub = sub.sort_values("coef")
        colors = [C.COLORS["green"] if v > 0 else C.COLORS["red"] for v in sub["coef"]]
        ax.barh(sub["feature"], sub["coef"], color=colors)
        ax.axvline(0, color="#333333", lw=0.9)
        ax.set_title(f"{mname}: standardised coefficients (Top 15)", loc="left")
        ax.set_xlabel("coefficient on log(SalePrice)")
    fig.suptitle("Linear effect direction and size — standardised so features are comparable",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(fig_dir, "fig03_linear_coefficients.png"))
    plt.close(fig)

    # ---------- fig04: Lasso 稀疏结构 ----------
    n_total = int((allcoef["model"] == "Lasso").sum())
    n_sel = len(lasso_sel)
    fig, ax = plt.subplots(figsize=(11, 6))
    top = lasso_sel.head(20).iloc[::-1]
    ax.barh(top["feature"], top["coef"], color=C.COLORS["green"])
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_xlabel("Lasso coefficient")
    ax.set_title(f"Lasso variable selection: {n_sel} of {n_total} features kept "
                 f"({100*(1-n_sel/n_total):.1f}% sparsity)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig04_lasso_selection.png"))
    plt.close(fig)

    # ---------- fig05: SHAP 全局重要性 ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    for ax, m in zip(axes, models_list):
        sub = shap_imp[shap_imp["model"] == m].nlargest(15, "mean_abs_shap").sort_values("mean_abs_shap")
        ax.barh(sub["feature"], sub["mean_abs_shap"],
                color=C.COLORS["primary"] if "Ridge" in m else C.COLORS["red"])
        ax.set_xlabel("mean |SHAP value|  (log price units)")
        ax.set_title(m, loc="left")
    fig.suptitle("SHAP global importance — how much each feature shifts the log price",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(fig_dir, "fig05_shap_global.png"))
    plt.close(fig)

    # ---------- fig06: 单样本瀑布 ----------
    wf = waterfall.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    colors = [C.COLORS["green"] if v > 0 else C.COLORS["red"] for v in wf["shap_value"]]
    ax.barh(wf["feature"], wf["shap_value"], color=colors)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_xlabel("SHAP contribution to the prediction (log price)")
    ax.set_title("How one house's price is explained, feature by feature", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig06_shap_waterfall.png"))
    plt.close(fig)

    # ---------- fig07: PDP + ICE ----------
    feats = [f for f in ["GrLivArea", "TotalSF", "OverallQual", "YearBuilt"]
             if f in pd_df["feature"].unique()]
    fig, axes = plt.subplots(1, len(feats), figsize=(5.0 * len(feats), 4.9))
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, feats):
        p = pd_df[pd_df["feature"] == feat].sort_values("value")
        ax.plot(p["value"], p["pd"], color=C.COLORS["red"], lw=2.4, label="PDP (average)")
        ic = ice_df[ice_df["feature"] == feat]
        for _, g in ic.groupby("sample"):
            g = g.sort_values("value")
            ax.plot(g["value"], g["ice"], color=C.COLORS["grey"], lw=0.5, alpha=0.3)
        ax.set_xlabel(feat)
        ax.set_ylabel("Predicted log(SalePrice)")
        ax.set_title(f"Partial dependence — {feat}", loc="left")
    axes[0].legend(loc="upper left")
    fig.suptitle("Partial dependence (red) with individual conditional expectation curves (grey)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(fig_dir, "fig07_pdp_ice.png"))
    plt.close(fig)

    # ---------- fig08: 交互强度 ----------
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    top_int = inter.head(12).iloc[::-1]
    labels = [f"{r.feature_a}  ×  {r.feature_b}" for r in top_int.itertuples()]
    ax.barh(labels, top_int["h_stat"], color=C.COLORS["purple"])
    ax.set_xlabel("H-statistic  (0 = purely additive)")
    ax.set_title("Strongest feature interactions found by the boosting model", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig08_interactions.png"))
    plt.close(fig)

    print("figures saved to", fig_dir)


# ================================================================ 主流程
def run():
    C.apply_style()
    ds = C.load_dataset()
    print(ds.summary())

    ridge, lasso, gbr = _fit_final_models(ds)
    models = {"Ridge": ridge, "Lasso": lasso, "Gradient Boosting": gbr}

    print("[1/5] permutation importance ...")
    perm_imp = permutation_importance(ds, models)

    print("[2/5] linear coefficients ...")
    allcoef, lasso_sel = linear_coefficients(ds, ridge, lasso)
    print(f"      Lasso kept {len(lasso_sel)} features")

    print("[3/5] SHAP analysis ...")
    shap_imp, waterfall, tree_explainer, sample = shap_analysis(ds, ridge, gbr)

    print("[4/5] partial dependence ...")
    pd_df, ice_df = partial_dependence(gbr, ds.X, C.CORE_CONTINUOUS)

    print("[5/5] interaction strength ...")
    # 候选特征 = 置换重要性 top12（限定计算规模）
    cand = (perm_imp[perm_imp["model"] == "Gradient Boosting"]
            .nlargest(12, "perm_importance_mean")["feature"].tolist())
    inter = interaction_strength(gbr, sample, cand)

    print("[6/6] plotting ...")
    plot_all(perm_imp, allcoef, lasso_sel, shap_imp, waterfall, inter, pd_df, ice_df)

    print("done.")


if __name__ == "__main__":
    run()
