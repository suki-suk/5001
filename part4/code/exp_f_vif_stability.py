# -*- coding: utf-8 -*-
"""
Part 4 · 实验 F：VIF 诊断与模型稳定性的连接（RQ2 的量化闭环）
===================================================================
背景
----
Part2 在原始特征上算出 **75/223 个特征 VIF > 10**、2 个无穷大、条件数 2.25×10¹⁵，
并据此判断"OLS 系数不可靠"。Part3 则在 Part1 的工程特征上发现
**10 折 CV 下 OLS 的 R² 崩溃到 −2.84**，而 Ridge/Lasso 稳定在 0.976。

这两个结论之间缺一个环节：**共线性到底把 OLS 的系数扭曲成了什么样？**
"不可靠"是一个定性判断，本实验把它变成可量化的证据。

方法
----
**Bootstrap 系数稳定性分析**：对训练集做 B 次有放回重采样，
每次重新拟合 OLS / Ridge / Lasso，记录每个特征的系数。
然后比较三个统计量：

* **相对标准差（RSD）** = std(β) / |mean(β)|  —— 越大说明该特征对样本扰动越敏感；
* **符号翻转率**   —— bootstrap 中系数正负号发生改变的比例，
                      这是共线性最典型的症状（两个高度相关的变量互相"抢"解释力，
                      换个样本就换人当主角）；
* **置信区间宽度** —— 95% bootstrap 区间覆盖了 0 的比例。

同时做 **特征工程前 / 后 VIF 对照**（Part1 README 提到 TotalBsmtSF–1stFlrSF = 0.82、
GarageCars–GarageArea = 0.88 被 TotalSF、GarageAreaPerCar 合并）：
用同一套 VIF 算法在 Part1 的工程矩阵上重算，验证工程特征是否真的缓解了共线性。

产出
----
* ``output/bootstrap_coef_stability.csv``
* ``output/vif_engineered.csv`` / ``vif_raw.csv``
* ``output/vif_before_after.csv``
* ``figures/fig10_coefficient_stability.png``
* ``figures/fig11_vif_before_after.png``
* ``figures/fig12_vif_sign_flip.png``

作者：Yifei Lv（Part4）
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C

N_BOOTSTRAP = 100
RNG_SEED = C.RANDOM_STATE

# bootstrap 结果缓存路径（重跑时复用，避免重复计算）
STAB_CACHE = os.path.join(C.OUT_DIR, "bootstrap_coef_stability.csv")


# ================================================================ 1. VIF 对照
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """对每一列计算方差膨胀因子（VIF）。

    VIF_j = 1 / (1 - R²_j)，R²_j 是把第 j 列对其余所有列做回归得到的 R²。

    实现：一次算出 Gram 矩阵 G = M'M（M 为中心化后的数据矩阵），
    之后对每一列 j 只解一个 (p-1)×(p-1) 线性方程组：

        A   = G 去掉第 j 行第 j 列      （= M₋ⱼ'M₋ⱼ）
        rhs = G 的第 j 列去掉第 j 个元素（= M₋ⱼ'xⱼ）
        xⱼ ≈ M₋ⱼ · beta

    相比逐列 lstsq（每次都要复制一份 n×p 的数据矩阵），这个写法快两个数量级。

    数值与语义说明：
    * 中心化相当于模型含截距，与 statsmodels 的 ``variance_inflation_factor`` 口径一致；
    * 当第 j 列可由其余列线性表出时残差为 0，R² = 1，VIF 恰为无穷大——
      这正是"完全共线"的正确表达，不是数值错误；
    * 零方差列直接跳过（VIF 无定义）。
    """
    data = X.to_numpy(dtype=float)
    keep = data.std(axis=0) > 0
    data = data[:, keep]
    cols = [c for c, k in zip(X.columns, keep) if k]

    n, p = data.shape
    M = data - data.mean(axis=0)
    G = M.T @ M                       # p×p Gram 矩阵

    vif = np.ones(p)
    all_idx = np.arange(p)
    for j in range(p):
        idx = all_idx[all_idx != j]
        A = G[np.ix_(idx, idx)]
        rhs = G[idx, j]
        try:
            beta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, rhs, rcond=None)[0]
        resid = M[:, j] - M[:, idx] @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(M[:, j] @ M[:, j])
        if ss_tot <= 1e-12:
            vif[j] = 1.0
            continue
        r2 = 1.0 - ss_res / ss_tot
        vif[j] = np.inf if r2 >= 1 - 1e-10 else 1.0 / (1.0 - r2)

    return pd.DataFrame({
        "feature": cols,
        "VIF": vif,
        "n_infinite": int(np.isinf(vif).sum()),
    }).sort_values("VIF", ascending=False, na_position="last").reset_index(drop=True)


def vif_summary(vif_df: pd.DataFrame, label: str) -> dict:
    """把 VIF 分成四档统计，便于比较两套矩阵的共线性结构。

    * ``完全冗余 (VIF=∞)``  —— 该列可由其它列精确线性表出（多为算术恒等式或
      完整 one-hot 组的哑变量陷阱）；
    * ``严重 (10 < VIF < ∞)`` —— 高度相关但非精确冗余，是真正威胁系数估计的一档；
    * ``中度 (5 < VIF ≤ 10)`` —— 需要留意但通常可接受；
    * ``轻微 (VIF ≤ 5)``     —— 基本无害。
    """
    v = vif_df["VIF"].to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    return {
        "matrix": label,
        "n_features": len(v),
        "n_exact_redundant": int(np.isinf(v).sum()),
        "n_severe_10_plus": int((finite > 10).sum()),
        "n_moderate_5_10": int(((finite > 5) & (finite <= 10)).sum()),
        "n_mild_le_5": int((finite <= 5).sum()),
        "n_vif_gt_10_total": int((finite > 10).sum()) + int(np.isinf(v).sum()),
        "max_finite_vif": float(finite.max()) if len(finite) else np.nan,
        "median_finite_vif": float(np.median(finite)) if len(finite) else np.nan,
    }


def build_raw_matrix() -> pd.DataFrame:
    """构造"特征工程前"的对照矩阵，口径与 Part2 完全对齐。

    Part2 用 patsy 的 ``C(var, Treatment)`` 做 one-hot，即 **每个类别组丢弃一个参照水平**。
    这里用 ``pd.get_dummies(drop_first=True)`` 复现同一口径。

    为什么必须对齐：完整 one-hot（保留全部水平）会让每个类别组内部恒有
    "各水平之和 = 1"这一恒等式，产生 k-1 个 VIF = ∞ 的伪冗余列。
    这不是数据的性质，而是编码方式的产物，只有在编码方式一致时
    "工程前 / 工程后"的 VIF 差异才能归因于特征工程本身。
    """
    clean = pd.read_csv(C.TRAIN_CLEAN)
    drop = set(C.LEAK_COLS) | {"SalePriceOutlier"}
    clean = clean.drop(columns=[c for c in drop if c in clean.columns])

    num_cols = [c for c in clean.columns
                if clean[c].dtype.kind in "if" and clean[c].nunique() > 1]
    cat_cols = [c for c in clean.columns
                if c not in num_cols and clean[c].nunique() > 1]

    out = clean[num_cols].copy()
    for c in cat_cols:
        d = pd.get_dummies(clean[c], prefix=c, drop_first=True, dtype=float)
        out = pd.concat([out, d], axis=1)
    return out


def prune_redundant(X: pd.DataFrame, max_iter: int = 50):
    """迭代剔除 VIF = ∞ 的精确冗余列，返回 (精简矩阵, 被剔除的列清单)。

    完整 one-hot 矩阵里，每个类别组内部存在 "各水平之和 = 1" 的恒等式，
    这些列的 VIF 恒为 ∞，会把真正有意义的严重共线性淹没在噪声里。
    逐轮剔除后剩下的矩阵才反映"有效"的共线性结构。
    剔除的列本身也是诊断信息——它们揭示了数据中的算术冗余
    （如 TotalSF 恰是各层面积之和，QualScore 恰是各质量分之和）。
    """
    current = X.copy()
    removed = []
    for _ in range(max_iter):
        vif = compute_vif(current)
        inf_cols = vif.loc[np.isinf(vif["VIF"]), "feature"].tolist()
        if not inf_cols:
            break
        removed.extend(inf_cols)
        current = current.drop(columns=inf_cols)
    return current, removed


# ================================================================ 2. Bootstrap 系数稳定性
def bootstrap_coefficients(ds, n_boot: int = N_BOOTSTRAP) -> pd.DataFrame:
    """对 OLS / Ridge / Lasso 各做 n_boot 次重采样，记录系数分布。"""
    from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = ds.X
    y = ds.y
    n = len(X)
    rng = np.random.default_rng(RNG_SEED)
    inner = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    grid = np.logspace(-3, 3, 100)

    specs = {
        "OLS": lambda: LinearRegression(),
        "Ridge": lambda: Pipeline([("s", StandardScaler()),
                                   ("m", RidgeCV(alphas=grid, cv=inner))]),
        "Lasso": lambda: Pipeline([("s", StandardScaler()),
                                   ("m", LassoCV(alphas=grid, cv=inner, max_iter=50000,
                                                 random_state=C.RANDOM_STATE))]),
    }

    records = []
    for name, factory in specs.items():
        coefs = []
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            Xb = X.iloc[idx]
            yb = y[idx]
            m = factory()
            m.fit(Xb, yb)
            if hasattr(m, "named_steps"):
                coefs.append(m.named_steps["m"].coef_)
            else:
                coefs.append(m.coef_)
            if (b + 1) % 25 == 0:
                print(f"    {name}: {b+1}/{n_boot} bootstraps done")

        arr = np.vstack(coefs)                       # (n_boot, n_features)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            rsd = np.where(np.abs(mu) > 1e-12, sd / np.abs(mu), np.inf)

        sign_flip = np.zeros(arr.shape[1])
        for j in range(arr.shape[1]):
            col = arr[:, j]
            nz = col[np.abs(col) > 1e-12]
            if len(nz) > 1:
                pos = (nz > 0).mean()
                sign_flip[j] = 2 * min(pos, 1 - pos)     # 越接近 0.5 越不稳定

        records.append(pd.DataFrame({
            "model": name,
            "feature": X.columns,
            "coef_mean": mu,
            "coef_std": sd,
            "rel_std": rsd,
            "sign_flip_rate": sign_flip,
            "ci_low": np.percentile(arr, 2.5, axis=0),
            "ci_high": np.percentile(arr, 97.5, axis=0),
            "frac_zero": (np.abs(arr) < 1e-10).mean(axis=0),
        }))

    out = pd.concat(records, ignore_index=True)
    out.to_csv(os.path.join(C.OUT_DIR, "bootstrap_coef_stability.csv"), index=False)
    return out


# ================================================================ 3. 作图
def plot_vif(before: pd.DataFrame, after: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    s_before = summary[summary["matrix"].str.startswith("Raw")].iloc[0]
    s_after = summary[summary["matrix"].str.startswith("Part 1")].iloc[0]

    # (a) VIF 分布对照
    ax = axes[0]
    for df, lab, color in ((before, "Raw features (Part 2 encoding)", C.COLORS["red"]),
                           (after, "Part 1 engineered matrix", C.COLORS["green"])):
        v = df["VIF"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        v = np.log10(np.clip(v, 1, None))
        ax.hist(v, bins=32, alpha=0.55, color=color, label=f"{lab}  (n={len(df)})")
    ax.axvline(1.0, ls="--", c="#333333", lw=1.2, label="VIF = 10")
    ax.set_xlabel("log₁₀(VIF)")
    ax.set_ylabel("Number of features")
    ax.set_title("(a) VIF distribution of the two matrices", loc="left")
    ax.legend(fontsize=7.5)

    # (b) 共线性分档计数（堆叠条形）
    ax = axes[1]
    labels = ["Raw features\n(Part 2 encoding)", "Part 1\nengineered matrix"]
    tiers = [("n_exact_redundant", "Exact redundancy (VIF = ∞)", "#6A0DAD"),
             ("n_severe_10_plus", "Severe (VIF > 10)", C.COLORS["red"]),
             ("n_moderate_5_10", "Moderate (5 < VIF ≤ 10)", C.COLORS["accent"]),
             ("n_mild_le_5", "Mild (VIF ≤ 5)", C.COLORS["green"])]
    bottom = np.zeros(2)
    for key, lab, color in tiers:
        vals = np.array([s_before[key], s_after[key]], dtype=float)
        ax.bar(labels, vals, 0.5, bottom=bottom, label=lab, color=color, alpha=0.92,
               edgecolor="white")
        for i, (v, b0) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(i, b0 + v / 2, f"{int(v)}", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=10)
        bottom += vals
    ax.set_ylabel("Number of features")
    ax.set_title("(b) Collinearity tiers in each matrix", loc="left")
    ax.legend(fontsize=8, loc="upper right")

    # (c) 同一特征在两套矩阵的 VIF
    ax = axes[2]
    common = set(before["feature"]) & set(after["feature"])
    b = before[before["feature"].isin(common)].set_index("feature")["VIF"]
    a = after[after["feature"].isin(common)].set_index("feature")["VIF"]
    common = sorted(common, key=lambda f: -max(b.get(f, 0), a.get(f, 0)))[:12]
    x = np.arange(len(common))
    ax.bar(x - 0.2, [b.get(f, np.nan) for f in common], 0.4, label="Raw features",
           color=C.COLORS["red"])
    ax.bar(x + 0.2, [a.get(f, np.nan) for f in common], 0.4, label="Engineered",
           color=C.COLORS["green"])
    ax.set_yscale("log")
    ax.set_xticks(x, [f.replace("_", "\n") for f in common], rotation=0, fontsize=7.5)
    ax.set_ylabel("VIF (log scale)")
    ax.axhline(10, ls="--", c="#333333", lw=1)
    ax.set_title("(c) Same feature in the two matrices", loc="left")
    ax.legend(fontsize=8)

    fig.suptitle("Multicollinearity in the raw matrix vs the engineered matrix",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(C.FIG_DIR, "fig11_vif_before_after.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)


def plot_stability(stab: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))

    # (a) 相对标准差对照
    ax = axes[0]
    top = stab[stab["model"] == "OLS"].nlargest(15, "rel_std")["feature"].tolist()
    y = np.arange(len(top))
    h = 0.8 / 3
    pal = [C.COLORS["primary"], C.COLORS["accent"], C.COLORS["green"]]
    for i, m in enumerate(["OLS", "Ridge", "Lasso"]):
        sub = stab[stab["model"] == m].set_index("feature").reindex(top)
        vals = np.clip(sub["rel_std"].to_numpy(), 0, np.nanpercentile(
            stab[stab["model"] == m]["rel_std"].replace(np.inf, np.nan).dropna(), 98))
        ax.barh(y + i * h - 0.4 + h / 2, vals, h, label=m, color=pal[i])
    ax.set_yticks(y, top)
    ax.invert_yaxis()
    ax.set_xlabel("Relative std of bootstrap coefficient  (higher = less stable)")
    ax.set_title("(a) Coefficient instability under resampling", loc="left")
    ax.legend()

    # (b) 符号翻转率
    ax = axes[1]
    models = ["OLS", "Ridge", "Lasso"]
    data = [stab[stab["model"] == m]["sign_flip_rate"].to_numpy() for m in models]
    bp = ax.boxplot(data, tick_labels=models, vert=False, patch_artist=True,
                    widths=0.55)
    for patch, color in zip(bp["boxes"], pal):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("#222222")
    ax.set_xlabel("Sign-flip rate across bootstrap samples")
    ax.set_title("(b) Sign reversal is the classic symptom of collinearity", loc="left")

    # (c) 系数 bootstrap 区间（挑选 8 个关键特征）
    ax = axes[2]
    feats = ["QualxTotalSF", "TotalSF", "OverallQual", "GrLivArea",
             "TotalBsmtSF", "GarageCars", "GarageArea", "1stFlrSF"]
    feats = [f for f in feats if f in set(stab["feature"])]
    idx = np.arange(len(feats))
    off = {"OLS": -0.26, "Ridge": 0.0, "Lasso": 0.26}
    for m in models:
        sub = stab[(stab["model"] == m)].set_index("feature").reindex(feats)
        ax.errorbar(sub["coef_mean"], idx + off[m],
                    xerr=[sub["coef_mean"] - sub["ci_low"], sub["ci_high"] - sub["coef_mean"]],
                    fmt="o", ms=4, capsize=3, lw=1.2, label=m, color=pal[models.index(m)])
    ax.set_yticks(idx, feats)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("Bootstrap coefficient  (95% CI)")
    ax.set_title("(c) Confidence intervals widen dramatically under OLS", loc="left")
    ax.legend(fontsize=8)

    fig.suptitle("Bootstrap evidence: collinearity makes OLS coefficients unstable, "
                 "ridge/lasso do not", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(C.FIG_DIR, "fig10_coefficient_stability.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)


# ================================================================ 主流程
def run():
    C.apply_style()
    ds = C.load_dataset()
    print(ds.summary())

    # ---------- 阶段 1：原始特征（Part2 口径：Treatment 参照组）----------
    print("[1/3] VIF on raw features (Treatment encoding, aligned with Part 2) ...")
    raw_full = build_raw_matrix()
    raw_pruned, raw_removed = prune_redundant(raw_full)
    print(f"    raw matrix: {raw_full.shape[1]} cols -> "
          f"{len(raw_removed)} exactly redundant removed -> {raw_pruned.shape[1]} cols")
    vif_raw = compute_vif(raw_pruned)
    vif_raw.to_csv(os.path.join(C.OUT_DIR, "vif_raw.csv"), index=False)
    pd.DataFrame({"removed_redundant_column": raw_removed}).to_csv(
        os.path.join(C.OUT_DIR, "vif_raw_redundant_removed.csv"), index=False)

    # ---------- 阶段 2：Part1 工程矩阵 ----------
    print("[2/3] VIF on Part 1 engineered matrix ...")
    eng_full = ds.X
    eng_pruned, eng_removed = prune_redundant(eng_full)
    print(f"    engineered matrix: {eng_full.shape[1]} cols -> "
          f"{len(eng_removed)} exactly redundant removed -> {eng_pruned.shape[1]} cols")
    vif_eng = compute_vif(eng_pruned)
    vif_eng.to_csv(os.path.join(C.OUT_DIR, "vif_engineered.csv"), index=False)
    pd.DataFrame({"removed_redundant_column": eng_removed}).to_csv(
        os.path.join(C.OUT_DIR, "vif_engineered_redundant_removed.csv"), index=False)

    # ---------- 汇总 ----------
    summary = pd.DataFrame([
        vif_summary(vif_raw, "Raw features (Part 2 encoding)"),
        vif_summary(vif_eng, "Part 1 engineered matrix"),
    ])
    summary.to_csv(os.path.join(C.OUT_DIR, "vif_before_after.csv"), index=False)
    print(summary.to_string(index=False))

    # 同一特征在两阶段的 VIF 对照
    common = sorted(set(vif_raw["feature"]) & set(vif_eng["feature"]),
                    key=lambda f: -max(vif_raw.set_index("feature")["VIF"].get(f, 0),
                                       vif_eng.set_index("feature")["VIF"].get(f, 0)))
    cmp_rows = []
    vr = vif_raw.set_index("feature")["VIF"]
    ve = vif_eng.set_index("feature")["VIF"]
    for f in common:
        cmp_rows.append({"feature": f, "vif_raw": vr.get(f), "vif_engineered": ve.get(f)})
    pd.DataFrame(cmp_rows).to_csv(
        os.path.join(C.OUT_DIR, "vif_paired_comparison.csv"), index=False)

    print("[3/3] bootstrap coefficient stability ...")
    if os.path.exists(STAB_CACHE):
        print("    using cached bootstrap results")
        stab = pd.read_csv(STAB_CACHE)
    else:
        stab = bootstrap_coefficients(ds)

    plot_vif(vif_raw, vif_eng, summary)
    plot_stability(stab)
    print("done.")


if __name__ == "__main__":
    run()
