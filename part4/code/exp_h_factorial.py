# -*- coding: utf-8 -*-
"""
Part 4 - Experiment H: Unified 2x2 factorial benchmark
=========================================================
Why this experiment exists
---------------------------
Part 2 and Part 3 each produced valid results, but on **different data matrices
and different sample sets**, so their headline numbers cannot be placed side by
side:

    Part 2 : train_clean.csv, 1460 rows, 223 patsy-expanded features
    Part 3 : train_features.csv, 1458 rows, 250 engineered / 342 expanded features

This experiment rebuilds the whole comparison as a proper **2 x 2 factorial
design**, so that every cell is measured under one identical protocol:

                          matrix A (250 cols)      matrix B (342 cols)
                          Part 1 engineered        + poly/spline expansion
    linear family          cell A1                   cell B1
    tree family            cell A2                   cell B2

Read the table horizontally for "which model family suits this matrix" and
vertically for "does automatic non-linear expansion buy anything".

Design rule
-----------
Matrix A and matrix B differ in **exactly one thing**: the 8 core continuous
variables are replaced by their polynomial + spline expansion in matrix B.
Everything else - the 242 passthrough columns, the sample set, the target, the
split, the CV folds, the metric code - is identical.

Protocol (identical for every cell)
------------------------------------
* data   : Part 1's ``train_features.csv``  (1458 rows x 250 cols)
* target : ``SalePriceLog = log(SalePrice)``
* split  : ``train_test_split(test_size=0.2, random_state=42)``
* CV     : ``KFold(n_splits=10, shuffle=True, random_state=42)``
* metrics: RMSE / MAE / R2, all on the log scale

Outputs
-------
* ``output/unified_factorial.csv``       - all cells, long format
* ``output/unified_factorial_wide.csv``   - the headline comparison table
* ``output/unified_factorial_deltas.csv`` - marginal effect decomposition
* ``figures/fig17_factorial_matrix.png``
* ``figures/fig18_factorial_deltas.png``

Author: Yifei Lv (Part 4)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C


# ---------------------------------------------------------------- 矩阵构造
def build_expander(ds, expand: bool):
    """Return the ColumnTransformer that turns matrix A into matrix B.

    Matrix A  : Part 1's engineered matrix, unchanged (250 columns).
    Matrix B  : the 8 core continuous variables are replaced by
                ``PolynomialFeatures(degree=2)`` (44 cols) plus
                ``SplineTransformer(n_knots=5, degree=3)`` (56 cols),
                exactly matching Part 3's B.py parameters.

    The other 242 columns pass through untouched in both matrices, so A and B
    differ in exactly one factor.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

    X = ds.X
    cont = [c for c in C.CORE_CONTINUOUS if c in X.columns]
    other = [c for c in X.columns if c not in cont]

    if not expand:
        return ColumnTransformer(
            [("core", Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler())]), cont),
             ("rest", "passthrough", other)],
            remainder="drop", verbose_feature_names_out=False)

    return ColumnTransformer(
        [
            ("nl_poly", Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("sc", StandardScaler()),
                                   ("poly", PolynomialFeatures(degree=2, include_bias=False))]),
             cont),
            ("nl_spline", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                    ("spline", SplineTransformer(n_knots=5, degree=3))]),
             cont),
            ("rest", "passthrough", other),
        ],
        remainder="drop", verbose_feature_names_out=False)


# ---------------------------------------------------------------- 模型族
def build_models(expand: bool):
    """Five candidates.

    * Linear models scale every column, then pick alpha by CV inside the
      training folds only.
    * Tree models are scale-invariant, so they consume the same expanded
      matrix directly - which means they see exactly the information the
      linear models see.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    alpha_grid = np.logspace(-3, 3, 100)

    def scaled(model):
        return Pipeline([("scale", StandardScaler()), ("model", model)])

    ridge = GridSearchCV(scaled(Ridge()), {"model__alpha": alpha_grid},
                         cv=inner_cv, scoring="neg_mean_squared_error", n_jobs=-1)
    lasso = GridSearchCV(scaled(Lasso(random_state=C.RANDOM_STATE,
                                       max_iter=50000, tol=1e-3)),
                         {"model__alpha": alpha_grid},
                         cv=inner_cv, scoring="neg_mean_squared_error", n_jobs=-1)

    return {
        "OLS": scaled(LinearRegression()),
        "Ridge": ridge,
        "Lasso": lasso,
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2,
            random_state=C.RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            random_state=C.RANDOM_STATE),
    }


# ---------------------------------------------------------------- 指标
def metrics(y_true, y_pred_log):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "MSE_log": mean_squared_error(y_true, y_pred_log),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_true, y_pred_log))),
        "MAE_log": mean_absolute_error(y_true, y_pred_log),
        "R2": r2_score(y_true, y_pred_log),
    }


# ---------------------------------------------------------------- 主流程
def run():
    C.apply_style()
    ds = C.load_dataset()
    print(ds.summary())

    X, y = ds.X, ds.y
    X_tr, X_te, y_tr, y_te = C.split_80_20(ds)

    rows = []
    for matrix_name, expand in [("A_250_feature_engineered", False),
                                ("B_342_poly_spline", True)]:
        pre = build_expander(ds, expand)
        Xtr_p = pre.fit_transform(X_tr)
        Xte_p = pre.transform(X_te)
        n_feat = Xtr_p.shape[1]
        print(f"\n[{matrix_name}]  expanded matrix = {n_feat} columns")

        # the full matrix, transformed by the same preprocessor inside each CV fold
        from sklearn.model_selection import cross_validate
        from sklearn.pipeline import Pipeline

        for mname, model in build_models(expand).items():
            model.fit(Xtr_p, y_tr)
            pred = model.predict(Xte_p)
            m = metrics(y_te, pred)

            pipe = Pipeline([("prep", build_expander(ds, expand)), ("model", model)])
            cv = cross_validate(
                pipe, X, y, cv=C.kfold_splits(), n_jobs=-1,
                scoring={"rmse": "neg_root_mean_squared_error",
                         "mae": "neg_mean_absolute_error",
                         "r2": "r2"},
            )

            row = {
                "matrix": matrix_name,
                "n_features": n_feat,
                "model": mname,
                **m,
                "CV_RMSE_log": -cv["test_rmse"].mean(),
                "CV_RMSE_std": cv["test_rmse"].std(),
                "CV_MAE_log": -cv["test_mae"].mean(),
                "CV_R2": cv["test_r2"].mean(),
                "CV_R2_std": cv["test_r2"].std(),
            }
            if hasattr(model, "best_params_"):
                row["alpha"] = model.best_params_["model__alpha"]
            rows.append(row)
            print(f"   {mname:20s} RMSE_log={m['RMSE_log']:.5f}  R2={m['R2']:+.4f}  "
                  f"CV_RMSE={row['CV_RMSE_log']:.5f}  CV_R2={row['CV_R2']:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "unified_factorial.csv"), index=False)

    # ---------------- wide comparison table ----------------
    wide = df.pivot(index="model", columns="matrix",
                    values=["RMSE_log", "R2", "CV_RMSE_log", "CV_R2"])
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(os.path.join(C.OUT_DIR, "unified_factorial_wide.csv"), index=False)

    # ---------------- marginal effect decomposition ----------------
    deltas = []
    for mname in df["model"].unique():
        a = df[(df.model == mname) & (df.matrix.str.startswith("A"))].iloc[0]
        b = df[(df.model == mname) & (df.matrix.str.startswith("B"))].iloc[0]
        deltas.append({
            "model": mname,
            "RMSE_A": a["RMSE_log"], "RMSE_B": b["RMSE_log"],
            "d_RMSE": b["RMSE_log"] - a["RMSE_log"],
            "d_RMSE_pct": (b["RMSE_log"] - a["RMSE_log"]) / a["RMSE_log"] * 100,
            "CV_R2_A": a["CV_R2"], "CV_R2_B": b["CV_R2"],
            "d_CV_R2": b["CV_R2"] - a["CV_R2"],
            "CV_RMSE_A": a["CV_RMSE_log"], "CV_RMSE_B": b["CV_RMSE_log"],
            "d_CV_RMSE": b["CV_RMSE_log"] - a["CV_RMSE_log"],
        })
    dd = pd.DataFrame(deltas)
    dd.to_csv(os.path.join(C.OUT_DIR, "unified_factorial_deltas.csv"), index=False)

    print("\n" + "=" * 92)
    print("2x2 FACTORIAL RESULT")
    print("=" * 92)
    print(dd.to_string(index=False))
    print("\nwide ->", os.path.join(C.OUT_DIR, "unified_factorial_wide.csv"))

    _plot(df, dd)
    return df, dd


# ---------------------------------------------------------------- 绘图
def _plot(df, dd):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3))

    ax = axes[0]
    models = list(dd["model"])
    x = np.arange(len(models))
    w = 0.38
    rmse_a = dd["RMSE_A"].to_numpy()
    rmse_b = dd["RMSE_B"].to_numpy()
    bars1 = ax.bar(x - w / 2, rmse_a, w, label="A · 250 cols (Part 1 engineered)",
                   color="#1F4E79")
    bars2 = ax.bar(x + w / 2, rmse_b, w, label="B · 342 cols (+ poly / spline)",
                   color="#C55A11")
    for bars in (bars1, bars2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.0004,
                    f"{r.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.set_ylabel("RMSE  (log scale, lower is better)")
    ax.set_title("80/20 holdout · identical split in every cell")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(rmse_a.max(), rmse_b.max()) * 1.24)

    ax2 = axes[1]
    cv_a = dd["CV_R2_A"].to_numpy()
    cv_b = dd["CV_R2_B"].to_numpy()
    bars1 = ax2.bar(x - w / 2, cv_a, w, label="A · 250 cols", color="#1F4E79")
    bars2 = ax2.bar(x + w / 2, cv_b, w, label="B · 342 cols", color="#C55A11")
    for bars in (bars1, bars2):
        for r in bars:
            ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.002,
                     f"{r.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=18, ha="right")
    ax2.set_ylabel("CV-10 R2  (higher is better)")
    ax2.set_title("10-fold cross-validation · the protocol that decides")
    ax2.legend(loc="lower right", fontsize=8)
    lo = min(cv_a.min(), cv_b.min())
    hi = max(cv_a.max(), cv_b.max())
    ax2.set_ylim(lo - (hi - lo) * 0.38, hi + (hi - lo) * 0.16)

    fig.suptitle("Factorial benchmark:  model family  x  feature matrix",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = os.path.join(C.FIG_DIR, "fig17_factorial_matrix.png")
    fig.savefig(p1)
    plt.close(fig)
    print("saved:", p1)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    colors = ["#C55A11" if v > 0 else "#2E7D32" for v in dd["d_RMSE_pct"]]
    bars = ax.barh(dd["model"], dd["d_RMSE_pct"], color=colors)
    for r, v in zip(bars, dd["d_RMSE_pct"]):
        ax.text(v + (0.4 if v >= 0 else -0.4), r.get_y() + r.get_height() / 2,
                f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("change in 80/20 RMSE when poly+spline is added  (%)")
    ax.set_title("Vertical comparison: does automatic non-linear expansion help?")
    ax.invert_yaxis()
    fig.tight_layout()
    p2 = os.path.join(C.FIG_DIR, "fig18_factorial_deltas.png")
    fig.savefig(p2)
    plt.close(fig)
    print("saved:", p2)


if __name__ == "__main__":
    run()
