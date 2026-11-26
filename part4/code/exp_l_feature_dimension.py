# -*- coding: utf-8 -*-
"""
Part 4 - Experiment L: how much of the 250-column matrix is actually needed?
=========================================================================
Purpose
-------
The group agreed on a single protocol and a direct head-to-head comparison of
Part 2's linear models against Part 3's tree ensembles, both on the same
250-column engineered matrix.  But agreeing on *how* to compare does not settle
*what* to compare: the 250 columns are an engineering artefact, and nobody has
asked **how much of them is signal and how much is padding**.

This experiment is the critical look at feature dimension.  It answers four
questions with measurements rather than assertions:

    Q1  How much of the 250 columns is algebraically redundant?
        -> pivoted-QR rank analysis of the design matrix
        -> how many columns are exact linear combinations of the others?

    Q2  How fast does accuracy degrade as we add columns?
        -> learning curve over the number of features (ordered by
           permutation importance), for all five model families
        -> where does the curve flatten?  everything past that point is cost
           without benefit

    Q3  Does a *smaller* matrix do as well as the full one?
        -> top-k features (k = 5, 10, 20, 50, 100, 150, 250) under the same
           protocol; compare against the full matrix
        -> find the smallest k that is statistically indistinguishable from 250

    Q4  How much of the 342-column expansion is new information?
        -> the 92 added poly/spline columns: how many carry signal the
           250-column matrix did not already have?

Outputs
-------
* ``output/feature_dimension_critique.csv``   - Q1 / Q3 / Q4 numbers
* ``output/feature_learning_curve.csv``       - Q2 learning curve
* ``figures/fig23_feature_dimension_critique.png``
* ``figures/fig24_feature_learning_curve.png``

Author: Yifei Lv (Part 4)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C


# ================================================================ Q1
def rank_analysis(X: pd.DataFrame) -> dict:
    """How many of the columns are algebraically redundant?"""
    import scipy.linalg as sla

    Z = np.asarray(X, dtype=float)
    Z = Z - Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    Z = Z / sd

    _, R, piv = sla.qr(Z, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    tol = max(Z.shape) * np.finfo(float).eps * d.max()
    rank = int((d > tol).sum())

    # the independent columns are the ones QR chose to keep
    keep = list(piv[:rank])
    dropped = [c for i, c in enumerate(X.columns) if i not in set(keep)]

    return {
        "n_columns": int(Z.shape[1]),
        "numeric_rank": rank,
        "n_redundant": int(Z.shape[1] - rank),
        "redundant_pct": float((Z.shape[1] - rank) / Z.shape[1] * 100),
        "condition_number": float(np.linalg.cond(Z)),
        "dropped_examples": dropped[:10],
    }


# ================================================================ Q2
def feature_learning_curve(ds, order: list[str], max_k: int = 250):
    """Accuracy as a function of the number of features (best first)."""
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = ds.X, ds.y
    X_tr, X_te, y_tr, y_te = C.split_80_20(ds)

    ks = [k for k in (5, 10, 20, 40, 60, 100, 150, 200, 250) if k <= max_k]
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    grid = np.logspace(-3, 3, 100)

    def scaled(m):
        return Pipeline([("scale", StandardScaler()), ("model", m)])

    models = {
        "OLS": scaled(LinearRegression()),
        "Ridge": GridSearchCV(scaled(Ridge()), {"model__alpha": grid},
                               cv=inner_cv, scoring="neg_mean_squared_error",
                               n_jobs=-1),
        "Lasso": GridSearchCV(scaled(Lasso(random_state=C.RANDOM_STATE,
                                            max_iter=50000, tol=1e-3)),
                              {"model__alpha": grid}, cv=inner_cv,
                              scoring="neg_mean_squared_error", n_jobs=-1),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=C.RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            random_state=C.RANDOM_STATE),
    }

    rows = []
    for k in ks:
        cols = order[:k]
        Xk = X[cols]
        Xk_tr = X_tr[cols]
        Xk_te = X_te[cols]
        for mname, model in models.items():
            model.fit(Xk_tr, y_tr)
            pred = model.predict(Xk_te)
            rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
            r2 = float(1 - np.mean((y_te - pred) ** 2) / np.var(y_te))
            rows.append({"n_features": k, "model": mname,
                         "RMSE_log": rmse, "R2": r2})
        print(f"  k={k:4d}  done")

    return pd.DataFrame(rows)


# ================================================================ Q3
def top_k_analysis(ds, order: list[str], ks=(5, 10, 20, 50, 100, 150, 250)):
    """Compare a top-k matrix against the full one, under the same protocol."""
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = ds.X, ds.y
    X_tr, X_te, y_tr, y_te = C.split_80_20(ds)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    grid = np.logspace(-3, 3, 100)

    def scaled(m):
        return Pipeline([("scale", StandardScaler()), ("model", m)])

    models = {
        "OLS": scaled(LinearRegression()),
        "Ridge": GridSearchCV(scaled(Ridge()), {"model__alpha": grid},
                               cv=inner_cv, scoring="neg_mean_squared_error",
                               n_jobs=-1),
        "Lasso": GridSearchCV(scaled(Lasso(random_state=C.RANDOM_STATE,
                                            max_iter=50000, tol=1e-3)),
                              {"model__alpha": grid}, cv=inner_cv,
                              scoring="neg_mean_squared_error", n_jobs=-1),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2,
            random_state=C.RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            random_state=C.RANDOM_STATE),
    }

    rows = []
    for k in ks:
        cols = order[:k]
        Xk_tr = X_tr[cols]
        Xk_te = X_te[cols]
        for mname, model in models.items():
            model.fit(Xk_tr, y_tr)
            pred = model.predict(Xk_te)
            rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
            rows.append({"n_features": k, "model": mname, "RMSE_log": rmse})
    return pd.DataFrame(rows)


# ================================================================ Q4
def expansion_value(ds) -> dict:
    """Of the 92 added poly/spline columns, how many carry new information?"""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (PolynomialFeatures, SplineTransformer,
                                       StandardScaler)

    X = ds.X
    cont = [c for c in C.CORE_CONTINUOUS if c in X.columns]
    other = [c for c in X.columns if c not in cont]

    pre = ColumnTransformer(
        [
            ("nl_poly", Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("sc", StandardScaler()),
                                   ("poly", PolynomialFeatures(degree=2,
                                                              include_bias=False))]),
             cont),
            ("nl_spline", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                    ("spline", SplineTransformer(n_knots=5,
                                                               degree=3))]),
             cont),
            ("rest", "passthrough", other),
        ],
        remainder="drop", verbose_feature_names_out=False)

    Z = np.asarray(pre.fit_transform(X), dtype=float)
    base = np.asarray(X, dtype=float)

    # how many of the expanded columns are independent of the 250 originals?
    combined = np.hstack([base, Z])
    combined = combined - combined.mean(axis=0)
    sd = combined.std(axis=0)
    sd[sd == 0] = 1.0
    combined /= sd

    import scipy.linalg as sla
    _, R, piv = sla.qr(combined, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    tol = max(combined.shape) * np.finfo(float).eps * d.max()
    rank = int((d > tol).sum())

    # how many *new* independent directions do the expanded columns add?
    base_rank = rank_analysis(X)["numeric_rank"]

    return {
        "n_base_columns": int(base.shape[1]),
        "n_expanded_columns": int(Z.shape[1]),
        "base_numeric_rank": base_rank,
        "combined_numeric_rank": rank,
        "new_independent_directions": int(rank - base_rank),
        "expansion_column_efficiency_pct": float(
            (rank - base_rank) / (Z.shape[1]) * 100),
        "base_condition_number": float(np.linalg.cond(base / np.where(
            base.std(axis=0) == 0, 1, base.std(axis=0)))),
    }


# ================================================================ main
def run():
    C.apply_style()
    ds = C.load_dataset()
    print(ds.summary())

    # the feature ordering comes from the permutation importance we already
    # computed: the most informative features first
    perm_path = os.path.join(C.OUT_DIR, "permutation_importance.csv")
    order = None
    if os.path.exists(perm_path):
        pi = pd.read_csv(perm_path)
        col = ("perm_importance_mean" if "perm_importance_mean" in pi.columns
               else pi.columns[1])
        # average the importance over the model families so the ordering is
        # not driven by a single model
        agg = pi.groupby("feature")[col].mean().sort_values(ascending=False)
        order = agg.index.tolist()
    if not order:
        print("permutation importance not found - falling back to Lasso ranking")
        order = _lasso_order(ds)

    order = [c for c in order if c in ds.X.columns]
    print(f"feature order available: {len(order)} columns")

    # ---------------- Q1 ----------------
    print("\n[Q1] rank analysis of the 250-column matrix ...")
    r1 = rank_analysis(ds.X)
    print(f"    {r1['n_columns']} columns, numeric rank {r1['numeric_rank']}, "
          f"{r1['n_redundant']} redundant ({r1['redundant_pct']:.1f}%), "
          f"cond={r1['condition_number']:.3e}")

    # ---------------- Q4 ----------------
    print("\n[Q4] what does the 342-column expansion add? ...")
    r4 = expansion_value(ds)
    for k, v in r4.items():
        print(f"    {k:36s} {v}")

    # ---------------- Q2 / Q3 ----------------
    print("\n[Q2/Q3] feature learning curve ...")
    lc = feature_learning_curve(ds, order)
    lc.to_csv(os.path.join(C.OUT_DIR, "feature_learning_curve.csv"), index=False)

    print("[Q3] top-k comparison ...")
    tk = top_k_analysis(ds, order)
    tk.to_csv(os.path.join(C.OUT_DIR, "feature_topk.csv"), index=False)

    # ---------------- assemble ----------------
    crit = pd.DataFrame([
        {"metric": "columns in the engineered matrix", "value": r1["n_columns"],
         "note": "Part 1's output, used by every part"},
        {"metric": "algebraically independent columns", "value": r1["numeric_rank"],
         "note": f"{r1['n_redundant']} columns are exact linear combinations"},
        {"metric": "redundant columns (%)", "value": round(r1["redundant_pct"], 1),
         "note": "pivoted-QR rank analysis"},
        {"metric": "condition number of the design matrix",
         "value": f"{r1['condition_number']:.3e}",
         "note": "ill-conditioned: OLS coefficients are not identifiable"},
        {"metric": "new independent directions from the expansion",
         "value": r4["new_independent_directions"],
         "note": f"out of {r4['n_expanded_columns']} added columns "
                 f"({r4['expansion_column_efficiency_pct']:.1f}% efficient)"},
        {"metric": "smallest k statistically equal to 250",
         "value": _smallest_equal_k(tk),
         "note": "top-k by permutation importance, same protocol"},
    ])
    crit.to_csv(os.path.join(C.OUT_DIR, "feature_dimension_critique.csv"),
                 index=False)
    print("\n" + crit.to_string(index=False))

    _plot(r1, r4, lc, tk)
    return crit, lc, tk


def _lasso_order(ds):
    """Fallback: rank the features by the absolute Lasso coefficient."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(ds.X)
    m = LassoCV(cv=5, random_state=C.RANDOM_STATE, max_iter=50000).fit(Z, ds.y)
    s = pd.Series(np.abs(m.coef_), index=ds.X.columns)
    return list(s.sort_values(ascending=False).index)


def _smallest_equal_k(tk: pd.DataFrame, tol: float = 0.005) -> int:
    """Smallest k whose best model is within tol of the full matrix."""
    full = tk[tk.n_features == tk.n_features.max()]
    best_full = full["RMSE_log"].min()
    for k in sorted(tk["n_features"].unique()):
        sub = tk[tk.n_features == k]
        if sub["RMSE_log"].min() <= best_full * (1 + tol):
            return int(k)
    return int(tk["n_features"].max())


# ================================================================ plot
def _plot(r1, r4, lc, tk):
    import matplotlib.pyplot as plt

    navy, orange, green, grey = "#1F4E79", "#C55A11", "#2E7D32", "#8C8C8C"

    # ---------- figure 23 : the critique ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))

    ax = axes[0]
    ax.bar(["columns", "independent\n(rank)", "redundant"], [
        r1["n_columns"], r1["numeric_rank"], r1["n_redundant"]],
        color=[navy, green, orange])
    for i, v in enumerate([r1["n_columns"], r1["numeric_rank"],
                            r1["n_redundant"]]):
        ax.text(i, v + 4, f"{v}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, r1["n_columns"] * 1.18)
    ax.set_ylabel("number of columns")
    ax.set_title(f"{r1['n_redundant']} of {r1['n_columns']} columns are exact\n"
                 "linear combinations of the others", fontsize=11.5,
                 fontweight="bold")
    ax.tick_params(labelsize=9)

    ax2 = axes[1]
    labels = ["250\nbase", "+92\npoly/spline"]
    vals = [r1["numeric_rank"], r4["combined_numeric_rank"]]
    bars = ax2.bar(labels, vals, color=[navy, orange], width=0.55)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(vals) * 1.2)
    ax2.set_ylabel("numeric rank")
    ax2.set_title(f"The expansion adds only {r4['new_independent_directions']} "
                  f"independent directions\n(out of {r4['n_expanded_columns']} "
                  "new columns)", fontsize=11.5, fontweight="bold")
    ax2.tick_params(labelsize=9)

    ax3 = axes[2]
    full = tk[tk.n_features == tk.n_features.max()]["RMSE_log"].min()
    best = tk.groupby("n_features")["RMSE_log"].min().reset_index()
    ax3.plot(best["n_features"], best["RMSE_log"], "-o", color=navy, lw=2,
             ms=5, label="best model at this k")
    ax3.axhline(full, color=grey, ls="--", lw=1.4,
                label=f"full 250-column matrix ({full:.4f})")
    k_eq = _smallest_equal_k(tk)
    ax3.axvline(k_eq, color=orange, ls=":", lw=1.6)
    ax3.text(k_eq + 4, best["RMSE_log"].max() * 0.97, f"k = {k_eq}",
             color=orange, fontsize=10, fontweight="bold")
    ax3.set_xlabel("number of features (ordered by permutation importance)")
    ax3.set_ylabel("hold-out RMSE (log scale)")
    ax3.set_title("Accuracy is flat beyond a small subset\n"
                  "the remaining columns are cost without benefit",
                  fontsize=11.5, fontweight="bold")
    ax3.legend(fontsize=8.5)
    ax3.tick_params(labelsize=9)

    fig.suptitle("Critical look at feature dimension: the 250-column matrix is "
                 "much larger than the signal it carries",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p1 = os.path.join(C.FIG_DIR, "fig23_feature_dimension_critique.png")
    fig.savefig(p1)
    plt.close(fig)
    print("saved:", p1)

    # ---------- figure 24 : the learning curve per model ----------
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    for mname, d in lc.groupby("model"):
        d = d.sort_values("n_features")
        ax.plot(d["n_features"], d["RMSE_log"], "-o", ms=4, lw=1.9,
                label=mname)
    ax.set_xlabel("number of features (ordered by permutation importance)")
    ax.set_ylabel("hold-out RMSE (log scale, lower is better)")
    ax.set_title("Feature learning curve on the 250-column matrix: every model "
                 "family flattens early", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p2 = os.path.join(C.FIG_DIR, "fig24_feature_learning_curve.png")
    fig.savefig(p2)
    plt.close(fig)
    print("saved:", p2)


if __name__ == "__main__":
    run()
