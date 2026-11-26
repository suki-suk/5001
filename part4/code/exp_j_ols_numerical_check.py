# -*- coding: utf-8 -*-
"""
Part 4 - Diagnostic: is the OLS "collapse" on matrix B real or numerical?
=========================================================================
Part 3's ``A.py`` reported OLS at CV R2 = +0.906 on the 342-column matrix,
while Part 4's ``exp_h_factorial.py`` reported -2.919 on what is supposed to
be the same model.  OLS predictions are **scale-equivariant**: rescaling the
columns cannot change the fitted values, so two numbers that differ this much
cannot both be "the model's performance".  One of them is numerical noise.

This script settles the question by fitting the *same* OLS model under
several equivalent scalings and comparing the predictions directly.

Outputs
-------
* ``output/ols_scaling_sensitivity.csv``  - predictions and CV scores per variant
* ``figures/fig21_ols_numerical_instability.png``

Author: Yifei Lv (Part 4)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C

CORE_CONT = ["GrLivArea", "TotalSF", "TotalBsmtSF", "LotArea",
             "GarageArea", "1stFlrSF", "YearBuilt", "OverallQual"]


def build_expander(X: pd.DataFrame, expand: bool, global_scale: bool):
    """ColumnTransformer for matrix A (250) or matrix B (342).

    ``global_scale`` adds a StandardScaler *after* the transformer, which
    rescales the passthrough columns.  OLS is invariant to that mathematically -
    so any change in the results is pure floating-point noise.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (PolynomialFeatures, SplineTransformer,
                                       StandardScaler)

    cont = [c for c in CORE_CONT if c in X.columns]
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


def run():
    C.apply_style()
    ds = C.load_dataset()
    X, y = ds.X, ds.y
    X_tr, X_te, y_tr, y_te = C.split_80_20(ds)

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    rows = []
    preds = {}

    variants = [
        ("A · 250 engineered", False, False),
        ("A · 250 engineered + global scale", False, True),
        ("B · 342 expanded", True, False),
        ("B · 342 expanded + global scale", True, True),
    ]

    for label, expand, gscale in variants:
        pre = build_expander(X, expand, gscale)
        Xtr = pre.fit_transform(X_tr)
        Xte = pre.transform(X_te)

        if gscale:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

        mdl = LinearRegression()
        mdl.fit(Xtr, y_tr)
        p = mdl.predict(Xte)
        preds[label] = p

        mse = float(np.mean((y_te - p) ** 2))
        r2 = float(1 - mse / np.var(y_te))

        cv = cross_val_score(
            LinearRegression(), Xtr, y_tr,
            cv=C.kfold_splits(), scoring="r2", n_jobs=1)

        # conditioning of the design matrix
        Z = np.asarray(Xtr, dtype=float)
        rank = int(np.linalg.matrix_rank(Z))
        cond = float(np.linalg.cond(Z))

        rows.append({
            "variant": label,
            "n_columns": Z.shape[1],
            "rank": rank,
            "rank_deficiency": Z.shape[1] - rank,
            "condition_number": cond,
            "holdout_R2": r2,
            "holdout_MSE": mse,
            "CV_R2_mean": float(cv.mean()),
            "CV_R2_std": float(cv.std()),
            "CV_R2_min": float(cv.min()),
            "CV_R2_max": float(cv.max()),
        })
        print(f"{label:38s} cols={Z.shape[1]:4d} rank={rank:4d} "
              f"cond={cond:.3e}  holdout R2={r2:+.4f}  CV R2={cv.mean():+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "ols_scaling_sensitivity.csv"),
              index=False)

    # ---- how much do the two scalings of matrix B actually disagree? ----
    pb = preds["B · 342 expanded"]
    pbs = preds["B · 342 expanded + global scale"]
    diff = float(np.max(np.abs(pb - pbs)))
    rel = float(np.mean(np.abs(pb - pbs)) / np.std(y_te))
    print(f"\nMatrix B, two equivalent scalings:")
    print(f"   max |prediction difference| = {diff:.4f}  (log scale)")
    print(f"   relative to target sd      = {rel:.4f}")

    _plot(df)
    return df


def _plot(df):
    import matplotlib.pyplot as plt

    navy, orange = "#1F4E79", "#C55A11"
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))

    # ---- left : CV R2 per variant ----
    ax = axes[0]
    labels = [v.replace(" · ", "\n") for v in df["variant"]]
    colors = [navy, "#3A6EA5", orange, "#E39A6B"]
    bars = ax.bar(range(len(df)), df["CV_R2_mean"], color=colors)
    ax.axhline(0, color="#B00020", lw=1.2)
    for i, r in df.iterrows():
        lo, hi = r["CV_R2_min"], r["CV_R2_max"]
        ax.plot([i, i], [lo, hi], color="#333333", lw=1.2, zorder=3)
        ax.text(i, hi + 0.25, f"{r['CV_R2_mean']:+.3f}", ha="center",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("10-fold CV R2  (OLS, matrix B)")
    ax.set_title("The same OLS model, two mathematically equivalent scalings")
    ax.set_ylim(-4.2, 1.5)
    ax.text(0.02, 0.06, "vertical bars = min-max across the 10 folds",
            transform=ax.transAxes, fontsize=7.5, style="italic", color="#555555")

    # ---- right : conditioning ----
    ax2 = axes[1]
    x = np.arange(len(df))
    w = 0.38
    b1 = ax2.bar(x - w / 2, df["rank_deficiency"], w, color=navy,
                 label="rank-deficient columns")
    b2 = ax2.bar(x + w / 2, df["condition_number"], w, color=orange,
                 label="condition number (log scale)")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("rank deficiency  /  condition number")
    ax2.set_title("Why: the expanded matrix is numerically singular")
    ax2.legend(loc="upper left", fontsize=8)
    for bars, col in ((b1, navy), (b2, orange)):
        for r in bars:
            ax2.text(r.get_x() + r.get_width() / 2, r.get_height() * 1.15,
                     f"{r.get_height():.1f}", ha="center", va="bottom",
                     fontsize=7.5, color=col)

    fig.suptitle("OLS on the expanded matrix is ill-conditioned: its score is a "
                 "numerical artifact, not a modelling result",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(C.FIG_DIR, "fig21_ols_numerical_instability.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved:", p)


if __name__ == "__main__":
    run()
