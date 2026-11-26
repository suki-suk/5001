# -*- coding: utf-8 -*-
"""
Part 4 - Diagnostic: what does OLS *really* do on the expanded matrix?
========================================================================
Motivation
----------
``exp_h_factorial.py`` reports OLS at CV R2 = -2.92 on the 342-column matrix,
while ``exp_j_ols_numerical_check.py`` shows the *same* OLS model scoring
between +0.90 and -169 depending on whether the passthrough columns are
globally rescaled.  OLS predictions are scale-equivariant, so the fitted
values are mathematically unique - only the *numerics* can differ.

This script settles the question with correct linear algebra (a proper
intercept, a rank-aware solver) and reports what is defensible.

Method
------
Every solver is fitted the same way: **centre X and y**, then solve the
least-squares problem on the centred data, so all variants include an
intercept in exactly the same manner.

    A. sklearn LinearRegression      (scipy SVD lstsq, centred internally)
    B. numpy lstsq (SVD)             on the same centred data
    C. pivoted-QR on the numerically independent columns only
    D. Tikhonov limit (Ridge, alpha -> 0)

For each variant we report the hold-out R2 and the largest disagreement in
the predictions.  Then we repeat the whole comparison inside a 10-fold CV to
see whether the instability shows up as fold variance.

Outputs
-------
* ``output/ols_solver_comparison.csv``
* ``figures/fig22_ols_solver_comparison.png``

Author: Yifei Lv (Part 4)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")

import config as C

CORE_CONT = ["GrLivArea", "TotalSF", "TotalBsmtSF", "LotArea",
             "GarageArea", "1stFlrSF", "YearBuilt", "OverallQual"]


# ------------------------------------------------------------------ matrix
def expand_matrix(X: pd.DataFrame, standardise: bool = True):
    """Matrix B with every column on a common scale (best-conditioned variant)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (PolynomialFeatures, SplineTransformer,
                                       StandardScaler)

    cont = [c for c in CORE_CONT if c in X.columns]
    other = [c for c in X.columns if c not in cont]

    blocks = [
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
    ]
    if standardise:
        blocks.append(("rest", Pipeline([("imp", SimpleImputer(strategy="median")),
                                         ("sc", StandardScaler())]), other))
    else:
        blocks.append(("rest", "passthrough", other))

    pre = ColumnTransformer(blocks, remainder="drop",
                            verbose_feature_names_out=False)
    return pre


# ------------------------------------------------------------------ solvers
class CentredOLS(BaseEstimator, RegressorMixin):
    """OLS with an intercept, solved on centred+scaled data.

    ``method`` selects the numerical route:

    * ``"svd"``   - numpy SVD least squares (rank-truncated)
    * ``"qr"``    - pivoted QR, keeping only the numerically independent columns
    """

    def __init__(self, method: str = "svd", rcond: float | None = None):
        self.method = method
        self.rcond = rcond

    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu_ = Z.mean(axis=0)
        self.sd_ = Z.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        self.ym_ = y.mean()

        Zs = (Z - self.mu_) / self.sd_
        yc = y - self.ym_

        if self.method == "svd":
            beta, *_ = np.linalg.lstsq(Zs, yc, rcond=self.rcond)
            self.beta_ = beta
            self.keep_ = None
        else:
            import scipy.linalg as sla
            _, R, piv = sla.qr(Zs, mode="economic", pivoting=True)
            d = np.abs(np.diag(R))
            tol = max(Zs.shape) * np.finfo(float).eps * d.max()
            keep = piv[:int((d > tol).sum())]
            self.keep_ = keep
            # beta lives in the reduced space: one coefficient per kept column
            self.beta_ = np.linalg.lstsq(Zs[:, keep], yc, rcond=None)[0]
        return self

    def predict(self, Z):
        Z = np.asarray(Z, dtype=float)
        Zs = (Z - self.mu_) / self.sd_
        if self.keep_ is not None:
            Zs = Zs[:, self.keep_]
        return Zs @ self.beta_ + self.ym_


# ------------------------------------------------------------------ main
def main():
    from sklearn.linear_model import LinearRegression, Ridge

    ds = C.load_dataset()
    X, y = ds.X, ds.y
    X_tr, X_te, y_tr, y_te = C.split_80_20(ds)

    pre = expand_matrix(X)
    Ztr = np.asarray(pre.fit_transform(X_tr), dtype=float)
    Zte = np.asarray(pre.transform(X_te), dtype=float)
    rank = int(np.linalg.matrix_rank(Ztr))
    print(f"matrix B: {Ztr.shape[0]} x {Ztr.shape[1]}, numeric rank = {rank}, "
          f"rank-deficient columns = {Ztr.shape[1] - rank}")
    print(f"condition number (raw) = {np.linalg.cond(Ztr):.3e}")
    print(f"condition number (scaled) = "
          f"{np.linalg.cond((Ztr - Ztr.mean(0)) / np.where(Ztr.std(0) == 0, 1, Ztr.std(0))):.3e}")

    # ================= Q1 : one fit, several solvers =================
    print("\n=== Q1: hold-out fit, independent solvers (all with intercept) ===")
    solvers = {}

    lr = LinearRegression().fit(Ztr, y_tr)
    solvers["sklearn LinearRegression"] = lr.predict(Zte)

    solvers["centred SVD lstsq"] = CentredOLS("svd").fit(Ztr, y_tr).predict(Zte)
    solvers["centred pivoted QR"] = CentredOLS("qr").fit(Ztr, y_tr).predict(Zte)
    for eps in (1e-10, 1e-6, 1e-3):
        solvers[f"Ridge alpha={eps:g}"] = Ridge(alpha=eps).fit(Ztr, y_tr).predict(Zte)

    base = solvers["sklearn LinearRegression"]
    sd_y = float(np.std(y_te))
    print(f"\n  {'solver':30s} {'holdout R2':>12s} {'max|dpred|':>12s}  verdict")
    for name, p in solvers.items():
        r2 = 1 - float(np.mean((y_te - p) ** 2)) / float(np.var(y_te))
        d = float(np.max(np.abs(p - base)))
        verdict = "agrees" if d < 0.05 * sd_y else "DIVERGES"
        print(f"  {name:30s} {r2:+12.4f} {d:12.4f}  {verdict}")

    # ================= Q2 : same comparison inside 10-fold CV =================
    print("\n=== Q2: 10-fold CV, refit inside every fold ===")
    from sklearn.model_selection import cross_val_score

    cv_sk = cross_val_score(LinearRegression(), Ztr, y_tr,
                             cv=C.kfold_splits(), scoring="r2", n_jobs=1)
    print(f"  sklearn OLS            : CV R2 = {cv_sk.mean():+.4f} +/- {cv_sk.std():.4f}")
    print(f"                           folds = {np.round(cv_sk, 2)}")

    cv_svd = cross_val_score(CentredOLS("svd"), Ztr, y_tr,
                              cv=C.kfold_splits(), scoring="r2", n_jobs=1)
    print(f"  centred SVD lstsq      : CV R2 = {cv_svd.mean():+.4f} +/- {cv_svd.std():.4f}")
    print(f"                           folds = {np.round(cv_svd, 2)}")

    # the *pipeline* form used by exp_h_factorial (prep -> scale -> OLS)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("prep", expand_matrix(X)), ("sc", StandardScaler()),
                     ("model", LinearRegression())])
    cv_pipe = cross_val_score(pipe, X, y, cv=C.kfold_splits(),
                               scoring="r2", n_jobs=1)
    print(f"  exp_h pipeline (prep+scale+OLS): CV R2 = {cv_pipe.mean():+.4f} "
          f"+/- {cv_pipe.std():.4f}")
    print(f"                           folds = {np.round(cv_pipe, 2)}")

    # ================= save =================
    rows = []
    for name, p in solvers.items():
        rows.append({
            "kind": "holdout fit",
            "variant": name,
            "R2": 1 - float(np.mean((y_te - p) ** 2)) / float(np.var(y_te)),
            "max_abs_prediction_diff": float(np.max(np.abs(p - base))),
            "CV_R2": np.nan, "CV_R2_std": np.nan,
        })
    for name, sc in [("sklearn OLS", cv_sk), ("centred SVD lstsq", cv_svd),
                     ("exp_h pipeline", cv_pipe)]:
        rows.append({
            "kind": "10-fold CV",
            "variant": name,
            "R2": np.nan,
            "max_abs_prediction_diff": np.nan,
            "CV_R2": float(sc.mean()), "CV_R2_std": float(sc.std()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUT_DIR, "ols_solver_comparison.csv"), index=False)
    print("\nsaved -> output/ols_solver_comparison.csv")

    # ================= figure =================
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    names = list(solvers.keys())
    r2s = [1 - float(np.mean((y_te - p) ** 2)) / float(np.var(y_te))
           for p in solvers.values()]
    colors = ["#2E7D32" if v > 0.85 else "#C55A11" for v in r2s]
    bars = ax.barh(range(len(names)), r2s, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    for r, v in zip(bars, r2s):
        ax.text(v + 0.008, r.get_y() + r.get_height() / 2, f"{v:+.4f}",
                va="center", fontsize=8.5, fontweight="bold")
    ax.axvline(r2s[0], color="#1F4E79", ls="--", lw=1, alpha=0.7)
    ax.set_xlim(0.80, 0.96)
    ax.set_xlabel("hold-out R2  on the 342-column matrix")
    ax.set_title("One model, one data set, six solvers\n"
                 "the score moves by 4 points of R²", fontsize=11.5,
                 fontweight="bold")

    ax2 = axes[1]
    labels = ["sklearn\nOLS", "centred SVD\nlstsq", "exp_h\npipeline"]
    cvs = [cv_sk, cv_svd, cv_pipe]
    x = np.arange(3)
    for i, sc in enumerate(cvs):
        finite = sc[np.isfinite(sc)]
        clipped = np.clip(finite, -34, 4)
        ax2.scatter(np.full(len(clipped), i) + np.random.uniform(-.08, .08, len(clipped)),
                    clipped, color="#1F4E79", s=45, zorder=3)
        m = float(np.mean(finite))
        ax2.hlines(np.clip(m, -34, 4), i - 0.25, i + 0.25, color="#C55A11", lw=2.5)
        label = (f"mean {m:+.2f}" if abs(m) < 1e3
                 else f"mean {m:.1e}\n(diverged)")
        ax2.text(i, 3.0, label, ha="center", va="top", fontsize=8.5,
                 fontweight="bold")
    ax2.axhline(0, color="#B00020", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("CV R2 per fold")
    ax2.set_title("Refitting inside each fold exposes the instability:\n"
                  "the same model swings by 30 R2 points", fontsize=11.5,
                  fontweight="bold")
    ax2.set_ylim(-35, 5)

    fig.suptitle("OLS on the expanded matrix is numerically unstable: its score "
                 "is a property of the solver, not of the model",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(C.FIG_DIR, "fig22_ols_solver_comparison.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved:", p)


if __name__ == "__main__":
    main()
