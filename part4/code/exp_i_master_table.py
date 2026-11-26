# -*- coding: utf-8 -*-
"""
Part 4 - Master summary: the 2 x 2 factorial benchmark across all four parts
=============================================================================
Collects the four members' results into ONE table so that every number in the
group comes from one protocol.

The design
----------
                      matrix A (250 cols)      matrix B (342 cols)
                      Part 1 engineered        + poly/spline expansion
    linear family     Part 2                   Part 2
    tree family       Part 3                   Part 3

Sources (all produced under the same protocol: 1458 rows, log(SalePrice),
train_test_split(test_size=0.2, random_state=42), KFold(10, shuffle=True,
random_state=42)):

* Part 2  ``appendix_code/part2/tables/model_mse_compare_matrix{A,B}.csv``
* Part 3  ``appendix_code/part3/tables/member3_B_80_20_mse.csv``           (A)
           ``appendix_code/part3/tables/member3_B2_expanded_80_20_mse.csv``(B)
           ``appendix_code/part3/tables/member3_results_B_10fold.csv``      (A)
           ``appendix_code/part3/tables/member3_B2_expanded_10fold.csv``    (B)
* Part 4  ``part4/output/unified_factorial.csv``   (the 10-fold CV numbers,
           produced once for every cell so there is a single source of truth)

Outputs
-------
* ``output/master_8cell_table.csv``    - the headline table
* ``output/master_8cell_deltas.csv``   - marginal effect decomposition
* ``figures/fig19_master_8cell.png``   - the figure used in the deck

Author: Yifei Lv (Part 4)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(ROOT, "..", "appendix_code", "part2", "tables")
P3 = os.path.join(ROOT, "..", "appendix_code", "part3", "tables")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def _read(path):
    return pd.read_csv(path)


def collect():
    rows = []

    # ---------------- Part 2 : linear family ----------------
    for tag, mat in [("A", "matrixA"), ("B", "matrixB")]:
        f = os.path.join(P2, f"model_mse_compare_{mat}.csv")
        if not os.path.exists(f):
            print("  missing:", f)
            continue
        d = _read(f)
        for _, r in d.iterrows():
            rows.append({
                "cell": f"{tag}1",
                "matrix": "A_250" if tag == "A" else "B_342",
                "family": "linear",
                "model": r["Model"].replace("LASSO", "Lasso"),
                "source": "Part 2",
                "MSE_80_20": r["Test_MSE_log"],
                "RMSE_80_20": r["log_scale_RMSE"],
                "R2_80_20": r.get("R2", np.nan),
                "alpha": r.get("Selected_alpha", np.nan),
                "sparsity": r.get("sparsity", np.nan),
            })

    # ---------------- Part 3 : tree family ----------------
    p3_80 = {
        "A": os.path.join(P3, "member3_B_80_20_mse.csv"),
        "B": os.path.join(P3, "member3_B2_expanded_80_20_mse.csv"),
    }
    p3_cv = {
        "A": os.path.join(P3, "member3_results_B_10fold.csv"),
        "B": os.path.join(P3, "member3_B2_expanded_10fold.csv"),
    }
    for tag in ("A", "B"):
        f80, fcv = p3_80[tag], p3_cv[tag]
        if not (os.path.exists(f80) and os.path.exists(fcv)):
            print("  missing:", f80 if not os.path.exists(f80) else fcv)
            continue
        d80 = _read(f80)
        dcv = _read(fcv)
        for _, r in d80.iterrows():
            # B.py names them B1_RF / B2_GBR ; B2_expanded.py names them B2_RF_expanded ...
            name = r["model"]
            is_rf = "RF" in name
            short = "Random Forest" if is_rf else "Gradient Boosting"
            cv_row = dcv[dcv["model"].str.contains("RF" if is_rf else "GBR", regex=True)]
            cv = cv_row.iloc[0] if len(cv_row) else None
            rows.append({
                "cell": f"{tag}2",
                "matrix": "A_250" if tag == "A" else "B_342",
                "family": "tree",
                "model": short,
                "source": "Part 3",
                "MSE_80_20": r["Test_MSE_log"],
                "RMSE_80_20": np.nan,
                "R2_80_20": np.nan,
                "alpha": np.nan,
                "sparsity": np.nan,
                "CV_RMSE_log": cv["RMSE_log"] if cv is not None else np.nan,
                "CV_R2": cv["R2"] if cv is not None else np.nan,
            })

    # ---------------- Part 4 : the single source of CV truth ----------------
    f = os.path.join(OUT, "unified_factorial.csv")
    if os.path.exists(f):
        d = _read(f)
        for _, r in d.iterrows():
            tag = "A" if r["matrix"].startswith("A") else "B"
            cell = f"{tag}1" if r["model"] in ("OLS", "Ridge", "Lasso") else f"{tag}2"
            for row in rows:
                if row["cell"] == cell and row["model"] == r["model"]:
                    row["CV_RMSE_log"] = r["CV_RMSE_log"]
                    row["CV_R2"] = r["CV_R2"]
                    row["n_features"] = r["n_features"]
                    break

    df = pd.DataFrame(rows)
    order = ["OLS", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"]
    df["_o"] = df["model"].apply(lambda m: order.index(m) if m in order else 99)
    df = df.sort_values(["cell", "_o"]).drop(columns="_o")
    return df


def build_deltas(df):
    out = []
    for model in df["model"].unique():
        a = df[(df.model == model) & (df.cell.str.startswith("A"))]
        b = df[(df.model == model) & (df.cell.str.startswith("B"))]
        if not len(a) or not len(b):
            continue
        a, b = a.iloc[0], b.iloc[0]
        out.append({
            "model": model,
            "family": a["family"],
            "MSE_A": a["MSE_80_20"], "MSE_B": b["MSE_80_20"],
            "d_MSE": b["MSE_80_20"] - a["MSE_80_20"],
            "d_MSE_pct": (b["MSE_80_20"] - a["MSE_80_20"]) / a["MSE_80_20"] * 100,
            "CV_R2_A": a.get("CV_R2", np.nan), "CV_R2_B": b.get("CV_R2", np.nan),
            "d_CV_R2": b.get("CV_R2", np.nan) - a.get("CV_R2", np.nan),
        })
    return pd.DataFrame(out)


def plot(df, dd):
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import config as C
    C.apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))

    models = ["OLS", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"]
    x = np.arange(len(models))
    w = 0.38
    colors = {"linear": "#1F4E79", "tree": "#C55A11"}

    ax = axes[0]
    for i, (mat, off, lbl) in enumerate([
            ("A_250", -w / 2, "A · 250 cols (Part 1 engineered)"),
            ("B_342", +w / 2, "B · 342 cols (+ poly / spline)")]):
        vals = [df[(df.model == m) & (df.matrix == mat)]["MSE_80_20"].iloc[0]
                if len(df[(df.model == m) & (df.matrix == mat)]) else np.nan
                for m in models]
        bars = ax.bar(x + off, vals, w, label=lbl,
                      color=["#1F4E79", "#3A6EA5", "#5B8FC7"] if i == 0
                      else ["#C55A11", "#D9793E", "#E39A6B"])
        for r in bars:
            if np.isfinite(r.get_height()):
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.0008,
                        f"{r.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=16, ha="right")
    ax.set_ylabel("Test MSE  (log scale, lower is better)")
    ax.set_title("80/20 holdout · identical split in all 8 cells")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 0.068)

    ax2 = axes[1]
    for mat, off, lbl in [("A_250", -w / 2, "A · 250 cols"),
                          ("B_342", +w / 2, "B · 342 cols")]:
        vals = [df[(df.model == m) & (df.matrix == mat)]["CV_R2"].iloc[0]
                if len(df[(df.model == m) & (df.matrix == mat)]) else np.nan
                for m in models]
        bars = ax2.bar(x + off, vals, w, label=lbl,
                       color=["#1F4E79", "#3A6EA5", "#5B8FC7"] if off < 0
                       else ["#C55A11", "#D9793E", "#E39A6B"])
        for r in bars:
            if np.isfinite(r.get_height()):
                ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01,
                         f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    ax2.axhline(0, color="#333333", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=16, ha="right")
    ax2.set_ylabel("CV-10 R2  (higher is better)")
    ax2.set_title("10-fold cross-validation · the protocol that decides")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_ylim(-0.55, 1.05)

    fig.suptitle("Unified 2x2 factorial benchmark:  model family  x  feature matrix",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(FIG, "fig19_master_8cell.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved:", p)


def main():
    df = collect()
    df.to_csv(os.path.join(OUT, "master_8cell_table.csv"), index=False)
    dd = build_deltas(df)
    dd.to_csv(os.path.join(OUT, "master_8cell_deltas.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=" * 100)
    print("MASTER 8-CELL TABLE")
    print("=" * 100)
    print(df[["cell", "matrix", "family", "model", "source", "n_features",
              "MSE_80_20", "R2_80_20", "CV_RMSE_log", "CV_R2"]].to_string(index=False))
    print()
    print("=" * 100)
    print("MARGINAL EFFECT OF AUTOMATIC NON-LINEAR EXPANSION")
    print("=" * 100)
    print(dd.to_string(index=False))
    plot(df, dd)


if __name__ == "__main__":
    main()
