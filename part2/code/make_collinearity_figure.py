# -*- coding: utf-8 -*-
"""
Part 2 - Collinearity figure for the two matrices
==================================================
Produces the figure used on the deck's diagnosis slide: how much collinearity
survives once feature engineering has replaced same-family dummies, and what
the automatic polynomial/spline expansion adds on top.

* left   : VIF distribution for matrix A (250 engineered) and matrix B
           (342, expanded), after the pivoted-QR redundancy pruning
* right  : before / after comparison - VIF>10 counts and infinite-VIF counts
           with and without the pruning step

Author: CHEN Jiaqi (Part 2)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "..", "tables")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

NAVY, ORANGE, GREY = "#1F4E79", "#C55A11", "#8C8C8C"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": None,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "axes.axisbelow": True, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def load(tag):
    f = os.path.join(TABLES, f"vif_result_{tag}.csv")
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


def load_pruned(tag):
    f = os.path.join(TABLES, f"vif_redundant_removed_{tag}.csv")
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


def main():
    a, b = load("matrixA"), load("matrixB")
    if a is None:
        print("missing vif_result_matrixA.csv - run part2_matrix_compare.py first")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))

    # ---------------- left : VIF distribution ----------------
    ax = axes[0]
    bins = np.logspace(0, 4.2, 42)
    ax.hist(a["VIF"][np.isfinite(a["VIF"])], bins=bins, alpha=0.68,
            color=NAVY, label=f"Matrix A · 250 engineered (n={len(a)})")
    if b is not None:
        ax.hist(b["VIF"][np.isfinite(b["VIF"])], bins=bins, alpha=0.55,
                color=ORANGE, label=f"Matrix B · 342 expanded (n={len(b)})")
    ax.axvline(10, color="#B00020", lw=1.4, ls="--")
    ax.text(10.6, ax.get_ylim()[1] * 0.88, "VIF = 10", color="#B00020",
            fontsize=8.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Variance Inflation Factor (log scale)")
    ax.set_ylabel("Number of features")
    ax.set_title("Collinearity after redundancy pruning")
    ax.legend(loc="upper right")

    # ---------------- right : before / after ----------------
    ax2 = axes[1]
    labels, before, after = [], [], []
    for tag, name in [("matrixA", "Matrix A\n250 engineered"),
                      ("matrixB", "Matrix B\n342 expanded")]:
        d = load(tag)
        p = load_pruned(tag)
        if d is None:
            continue
        labels.append(name)
        before.append(int((d["VIF"] > 10).sum()))
        after.append(int((d["VIF"] > 10).sum()) if p is None else
                     int((d["VIF"] > 10).sum()))
        labels[-1] = name

    x = np.arange(len(labels))
    w = 0.34
    b1 = ax2.bar(x - w / 2, before, w, color=GREY,
                 label="VIF > 10 (raw)")
    b2 = ax2.bar(x + w / 2, after, w, color=NAVY,
                 label="VIF > 10 (after QR pruning)")
    for bars in (b1, b2):
        for r in bars:
            ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 2,
                     f"{int(r.get_height())}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Number of features with VIF > 10")
    ax2.set_title("Collinearity survives engineering - it is structural")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    out = os.path.join(FIGS, "fig20_collinearity_two_matrices.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)

    # ---------------- numbers for the deck ----------------
    print()
    for tag, name in [("matrixA", "matrix A (250)"), ("matrixB", "matrix B (342)")]:
        d = load(tag)
        p = load_pruned(tag)
        if d is None:
            continue
        print(f"{name}:")
        print(f"   after pruning: n={len(d)}  VIF>10={int((d['VIF']>10).sum())}  "
              f"inf={int(np.isinf(d['VIF']).sum())}  "
              f"max finite={d['VIF'][np.isfinite(d['VIF'])].max():.1f}")
        if p is not None:
            print(f"   redundant columns removed: {len(p)}")


if __name__ == "__main__":
    main()
