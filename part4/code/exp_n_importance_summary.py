# -*- coding: utf-8 -*-
"""
Part 4 - Figure N: the feature-importance summary table
=========================================================
The importance numbers already exist (``permutation_importance.csv``,
``shap_global_importance.csv``, ``linear_coefficients.csv``); what was missing
was a single place where the three of them can be read side by side for the
same features, so that the reader can see *which* conclusions are shared and
*which* depend on the metric.

This script builds that table:

    rows    : the features that matter, ranked by permutation importance
    columns : the rank the feature receives under each of the four measures
              (permutation importance, TreeSHAP, Linear SHAP, |standardised
              Lasso coefficient|) plus the four members' own importance lists

It also computes the rank correlation between the measures, which is the
quantitative version of "the head agrees, the tail does not".

Outputs
-------
* ``output/importance_summary_table.csv``
* ``output/importance_rank_agreement.csv``
* ``figures/fig28_importance_summary_table.png``

Author: Yifei Lv (Part 4)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C

TOP_N = 15


def _rank(series: pd.Series) -> pd.Series:
    """Dense rank, 1 = largest."""
    return series.rank(ascending=False, method="min").astype(int)


def build_table():
    perm = pd.read_csv(os.path.join(C.OUT_DIR, "permutation_importance.csv"))
    shap = pd.read_csv(os.path.join(C.OUT_DIR, "shap_global_importance.csv"))
    coef = pd.read_csv(os.path.join(C.OUT_DIR, "linear_coefficients.csv"))

    # --- four measures, each giving a rank per feature -------------------
    m_perm = (perm[perm.model == "Gradient Boosting"]
              .set_index("feature")["perm_importance_mean"])
    m_shap_t = (shap[shap.model == "GBR (TreeSHAP)"]
                .set_index("feature")["mean_abs_shap"])
    m_shap_l = (shap[shap.model == "Ridge (Linear SHAP)"]
                .set_index("feature")["mean_abs_shap"])
    m_coef = (coef[(coef.model == "Lasso") & (coef.coef != 0)]
              .set_index("feature")["abs_coef"])

    feats = [f for f in m_perm.sort_values(ascending=False).index[:TOP_N]]

    rows = []
    for f in feats:
        rows.append({
            "feature": f,
            "perm_imp": float(m_perm.get(f, np.nan)),
            "rank_perm": int(_rank(m_perm).get(f, np.nan)),
            "rank_shap_tree": int(_rank(m_shap_t).get(f, np.nan)),
            "rank_shap_linear": int(_rank(m_shap_l).get(f, np.nan)),
            "rank_lasso_coef": (int(_rank(m_coef).get(f, np.nan))
                                if f in m_coef.index else np.nan),
            "lasso_coef": float(m_coef.get(f, np.nan)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "importance_summary_table.csv"),
              index=False)

    # --- how much do the measures agree? ---------------------------------
    from scipy.stats import spearmanr

    cols = ["rank_perm", "rank_shap_tree", "rank_shap_linear", "rank_lasso_coef"]
    agree = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = df[cols[i]], df[cols[j]]
            ok = a.notna() & b.notna()
            rho = (spearmanr(a[ok], b[ok]).statistic
                   if ok.sum() > 2 else np.nan)
            agree.append({
                "measure_1": cols[i], "measure_2": cols[j],
                "spearman_rho": float(rho) if not np.isnan(rho) else np.nan,
                "n_common": int(ok.sum()),
                "mean_abs_rank_gap": float((a[ok] - b[ok]).abs().mean()),
            })
    adf = pd.DataFrame(agree)
    adf.to_csv(os.path.join(C.OUT_DIR, "importance_rank_agreement.csv"),
               index=False)
    return df, adf


def plot(df, adf):
    import matplotlib.pyplot as plt

    navy, orange, green, purple = "#1F4E79", "#C55A11", "#2E7D32", "#7B4EA3"

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.4),
                             gridspec_kw={"width_ratios": [1.62, 1]})

    # ================= left : the summary table =================
    ax = axes[0]
    ax.axis("off")

    headers = ["Feature", "Perm.\nimp.", "Rank:\nperm", "Rank:\nTreeSHAP",
               "Rank:\nLinSHAP", "Rank:\nLasso coef"]
    cell = []
    for _, r in df.iterrows():
        cell.append([
            r["feature"],
            f"{r['perm_imp']:.4f}",
            str(int(r["rank_perm"])) if pd.notna(r["rank_perm"]) else "-",
            str(int(r["rank_shap_tree"])) if pd.notna(r["rank_shap_tree"]) else "-",
            str(int(r["rank_shap_linear"])) if pd.notna(r["rank_shap_linear"]) else "-",
            str(int(r["rank_lasso_coef"])) if pd.notna(r["rank_lasso_coef"]) else "-",
        ])

    tbl = ax.table(cellText=cell, colLabels=headers, loc="center",
                   cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.86])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.85)
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor("#D0D0D0")
        if r == 0:
            cellobj.set_facecolor(navy)
            cellobj.set_text_props(color="white", fontweight="bold", fontsize=9)
        else:
            cellobj.set_facecolor("#F4F8FB" if r % 2 else "white")
            if c == 0:
                cellobj.set_text_props(fontweight="bold", ha="left")
            if c == 2:
                cellobj.set_text_props(fontweight="bold", color=orange)

    ax.set_title("The same 15 features ranked by four different importance "
                 "measures — the head agrees, the tail does not",
                 loc="left", fontsize=12.5, fontweight="bold", pad=16)

    # ================= right : rank agreement =================
    ax2 = axes[1]
    labels, rhos = [], []
    pretty = {
        "rank_perm": "perm vs perm", "rank_shap_tree": "",
        "rank_shap_linear": "", "rank_lasso_coef": "",
    }
    names = {
        "rank_perm": "Permutation",
        "rank_shap_tree": "TreeSHAP",
        "rank_shap_linear": "Linear SHAP",
        "rank_lasso_coef": "Lasso coef",
    }
    for _, r in adf.iterrows():
        a, b = r["measure_1"], r["measure_2"]
        labels.append(f"{names[a]}\nvs {names[b]}")
        rhos.append(r["spearman_rho"])

    y = np.arange(len(labels))
    colors = [green if v >= 0.8 else orange if v >= 0.5 else "#B00020"
              for v in rhos]
    bars = ax2.barh(y, rhos, color=colors, height=0.62)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    for b, v in zip(bars, rhos):
        ax2.text(v - 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                 va="center", ha="right", fontsize=9.5, fontweight="bold")
    ax2.axvline(0.8, color="#666666", ls="--", lw=1.1)
    ax2.text(0.81, -0.75, "0.8", fontsize=8, color="#666666")
    ax2.set_xlim(-0.05, 1.02)
    ax2.set_xlabel("Spearman rank correlation between the two measures")
    ax2.set_title("Do the measures agree with each other?",
                  loc="left", fontsize=12.5, fontweight="bold", pad=16)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("Feature importance: one set of features, four ways to rank them",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(C.FIG_DIR, "fig28_importance_summary_table.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved:", p)


def run():
    df, adf = build_table()
    print(df.to_string(index=False))
    print()
    print(adf.to_string(index=False))
    plot(df, adf)


if __name__ == "__main__":
    run()
