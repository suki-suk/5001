"""EDA for the House Prices report: figures, tables, supplementary charts, gallery.

    python code/02_eda.py

Produces, deterministically and in one pass:
  figures/fig*.png              13 figures used in the report
  figures/eda_*.png             3 supplementary charts (missingness matrix, pairplot,
                                price by dwelling type) + a one-page contact sheet
  figures/eda_gallery.html      browser gallery with a caption per figure
  tables/*.csv                  every table quoted in the report
  reports/eda_stats.json        numbers embedded in the report prose

The script is idempotent: rerunning overwrites every output.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import house_prices_pipeline as hp

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE, ORANGE, GREEN = "#2E6F9E", "#E07B39", "#3C8D6B"


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def report_charts() -> dict:
    """The 13 figures and all tables that the Word report cites."""
    train, test = hp.load_raw()
    stats = {**hp.audit(train, test)}
    y = train.SalePrice
    logy = np.log(y)

    # ---------------------------------------------------------------- fig 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(y, bins=50, color=BLUE, edgecolor="white")
    axes[0].axvline(y.mean(), color=ORANGE, ls="--", lw=1.5, label=f"mean ${y.mean():,.0f}")
    axes[0].axvline(y.median(), color=GREEN, ls=":", lw=1.5, label=f"median ${y.median():,.0f}")
    axes[0].set(title="SalePrice is right-skewed", xlabel="SalePrice (USD)", ylabel="Number of sales")
    axes[0].legend(frameon=False)
    axes[1].hist(logy, bins=50, color=BLUE, edgecolor="white")
    axes[1].set(title=f"log(SalePrice) is near-normal (skew {logy.skew():.2f})",
                xlabel="log(SalePrice)", ylabel="Number of sales")
    fig.suptitle(f"Target distribution (n = {len(y):,} sales)", y=1.02)
    fig.tight_layout()
    save(fig, "fig01_target_distribution.png")

    # ---------------------------------------------------------------- fig 2
    miss = hp.missing_table(train, test)
    show = miss.sort_values("pct_train", ascending=False).head(16).iloc[::-1]
    colors = [ORANGE if "Structural" in r else BLUE for r in show.interpretation]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(show.column, show.pct_train, color=colors)
    for i, (pct, n) in enumerate(zip(show.pct_train, show.missing_train)):
        ax.text(pct + 0.8, i, f"{pct:.1f}%  (n={n:,})", va="center", fontsize=8.5)
    ax.set(xlabel="Share of training rows missing (%)", xlim=(0, 112))
    ax.set_title("Missingness is mostly structural, not missing data", pad=14)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label='Structural (NA = "no basement/garage/...")'),
                       Patch(color=BLUE, label="True missing value")],
              frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig02_missing_values.png")

    # ---------------------------------------------------------------- fig 3
    num = train.select_dtypes("number").drop(columns=["Id"])
    corr = num.corr()["SalePrice"].drop("SalePrice")
    top = corr.abs().sort_values(ascending=False).head(12).index.tolist()
    cm = num[top + ["SalePrice"]].corr()
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=.6, cbar_kws={"shrink": .72}, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation among the strongest raw numeric predictors")
    fig.tight_layout()
    save(fig, "fig03_correlation.png")

    # ---------------------------------------------------------------- fig 4
    q = train.groupby("OverallQual").SalePrice.agg(["median", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sns.boxplot(data=train, x="OverallQual", y="SalePrice", color=BLUE, fliersize=2.5, ax=ax)
    ax2 = ax.twinx()
    ax2.plot(q.OverallQual, q["median"], color=ORANGE, marker="o", lw=1.6, label="median")
    ax2.set_ylabel("Median SalePrice (USD)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.set_ylim(0, 900000)
    ax.set(xlabel="Overall material and finish quality (1 = very poor, 10 = excellent)",
           ylabel="SalePrice (USD)", ylim=(0, 800000))
    ax.set_title("OverallQual is the single strongest raw predictor (r = %.2f)" % corr["OverallQual"])
    fig.tight_layout()
    save(fig, "fig04_quality_price.png")

    # ---------------------------------------------------------------- fig 5
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(train.GrLivArea, train.SalePrice, c=train.YearBuilt, cmap="viridis",
                    s=22, alpha=.75, edgecolor="none")
    out = train[(train.GrLivArea > 4000) & (train.SalePrice < 300000)]
    ax.scatter(out.GrLivArea, out.SalePrice, s=110, facecolor="none", edgecolor="red",
               lw=1.6, label="partial-sale outliers (Id 524, 1299)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Year built")
    ax.set(xlabel="Above-grade living area (sq ft)", ylabel="SalePrice (USD)")
    ax.set_title("Living area vs price: two partial sales break the trend")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig05_size_price.png")

    # ---------------------------------------------------------------- fig 5b
    # dedicated view of the four largest houses, for the outlier discussion
    big = train[train.GrLivArea > 2500]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(train.GrLivArea, train.SalePrice, s=16, alpha=.3, color=BLUE, label="other sales")
    ax.scatter(big.GrLivArea, big.SalePrice, s=40, color=BLUE, edgecolor="white", zorder=3,
               label="houses above 2,500 sq ft")
    for _, row in big[big.SalePrice < 300000].iterrows():
        ax.annotate(f"Id {row.Id}: ${row.SalePrice/1000:.0f}k", (row.GrLivArea, row.SalePrice),
                    textcoords="offset points", xytext=(-10, 18), ha="right", fontsize=8.5,
                    color="red")
    ax.set(xlabel="Above-grade living area (sq ft)", ylabel="SalePrice (USD)",
           xlim=(2200, 6000), ylim=(120000, 820000))
    ax.set_title("The two partial sales are the only large houses priced below $300k")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig13_outliers.png")

    # ---------------------------------------------------------------- fig 6
    nb = (train.groupby("Neighborhood").SalePrice
          .agg(median="median", n="size").sort_values("median"))
    fig, ax = plt.subplots(figsize=(9, 7.6))
    bars = ax.barh(nb.index, nb["median"], color=BLUE)
    ax.barh(nb.index, nb["median"], color=[ORANGE if n < 30 else BLUE for n in nb.n])
    for i, (m, n) in enumerate(zip(nb["median"], nb.n)):
        ax.text(m + 2500, i, f"${m/1000:.0f}k  (n={n})", va="center", fontsize=8)
    ax.set(xlabel="Median SalePrice (USD)", xlim=(0, nb["median"].max() * 1.25))
    ax.set_title("Location: median price by neighbourhood (orange = fewer than 30 sales)")
    fig.tight_layout()
    save(fig, "fig06_neighborhood.png")

    # ---------------------------------------------------------------- fig 7
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    order = ["Po", "Fa", "TA", "Gd", "Ex"]
    sns.boxplot(data=train, x="KitchenQual", y="SalePrice", order=order, color=BLUE, fliersize=2, ax=axes[0])
    sns.boxplot(data=train, x="ExterQual", y="SalePrice", order=order, color=GREEN, fliersize=2, ax=axes[1])
    for ax, col in zip(axes, ["KitchenQual", "ExterQual"]):
        ax.set(xlabel=f"{col} (Po < Fa < TA < Gd < Ex)", ylim=(0, 800000))
        ax.tick_params(axis="x", rotation=0)
    axes[0].set_ylabel("SalePrice (USD)")
    fig.suptitle("Kitchen and exterior quality separate the price distribution cleanly", y=1.02)
    fig.tight_layout()
    save(fig, "fig07_quality_categorical.png")

    # ---------------------------------------------------------------- fig 8
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sc_order = train.groupby("SaleCondition").SalePrice.median().sort_values().index
    sns.boxplot(data=train, x="SaleCondition", y="SalePrice", order=sc_order,
                color=BLUE, fliersize=2, ax=axes[0])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(xlabel="SaleCondition", ylabel="SalePrice (USD)", ylim=(0, 800000))
    st_order = train.groupby("SaleType").SalePrice.median().sort_values().index
    sns.boxplot(data=train, x="SaleType", y="SalePrice", order=st_order,
                color=ORANGE, fliersize=2, ax=axes[1])
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(xlabel="SaleType", ylabel="", ylim=(0, 800000))
    fig.suptitle("Sale circumstances shift prices: partial (new) sales sit low despite large houses",
                 y=1.03)
    fig.tight_layout()
    save(fig, "fig08_sale_circumstances.png")

    # ---------------------------------------------------------------- fig 9
    train["Decade"] = (train.YearBuilt // 10 * 10).astype(int)
    dec = train.groupby("Decade").SalePrice.median()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(dec.index, dec.values, marker="o", color=BLUE, lw=1.8)
    ax.fill_between(dec.index, dec.values, color=BLUE, alpha=.12)
    ax.set(xlabel="Decade built", ylabel="Median SalePrice (USD)",
           title="Newer houses sell for more, but the relationship is not linear")
    ax.set_xticks(range(1870, 2011, 20))
    fig.tight_layout()
    save(fig, "fig09_year_built.png")

    # --------------------------------------------------------------- fig 10
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    m = train.groupby("MoSold").SalePrice.agg(["median", "size"])
    axes[0].bar(m.index, m["size"], color=BLUE, alpha=.8)
    axes[0].set(xlabel="Month sold", ylabel="Number of sales", xticks=range(1, 13),
                title="Sales volume peaks in May-July; monthly medians are noisy")
    ax2 = axes[0].twinx()
    ax2.plot(m.index, m["median"] / 1000, color=ORANGE, marker="o", lw=1.6)
    ax2.set_ylabel("Median price ($k)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    yr = train.groupby("YrSold").SalePrice.agg(["median", "size"])
    axes[1].bar(yr.index.astype(str), yr["size"], color=BLUE, alpha=.8)
    axes[1].set(xlabel="Year sold", ylabel="Number of sales",
                title="Volume is stable across years; median price falls in 2010")
    ax3 = axes[1].twinx()
    ax3.plot(yr.index.astype(str), yr["median"] / 1000, color=ORANGE, marker="o", lw=1.6)
    ax3.set_ylabel("Median price ($k)", color=ORANGE)
    ax3.tick_params(axis="y", colors=ORANGE)
    fig.suptitle("Time effects are small but real", y=1.03)
    fig.tight_layout()
    save(fig, "fig10_seasonality.png")

    # --------------------------------------------------------------- fig 11
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(train.TotalBsmtSF, train["1stFlrSF"], s=14, alpha=.5, color=BLUE)
    axes[0].plot([0, 3000], [0, 3000], ls="--", color="grey", lw=1)
    axes[0].set(xlabel="Total basement area (sq ft)", ylabel="First floor area (sq ft)",
                title="TotalBsmtSF vs 1stFlrSF (r = %.2f)" % train.TotalBsmtSF.corr(train["1stFlrSF"]))
    axes[1].scatter(train.GarageCars + np.random.default_rng(0).normal(0, .06, len(train)),
                    train.GarageArea, s=14, alpha=.45, color=GREEN)
    axes[1].set(xlabel="Garage capacity (cars, jittered)", ylabel="Garage area (sq ft)",
                title="GarageCars vs GarageArea (r = %.2f)" % train.GarageCars.corr(train.GarageArea))
    fig.suptitle("Redundant raw measurements: near-duplicate information", y=1.02)
    fig.tight_layout()
    save(fig, "fig11_multicollinearity.png")

    # --------------------------------------------------------------- fig 12
    # build the engineered frame once (clean -> ordinal -> rare -> engineer)
    c_tr, c_te, _ = hp.clean(train, test)
    o_tr, o_te = hp.ordinal_encode(c_tr.drop(columns=["SalePrice"]), c_te)
    r_tr, _, _ = hp.collapse_rare(o_tr, o_te)
    f_tr = hp.engineer(r_tr)
    f_tr["SalePrice"] = c_tr.SalePrice.values
    cand = f_tr.select_dtypes("number").drop(columns=["Id"])
    corrs = cand.corr()["SalePrice"].drop("SalePrice").sort_values()
    new_feats = [c for c in f_tr.columns if c not in r_tr.columns]
    top_new = corrs[corrs.index.isin(new_feats)].abs().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(top_new.index[::-1], corrs[top_new.index[::-1]], color=ORANGE)
    for i, v in enumerate(corrs[top_new.index[::-1]]):
        ax.text(v + .01 if v >= 0 else v - .01, i, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Pearson correlation with SalePrice", xlim=(-0.1, 0.95))
    ax.set_title("The engineered features carry the signal of the raw quantities they summarise")
    fig.tight_layout()
    save(fig, "fig12_engineered_corr.png")

    # -------------------------------------------------------------- tables
    pd.DataFrame({
        "statistic": ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew", "skew_log"],
        "value": [len(y), y.mean(), y.std(), y.min(), y.quantile(.25), y.median(),
                  y.quantile(.75), y.max(), y.skew(), logy.skew()],
    }).round(2).to_csv(TAB / "target_summary.csv", index=False)

    corr.rename("corr_with_SalePrice").sort_values(key=abs, ascending=False).round(3).to_csv(
        TAB / "numeric_correlations.csv")
    nb.assign(nb_share=(nb.n / len(train) * 100).round(1)).round(1).to_csv(
        TAB / "neighborhood_price.csv")
    q.round(1).to_csv(TAB / "quality_price.csv", index=False)
    (train.groupby("SaleCondition").SalePrice.agg(["size", "median", "mean"])
     .sort_values("median").round(0).to_csv(TAB / "sale_condition.csv"))

    # ------------------------------------------------- variable typology
    families = {
        "Location and lot": ["MSZoning", "LotFrontage", "LotArea", "Street", "Alley",
                             "LotShape", "LandContour", "Utilities", "LotConfig",
                             "LandSlope", "Neighborhood", "Condition1", "Condition2"],
        "Dwelling type and exterior": ["MSSubClass", "BldgType", "HouseStyle", "RoofStyle",
                                       "RoofMatl", "Exterior1st", "Exterior2nd",
                                       "MasVnrType", "MasVnrArea", "Foundation"],
        "Quality and condition": ["OverallQual", "OverallCond", "ExterQual", "ExterCond",
                                  "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
                                  "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
                                  "Functional"],
        "Size and layout": ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
                            "GrLivArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                            "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
                            "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces"],
        "Basement": ["BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
        "Garage": ["GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea"],
        "Outdoor and extras": ["PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                               "3SsnPorch", "ScreenPorch", "PoolArea", "Fence",
                               "MiscFeature", "MiscVal"],
        "Time and sale": ["YearBuilt", "YearRemodAdd", "MoSold", "YrSold", "SaleType",
                          "SaleCondition"],
        "Systems": ["Heating", "CentralAir", "Electrical"],
    }
    scale_of = {}
    for c in ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
              "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
              "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
              "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal"]:
        scale_of[c] = "Continuous"
    for c in ["OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFullBath",
              "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
              "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageYrBlt"]:
        scale_of[c] = "Discrete"
    for c in hp.ORDINAL_MAPS:
        scale_of[c] = "Ordinal"
    for c in hp.NOMINAL:
        scale_of.setdefault(c, "Nominal")
    scale_of["Electrical"] = "Nominal"

    rows = []
    assigned = set()
    for fam, cols in families.items():
        for c in cols:
            rows.append({"feature": c, "family": fam, "scale": scale_of.get(c, "?"),
                         "n_unique": int(train[c].nunique(dropna=False)),
                         "missing_train_pct": round(train[c].isna().mean() * 100, 2),
                         "missing_test_pct": round(test[c].isna().mean() * 100, 2)})
            assigned.add(c)
    missing = [c for c in train.columns
               if c not in assigned and c not in ("Id", "SalePrice", "Decade")]
    assert not missing, missing
    typ = pd.DataFrame(rows)
    typ.to_csv(TAB / "variable_typology.csv", index=False)
    typ.groupby(["family", "scale"]).size().unstack(fill_value=0).to_csv(TAB / "typology_summary.csv")
    print(typ.scale.value_counts().to_dict(), "| features:", len(typ))

    stats.update({
        "corr_top": corr.abs().sort_values(ascending=False).head(8).round(3).to_dict(),
        "neigh_max": {"name": nb.index[-1], "median": int(nb["median"].iloc[-1]), "n": int(nb.n.iloc[-1])},
        "neigh_min": {"name": nb.index[0], "median": int(nb["median"].iloc[0]), "n": int(nb.n.iloc[0])},
        "neigh_ratio": round(float(nb["median"].iloc[-1] / nb["median"].iloc[0]), 2),
        "quality_median_low": int(q.loc[q.OverallQual == q.OverallQual.min(), "median"].iat[0]),
        "quality_median_high": int(q.loc[q.OverallQual == q.OverallQual.max(), "median"].iat[0]),
        "mo_sold_peak": int(m["size"].idxmax()),
        "top_new_feature": top_new.index[0],
        "top_new_feature_corr": round(float(corrs[top_new.index[0]]), 3),
    })
    with open(ROOT / "reports" / "eda_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(json.dumps({k: stats[k] for k in ["neigh_max", "neigh_min", "neigh_ratio",
                                            "quality_median_low", "quality_median_high",
                                            "top_new_feature", "top_new_feature_corr"]}, indent=2))
    return stats


# ---------------------------------------------------------------------------
# Supplementary visuals that support the presentation but are not in the report
# ---------------------------------------------------------------------------

def extra_charts() -> None:
    """Missingness matrix, price-tier pairplot and price by dwelling type."""
    train, test = hp.load_raw()

    # --- A. missingness matrix -------------------------------------------
    miss_cols = train.columns[train.isna().any()].tolist()
    binary = train[miss_cols].isna().astype(int)
    order = binary.apply(lambda r: "".join(map(str, r.values)), axis=1).sort_values().index
    binary = binary.loc[order]

    # 0 = recorded, 1 = structural absence, 2 = genuinely missing value
    true_missing = {"LotFrontage", "Electrical"}
    mat = np.zeros(binary.shape, dtype=int)
    for j, c in enumerate(miss_cols):
        mat[:, j] = np.where(binary[c].values == 1, 2 if c in true_missing else 1, 0)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(["#F7F9FB", "#7FA8C9", "#E07B39"])
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
              extent=[-0.5, len(miss_cols) - 0.5, len(mat) - 0.5, -0.5])
    ax.set_xticks(range(len(miss_cols)))
    ax.set_xticklabels(miss_cols, rotation=90, fontsize=7.5)
    ax.set_yticks([0, 500, 1000, 1459])
    ax.set_yticklabels(["0", "500", "1,000", "1,460"])
    ax.set(ylabel="Training rows (sorted by pattern)", xlabel="")
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Missingness arrives in blocks: whole rows lack the same features", pad=32)
    ax.legend(handles=[
        Patch(facecolor="#F7F9FB", edgecolor="#D0D7DE", label="recorded"),
        Patch(facecolor="#7FA8C9", label="structural absence (no such feature)"),
        Patch(facecolor="#E07B39", label="genuinely missing value"),
    ], loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "eda_missing_matrix.png")

    # --- B. pairplot by price tier ---------------------------------------
    d = train[["SalePrice", "OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt"]].copy()
    d["Price tier"] = pd.qcut(d.SalePrice, 3, labels=["Low (bottom third)",
                                                      "Middle", "High (top third)"])
    g = sns.pairplot(d.drop(columns="SalePrice"), hue="Price tier", corner=True,
                     height=1.85, plot_kws={"s": 9, "alpha": .55, "linewidth": 0},
                     diag_kws={"fill": True, "alpha": .35})
    g.figure.suptitle("Quality, size and age separate the price tiers jointly",
                      y=1.01, fontweight="bold")
    g.figure.savefig(FIG / "eda_pairplot.png", dpi=170, bbox_inches="tight")
    plt.close(g.figure)
    print("wrote eda_pairplot.png")

    # --- C. price by dwelling type ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    bld = train.groupby("BldgType").SalePrice.median().sort_values().index
    sns.boxplot(data=train, x="BldgType", y="SalePrice", order=bld, color=BLUE,
                fliersize=2, ax=axes[0])
    axes[0].set(xlabel="Building type", ylabel="SalePrice (USD)", ylim=(0, 800000))
    axes[0].set_title("Single-family homes dominate both count and price")

    sty = train.groupby("HouseStyle").SalePrice.median().sort_values().index
    sns.boxplot(data=train, x="HouseStyle", y="SalePrice", order=sty, color=GREEN,
                fliersize=2, ax=axes[1])
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(xlabel="House style", ylabel="", ylim=(0, 800000))
    axes[1].set_title("Two-storey and split-level homes sell above bungalows")
    fig.tight_layout()
    save(fig, "eda_types.png")


GALLERY = [
    ("fig01_target_distribution.png", "Target distribution",
     "SalePrice is right-skewed (1.88); log(SalePrice) is near-normal (0.12)."),
    ("fig02_missing_values.png", "Missing values by column",
     "Most blanks are structural absence, not lost data."),
    ("eda_missing_matrix.png", "Missingness matrix (supplementary)",
     "Blue bands = structural absence; orange = genuinely missing values."),
    ("fig03_correlation.png", "Correlation among top predictors",
     "Quality and size lead; the size block is internally redundant."),
    ("fig04_quality_price.png", "Price by overall quality",
     "A monotone ladder from $50k to $432k median."),
    ("fig07_quality_categorical.png", "Price by kitchen / exterior quality",
     "Ordered labels behave as an ordinal scale."),
    ("fig05_size_price.png", "Living area vs price",
     "Positive but heteroscedastic; the two partial sales break the trend."),
    ("fig13_outliers.png", "Outlier close-up",
     "Only Ids 524 and 1299 pair >4,000 sq ft with a price below $300k."),
    ("fig06_neighborhood.png", "Median price by neighbourhood",
     "Location alone spans a factor of 3.6."),
    ("fig09_year_built.png", "Price by decade built",
     "Newer stock sells higher; the effect flattens before 1980."),
    ("fig10_seasonality.png", "Volume and price over time",
     "Sales peak in May-July; 2010 medians dip."),
    ("fig08_sale_circumstances.png", "Price by sale condition",
     "Partial and abnormal sales are systematically cheaper."),
    ("fig11_multicollinearity.png", "Near-duplicate measurements",
     "Basement vs 1st floor 0.82; garage cars vs area 0.88."),
    ("fig12_engineered_corr.png", "Engineered features vs price",
     "QualxTotalSF (0.86) beats raw TotalSF (0.78)."),
    ("eda_pairplot.png", "Pairplot by price tier (supplementary)",
     "The tiers separate on quality and size together, not on any single variable."),
    ("eda_types.png", "Price by dwelling type (supplementary)",
     "Single-family and two-storey homes dominate the upper range."),
]


def gallery() -> None:
    """Browser gallery: one card per figure, captions included."""
    cards = "\n".join(
        f"""    <figure class="card">
      <img src="{name}" alt="{title}">
      <figcaption><strong>{title}</strong><br>{cap}</figcaption>
    </figure>"""
        for name, title, cap in GALLERY)
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Ames House Prices - EDA gallery</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f5f6f8; color: #1f2933; }}
  header {{ background: #1f3a5f; color: #fff; padding: 26px 40px; }}
  header h1 {{ margin: 0 0 6px; font-size: 24px; }}
  header p {{ margin: 0; opacity: .85; font-size: 14px; }}
  main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr));
          gap: 22px; padding: 28px 40px 48px; }}
  .card {{ margin: 0; background: #fff; border-radius: 10px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  .card img {{ width: 100%; display: block; }}
  figcaption {{ padding: 12px 14px 14px; font-size: 13px; line-height: 1.45; color: #3e4c59; }}
  figcaption strong {{ color: #1f3a5f; font-size: 13.5px; }}
</style>
</head>
<body>
<header>
  <h1>Ames House Prices — exploratory data analysis</h1>
  <p>1,460 training sales · 79 explanatory variables · 16 figures. Generated by <code>code/02_eda.py</code>.</p>
</header>
<main>
{cards}
</main>
</body>
</html>
"""
    (FIG / "eda_gallery.html").write_text(html, encoding="utf-8")
    print("wrote eda_gallery.html")


def contact_sheet() -> None:
    """One-page overview of all 16 figures, for slides and quick review."""
    fig, axes = plt.subplots(4, 4, figsize=(17, 16))
    for ax, (name, _, _) in zip(axes.ravel(), GALLERY):
        ax.imshow(mpimg.imread(FIG / name))
        ax.set_axis_off()
    for ax in axes.ravel()[len(GALLERY):]:
        ax.set_axis_off()
    fig.suptitle("EDA overview — Ames house prices (1,460 sales, 79 variables)",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.6, w_pad=1.0)
    fig.savefig(FIG / "eda_contact_sheet.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote eda_contact_sheet.png")


def main() -> None:
    report_charts()
    extra_charts()
    gallery()
    contact_sheet()
    print(f"EDA complete: {len(GALLERY)} figures + gallery + contact sheet")


if __name__ == "__main__":
    main()
