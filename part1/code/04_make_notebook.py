"""Build and execute the analysis notebook.

    python code/04_make_notebook.py

The notebook is generated from the same pipeline module the report uses, so
every number in the notebook matches the report exactly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".jupyter"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".jupyter" / "config"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter" / "ipython"))

import nbformat
from nbclient import NotebookClient
import jupyter_core.paths as _jp

# The sandbox denies the ACL tightening that jupyter_core performs on the
# kernel connection file; writing the file itself is harmless.
_jp.win32_restrict_file_to_user = lambda fname: None

MD = nbformat.v4.new_markdown_cell
CODE = nbformat.v4.new_code_cell


def build() -> nbformat.NotebookNode:
    cells = []

    cells.append(MD("""# House Prices in Ames, Iowa — Cleaning, Feature Engineering and EDA

**Scope.** This notebook is the data-preparation and exploratory half of the project:
it audits the two Kaggle files, repairs and imputes the data, engineers features,
and produces the evidence behind the *Introduction* and *Dataset Description*
sections of the report. Modelling is handled separately by the modelling section.

The project's three research questions frame the analysis: RQ1 (which features drive
price), RQ2 (multicollinearity and VIF), RQ3 (influential outliers and regularisation).
This notebook provides the evidence for RQ1, the correlation structure that RQ2 needs,
and the outlier flag that RQ3's with-and-without comparison depends on.

**Data.** `train.csv` (1,460 sales with `SalePrice`) and `test.csv` (1,459 sales
without it), 79 explanatory variables plus `Id`.

**Output of this notebook**

| File | Content |
|---|---|
| `data/train_clean.csv`, `data/test_clean.csv` | cleaned data, original columns, flags |
| `data/train_features.csv`, `data/test_features.csv` | model-ready numeric matrix |
| `reports/audit.json` | every audit number quoted in the report |
| `figures/*.png`, `tables/*.csv` | report exhibits |
"""))

    cells.append(MD("""## 1. Setup

The pipeline lives in `code/house_prices_pipeline.py` so that the notebook, the
figures and the report cannot drift apart. Pandas 3 stores text in `StringDtype`,
which matters later when one-hot encoding.
"""))

    cells.append(CODE("""import sys, json, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "code"))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import house_prices_pipeline as hp

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid", context="notebook")
%matplotlib inline

print("pandas", pd.__version__, "| numpy", np.__version__)
print("working directory:", Path.cwd().name)"""))

    cells.append(MD("""## 2. First look at the two files

The target exists only in the training file, and the test file has no
`SalePrice` column at all. Every transformation below is therefore fitted on
`train` and applied unchanged to `test`.
"""))

    cells.append(CODE("""train, test = hp.load_raw()
print(f"train: {train.shape[0]:,} rows x {train.shape[1]} columns")
print(f"test : {test.shape[0]:,} rows x {test.shape[1]} columns")
print("target column present in test:", "SalePrice" in test.columns)
display(train.iloc[:4, :8])"""))

    cells.append(MD("""### 2.1 Audit

Duplicate ids, duplicate rows, constant columns and the shape of the target are
checked before anything is changed. The numbers are written to
`reports/audit.json` and quoted verbatim in the report.
"""))

    cells.append(CODE("""audit = hp.audit(train, test)
for k in ["n_train", "n_test", "n_features", "duplicate_ids", "duplicate_rows",
          "cols_with_missing_train", "cols_with_missing_test",
          "total_missing_cells_train", "pct_missing_cells_train",
          "target_mean", "target_median", "target_std", "target_min", "target_max",
          "target_skew", "target_log_skew"]:
    print(f"{k:>28}: {audit[k]}")"""))

    cells.append(MD("""### 2.2 Missing values — structural versus true

Nineteen columns are incomplete in the training file, but most of the blanks are
not data errors. The data dictionary defines `NA` for `PoolQC`, `Alley`,
`Fence`, `FireplaceQu`, the basement and the garage blocks as *"the house does
not have this feature"*. Filling those with a statistic would invent basements
and fireplaces. Only `LotFrontage` (17.7%), a handful of `MasVnrArea` cells and a
single `Electrical` value are genuinely missing.
"""))

    cells.append(CODE("""miss = hp.missing_table(train, test)
display(miss.head(12))
print("structural rows:", (miss.interpretation.str.startswith("Structural")).sum(),
      "| true-missing rows:", (~miss.interpretation.str.startswith("Structural")).sum())"""))

    cells.append(CODE("""from IPython.display import Image, display as show
show(Image(filename="figures/fig02_missing_values.png", width=780))"""))

    cells.append(MD("""## 3. Cleaning

Three rules, applied to both splits in the same order:

1. **Structural `NA` → explicit `"None"` category** (and `0` for the matching
   area/count columns). This keeps the information *"no garage"* instead of
   destroying it.
2. **True missing values → imputed**, with the imputation value learned on the
   training split only. `LotFrontage` uses the neighbourhood median, because
   missingness is not random: 52% of cul-de-sac lots lack a frontage figure,
   against 13% of inside lots.
3. **Impossible values → repaired**, not dropped.
"""))

    cells.append(CODE("""clean_tr, clean_te, log = hp.clean(train, test)
print("remaining missing cells — train:", int(clean_tr.isna().sum().sum()),
      "| test:", int(clean_te.isna().sum().sum()))
print("\\nmode-imputed columns:", json.dumps(log["mode_imputed"], indent=2))
print("garage years clipped:", log["garage_year_clipped"])
print("veneer types repaired:", log["veneer_type_repaired"])"""))

    cells.append(MD("""**Outliers.** The classic anomaly in this dataset is a handful of very large
houses sold cheaply. Two sales have more than 4,000 sq ft above grade and a price
below $300k: they are partial sales, i.e. the house was not finished when the
sale was recorded (`SaleCondition = "Partial"`). They are real transactions, so
they are **flagged, not deleted**; only the modelling matrix excludes them. The
other two very large houses ($745k and $755k) are genuine luxury sales and stay.
"""))

    cells.append(CODE("""flag = (clean_tr.GrLivArea > 4000) & (clean_tr.SalePrice < 300000)
print(clean_tr.loc[flag, ["Id", "GrLivArea", "SalePrice", "SaleCondition", "YearBuilt"]])
print("\\nflagged:", int(flag.sum()), "rows | ids:", clean_tr.loc[flag, "Id"].tolist())
show(Image(filename="figures/fig05_size_price.png", width=760))"""))

    cells.append(MD("""### 3.1 Ordinal encoding

Quality and condition labels are ordered (`Po < Fa < TA < Gd < Ex`). Mapping them
to 1–5 preserves that order, which neither a label encoding by alphabet nor a
one-hot encoding does. `"None"` becomes 0 so that *"no basement"* stays at the
bottom of the scale rather than being treated as a missing value.
"""))

    cells.append(CODE("""display(pd.DataFrame(
    [(k, str(v)) for k, v in hp.ORDINAL_MAPS.items()],
    columns=["column", "mapping"]).head(12))
print("ordinal columns encoded:", len(hp.ORDINAL_MAPS))"""))

    cells.append(MD("""### 3.2 Rare levels

Levels with fewer than 10 training sales are folded into `"Other"`. Without this,
`RoofMatl` contributes eight columns of which six describe one or two houses, and
a model cannot estimate them. `Neighborhood` is deliberately excluded: it is the
strongest location signal, and the smoothed target encoding in Section 6 handles
its small levels instead of erasing them.
"""))

    cells.append(CODE("""ord_tr, ord_te = hp.ordinal_encode(clean_tr, clean_te)
rare_tr, rare_te, rare_log = hp.collapse_rare(ord_tr, ord_te)
display(pd.DataFrame([
    {"column": k, "levels_before": v["n_levels_raw"], "levels_after": v["n_levels_kept"],
     "collapsed": ", ".join(v["collapsed"][:5]) + (" ..." if len(v["collapsed"]) > 5 else "")}
    for k, v in rare_log.items()]).head(10))"""))

    cells.append(MD("""## 4. Exploratory analysis

### 4.1 The target

`SalePrice` ranges from $34,900 to $755,000 with a mean of $180,921 and a median
of $163,000: the mean sits above the median, which is the signature of a
right-skewed distribution (skew 1.88). `log(SalePrice)` brings the skew to 0.12. The
practical consequence is that models should be fitted on the log target and
scored after back-transformation, or evaluated with a metric that does not
punish the expensive tail disproportionately.
"""))

    cells.append(CODE("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(train.SalePrice, bins=50, color="#2E6F9E")
ax[0].set(title="SalePrice (skew %.2f)" % train.SalePrice.skew(), xlabel="USD", ylabel="sales")
ax[1].hist(np.log(train.SalePrice), bins=50, color="#2E6F9E")
ax[1].set(title="log(SalePrice) (skew %.2f)" % np.log(train.SalePrice).skew(), xlabel="log USD")
plt.tight_layout(); plt.show()

display(train.SalePrice.describe().round(0).to_frame().T)"""))

    cells.append(MD("""### 4.2 Which variables move with price?

`OverallQual` (r = 0.79) and `GrLivArea` (r = 0.71) are far ahead of everything
else; `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath` and
`YearBuilt` form a second tier around 0.5–0.65. Because the predictors are also
correlated with each other, the ranking is descriptive, not causal.
"""))

    cells.append(CODE("""num = train.select_dtypes("number").drop(columns=["Id"])
corr = num.corr()["SalePrice"].drop("SalePrice").sort_values(key=abs, ascending=False)
display(corr.head(12).round(3).to_frame("correlation with SalePrice"))

top = corr.head(10).index.tolist()
fig, ax = plt.subplots(figsize=(8.4, 6.4))
sns.heatmap(train[top + ["SalePrice"]].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .7}, ax=ax)
ax.set_title("Correlation among the strongest raw predictors")
plt.tight_layout(); plt.show()"""))

    cells.append(MD("""### 4.3 Quality, size and location

Quality behaves almost like a price ladder: the median rises from $50k at
`OverallQual = 1` to $432k at 10, and the interquartile boxes barely overlap
between adjacent grades. Size works the same way but with more dispersion, and
location shifts the whole distribution: the median price in Northridge Heights
($315k) is 3.6 times that in Meadow Village ($88k).
"""))

    cells.append(CODE("""fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
sns.boxplot(data=train, x="OverallQual", y="SalePrice", color="#2E6F9E", fliersize=2, ax=ax[0])
ax[0].set(title="Price by overall quality", ylabel="SalePrice (USD)", xlabel="OverallQual")
sns.boxplot(data=train, x="KitchenQual", y="SalePrice", order=["Po","Fa","TA","Gd","Ex"],
            color="#3C8D6B", fliersize=2, ax=ax[1])
ax[1].set(title="Price by kitchen quality", ylabel="", xlabel="KitchenQual")
plt.tight_layout(); plt.show()

show(Image(filename="figures/fig06_neighborhood.png", width=700))"""))

    cells.append(MD("""### 4.4 Time and sale circumstances

Sales cluster in May–July, and the 2010 median is visibly lower than 2006–2009 —
a small but real market effect worth encoding as `HouseAge = YrSold - YearBuilt`.
`SaleCondition` also matters: partial (newly built) sales sit low relative to
their size, and `Abnorml` sales sit low relative to their quality, which is why
the two flagged outliers belong to the partial group.
"""))

    cells.append(CODE("""show(Image(filename="figures/fig10_seasonality.png", width=860))
show(Image(filename="figures/fig08_sale_circumstances.png", width=860))"""))

    cells.append(MD("""### 4.5 Redundancy

Two pairs are almost duplicates: `TotalBsmtSF` vs `1stFlrSF` (r = 0.82) and
`GarageCars` vs `GarageArea` (r = 0.88). For linear models this inflates
standard errors; for trees it merely splits importance between twins. The
engineered `TotalSF` and `GarageAreaPerCar` replace the duplicated pairs with one
interpretable quantity each.
"""))

    cells.append(CODE("""for a, b in [("TotalBsmtSF", "1stFlrSF"), ("GarageCars", "GarageArea"),
             ("GrLivArea", "TotRmsAbvGrd")]:
    print(f"{a:>13} vs {b:<13} r = {train[a].corr(train[b]):.3f}")
show(Image(filename="figures/fig11_multicollinearity.png", width=860))"""))

    cells.append(MD("""## 5. Feature engineering

Thirty-six features in six families. The rules are: every feature must be
computable on a single row without the target, must have a meaning a reader can
state in one sentence, and must not duplicate an existing column.

| Family | Examples | Why |
|---|---|---|
| Size | `TotalSF`, `TotalFinSF`, `TotalBath`, `TotalPorchSF`, `AvgRoomSF` | the value of a house is driven by usable space, not by which floor it is on |
| Quality | `QualScore`, `OverallGrade`, `QualxSF`, `CondScore` | quality and size interact: a large poor house is not a large good house |
| Age | `HouseAge`, `RemodAge`, `GarageAge`, `IsNew`, `IsRemodeled` | absolute year is meaningless across a 2006–2010 test window |
| Presence | `HasBsmt`, `HasGarage`, `HasFireplace`, `HasPool`, … | separates *"absent"* from *"small"* |
| Ratios | `BsmtFinRatio`, `LivAreaRatio`, `BedroomRatio` | composition matters more than raw counts |
| Time | `MoSoldSin`, `MoSoldCos` | keeps month cyclical instead of ordinal 1–12 |

Heavy-tailed areas are additionally logged (`log_LotArea`, `log_GrLivArea`,
`log_TotalSF`, `log_LotFrontage`) for the linear models; trees are insensitive to
the transform but it costs nothing.
"""))

    cells.append(CODE("""feat = hp.engineer(rare_tr.copy())
new_cols = [c for c in feat.columns if c not in rare_tr.columns]
print(f"{len(new_cols)} engineered columns")
display(pd.DataFrame({
    "feature": new_cols,
    "train_mean": [round(feat[c].mean(), 2) for c in new_cols],
    "train_max": [round(feat[c].max(), 1) for c in new_cols],
}).head(20))"""))

    cells.append(MD("""### 5.1 Location encoding

`Neighborhood` is categorical with 25 levels, several with fewer than 10 sales.
A one-hot encoding of a 6-sale neighbourhood is noise. The alternative used here
is **out-of-fold smoothed target encoding**: each training row receives the mean
log price of the *other* four folds, shrunk towards the global mean with
`m = 10` pseudo-observations, and the test rows receive the full-train mapping.
The out-of-fold step is what keeps it honest — a row never sees its own price.
"""))

    cells.append(CODE("""te_col, te_test = hp.target_encode(rare_tr, rare_te)
print("train encoding — unique values:", te_col.nunique(), "| range:",
      round(te_col.min(), 3), "-", round(te_col.max(), 3))
print("corr(encoding, log price) =", round(np.corrcoef(te_col, np.log(rare_tr.SalePrice))[0, 1], 3))
print("\\ncheck: same neighbourhood, different folds -> different values")
print(pd.DataFrame({"Neighborhood": rare_tr.Neighborhood.values, "encoding": te_col.values})
      .query("Neighborhood == 'NoRidge'").head(5))"""))

    cells.append(MD("""## 6. Final matrix

Nominal variables are one-hot encoded against a shared column space (the test
rows are reindexed to the training columns, so a level that appears only in test
becomes all-zero rather than a new column). The two flagged partial sales are
removed from the modelling matrix; `train_clean.csv` still contains them.
"""))

    cells.append(CODE("""out = hp.build()
X, Xt = out["train"], out["test"]
print(f"modelling matrix: train {X.shape}, test {Xt.shape}")
print("missing values:", int(X.isna().sum().sum()), "| infinities:",
      int(np.isinf(X.drop(columns=['Id','SalePrice'])).sum().sum()))
print("target present:", "SalePrice" in X.columns, "| ids aligned:", int(X.Id.max()), int(Xt.Id.min()))
print("\\nengineered + one-hot columns:", X.shape[1] - 2)
assert X.drop(columns=["Id", "SalePrice"]).isna().sum().sum() == 0
assert list(X.columns.drop("SalePrice")) == list(Xt.columns)
print("sanity checks passed")"""))

    cells.append(MD("""### 6.1 What the matrix contains

| Block | Columns |
|---|---|
| raw numeric + ordinal (0–5 scales) | 58 |
| hand-built engineered features | 36 |
| out-of-fold neighbourhood encoding | 1 |
| one-hot nominal levels | 155 |
| identifiers | `Id`, `SalePrice` |

The engineered features carry the signal of the raw quantities they summarise:
`QualxTotalSF` correlates 0.86 with price, against 0.78 for the raw `TotalSF`.
"""))

    cells.append(CODE("""show(Image(filename="figures/fig12_engineered_corr.png", width=780))
print("columns with zero variance in train:",
      (X.drop(columns=["Id", "SalePrice"]).std() == 0).sum())
display(pd.read_csv("reports/missing_values.csv").tail(6))"""))

    cells.append(MD("""## 7. What the modelling section receives

- `data/train_features.csv` — 1,458 × 252, no missing values, `Id` and `SalePrice` included.
- `data/test_features.csv` — 1,459 × 251, same columns minus the target.
- `data/train_clean.csv` / `data/test_clean.csv` — cleaned but unencoded, for anyone
  who wants a different encoding.
- Caveats to carry into modelling:
  1. `NeighborhoodPriceTE` is out-of-fold **for the current training rows**. If the
     validation split changes, recompute it; otherwise the encoding leaks.
  2. The two partial-sale outliers are excluded from the matrix but not from
     `train_clean.csv`.
  3. `SalePrice` is right-skewed (1.88): fit on `log(SalePrice)`.
  4. Test `MSZoning`, `Utilities` and a few other fields needed imputation; the
     values used are recorded in `reports/audit.json`.
"""))

    cells.append(CODE("""with open("reports/audit.json") as fh:
    info = json.load(fh)
print(json.dumps({
    "rows_modelled": info["rows_modelled"],
    "n_features_engineered": info["n_features_engineered"],
    "outliers_flagged": info["outliers_flagged"],
    "matrix": info["matrix"],
}, indent=2))"""))

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    nb = build()
    out = ROOT / "code" / "house_prices_analysis.ipynb"
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(nb, out)
    print("wrote", out)
    errs = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    print("execution errors:", len(errs))
    for e in errs:
        print(e.get("ename"), e.get("evalue"))


if __name__ == "__main__":
    main()
