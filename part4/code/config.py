# -*- coding: utf-8 -*-
"""
Part 4 · 统一数据接口
========================
本模块是 Part4 全部脚本的唯一数据入口，保证四个成员的结果放在同一个口径下比较：

* 输入：Part1 交付的特征矩阵 ``part1/5001/data/train_features.csv``（1458 × 252）
  与 ``test_features.csv``（1459 × 251），以及 ``train_clean.csv``（含 log(SalePrice)）。
* 目标：``SalePriceLog = log(SalePrice)``（与 Part2 / Part3 完全一致）。
* 划分：``train_test_split(test_size=0.2, random_state=42)``（与 Part2 / Part3 对齐）。
* 交叉验证：``KFold(n_splits=10, shuffle=True, random_state=42)``（与 Part3 对齐）。
* 异常样本：``SalePriceOutlier``（Id 524 / 1299）默认 **不剔除**，与 Part2 / Part3 一致；
  需要做 RQ3 对照时可通过 ``drop_outlier=True`` 打开。

作者：Yifei Lv（Part4 负责人）
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 多进程临时目录
# 本机 Windows 用户名含中文字符，joblib/loky 的子进程管道默认用 ASCII 编码，
# 在含中文的临时目录下会抛 UnicodeEncodeError。这里把 joblib 的临时目录
# 指向一个纯 ASCII 路径（如 C:\sdsc5001_tmp），才能安全地使用 n_jobs=-1 并行。
_ASCII_TMP = os.path.join(
    os.environ.get("SystemDrive", "C:") + "\\", "sdsc5001_tmp"
)
try:
    os.makedirs(_ASCII_TMP, exist_ok=True)
    # 验证该路径确实是纯 ASCII
    _ASCII_TMP.encode("ascii")
except Exception:                                     # pragma: no cover
    _ASCII_TMP = r"C:\sdsc5001_tmp"
    os.makedirs(_ASCII_TMP, exist_ok=True)

os.environ["JOBLIB_TEMP_FOLDER"] = _ASCII_TMP
os.environ["TMPDIR"] = _ASCII_TMP
os.environ["TEMP"] = _ASCII_TMP
os.environ["TMP"] = _ASCII_TMP

# ---------------------------------------------------------------- 路径配置
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PART1_DIR = os.path.join(ROOT, "part1", "5001")
FEATURES_DIR = os.path.join(PART1_DIR, "data")
FIG_DIR = os.path.join(ROOT, "part4", "figures")
OUT_DIR = os.path.join(ROOT, "part4", "output")

TRAIN_FEATURES = os.path.join(FEATURES_DIR, "train_features.csv")
TEST_FEATURES = os.path.join(FEATURES_DIR, "test_features.csv")
TRAIN_CLEAN = os.path.join(FEATURES_DIR, "train_clean.csv")
SAMPLE_SUB = os.path.join(ROOT, "data_raw", "sample_submission.csv")

# ---------------------------------------------------------------- 全局口径常量
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 10
TARGET = "SalePriceLog"          # log 尺度目标
ID_COL = "Id"
PRICE_COL = "SalePrice"
OUTLIER_FLAG = "SalePriceOutlier"
LEAK_COLS = [ID_COL, PRICE_COL, TARGET, OUTLIER_FLAG]

# Part3 选定的 8 个核心连续变量（用于非线性扩展 / 局部解释）
CORE_CONTINUOUS = [
    "GrLivArea", "TotalSF", "TotalBsmtSF", "LotArea",
    "GarageArea", "1stFlrSF", "YearBuilt", "OverallQual",
]


def _ensure_dirs() -> None:
    for d in (FIG_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)


@dataclass
class Dataset:
    """统一的数据容器。"""

    train: pd.DataFrame      # 特征 + 目标 + 标记
    test: pd.DataFrame       # 特征（无目标）
    feature_cols: list       # 建模用特征列（不含泄漏列）
    n_train: int
    n_test: int

    # ---------------------------------------------------------- 便捷属性
    @property
    def y(self) -> np.ndarray:
        return self.train[TARGET].to_numpy(dtype=float)

    @property
    def X(self) -> pd.DataFrame:
        return self.train[self.feature_cols]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test[self.feature_cols]

    def summary(self) -> str:
        return (
            f"Dataset(train={self.n_train}, test={self.n_test}, "
            f"n_features={len(self.feature_cols)})"
        )


def load_dataset(drop_outlier: bool = False) -> Dataset:
    """加载 Part1 的特征矩阵并对齐目标列。

    Parameters
    ----------
    drop_outlier:
        是否剔除两个 partial-sale 异常样本（Id 524 / 1299）。
        默认 ``False``，与 Part2 / Part3 保持一致。
    """
    _ensure_dirs()

    feat = pd.read_csv(TRAIN_FEATURES)
    test_feat = pd.read_csv(TEST_FEATURES)
    clean = pd.read_csv(TRAIN_CLEAN)

    # 目标与标记来自 clean（特征矩阵中 SalePrice 已被剔除/保留视版本而定）
    meta_cols = [c for c in (ID_COL, TARGET, PRICE_COL, OUTLIER_FLAG) if c in clean.columns]
    meta = clean[meta_cols].copy()

    train = feat.merge(meta, on=ID_COL, how="left", suffixes=("", "_meta"))
    # 若特征矩阵本身已含 SalePrice 列，删除重复列
    for c in (TARGET, PRICE_COL, OUTLIER_FLAG):
        if c in train.columns and f"{c}_meta" in train.columns:
            train = train.drop(columns=[f"{c}_meta"])

    if train[TARGET].isna().any():
        raise ValueError("存在无法匹配 SalePriceLog 的样本，请检查 Part1 产物是否一致。")

    test = test_feat.copy()

    if drop_outlier:
        train = train[train[OUTLIER_FLAG] == 0].reset_index(drop=True)

    feature_cols = [c for c in train.columns if c not in LEAK_COLS]

    # 校验：train / test 特征列必须完全一致
    test_feature_cols = [c for c in test.columns if c not in (ID_COL, PRICE_COL, TARGET)]
    missing_in_test = set(feature_cols) - set(test_feature_cols)
    if missing_in_test:
        raise ValueError(f"测试集缺少特征列：{sorted(missing_in_test)[:10]}")

    return Dataset(
        train=train,
        test=test,
        feature_cols=feature_cols,
        n_train=len(train),
        n_test=len(test),
    )


def split_80_20(ds: Dataset):
    """与 Part2 / Part3 完全对齐的 80/20 划分。"""
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_tr, X_te, y_tr, y_te


def kfold_splits(n_splits: int = N_SPLITS):
    """与 Part3 对齐的 10 折 CV 划分器。"""
    from sklearn.model_selection import KFold

    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------- 绘图统一样式
def apply_style() -> None:
    """全项目统一的 matplotlib 风格，保证 Part4 所有图表视觉一致。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        # 注意：不要用 savefig.bbox="tight"。在较新的 matplotlib 上它会把
        # 画布撑成异常高度（标题与坐标轴标签被裁切或压在一起）。
        # 各绘图脚本已显式调用 fig.tight_layout()，这里只需关闭 tight bbox。
        "savefig.bbox": None,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.autolayout": False,
    })


# 与 PPT 视觉一致的主色板
COLORS = {
    "primary": "#1F4E79",     # 深蓝 —— OLS / 基准
    "accent":  "#C55A11",     # 橙 —— Ridge
    "green":   "#2E7D32",     # 绿 —— Lasso / 推荐
    "purple":  "#6A3D9A",     # 紫 —— RF
    "red":     "#B22222",     # 红 —— GBR / 强调
    "grey":    "#8C8C8C",
    "light":   "#D9D9D9",
    "ink":     "#1A1A1A",
}

MODEL_COLORS = {
    "OLS": COLORS["primary"],
    "Ridge": COLORS["accent"],
    "Lasso": COLORS["green"],
    "Random Forest": COLORS["purple"],
    "Gradient Boosting": COLORS["red"],
}


if __name__ == "__main__":
    ds = load_dataset()
    print(ds.summary())
    print("特征列数:", len(ds.feature_cols))
    print("核心连续变量命中:", [c for c in CORE_CONTINUOUS if c in ds.feature_cols])
    print("目标统计:", pd.Series(ds.y).describe().to_dict())
