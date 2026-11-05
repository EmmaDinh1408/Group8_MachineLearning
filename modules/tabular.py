"""Bài 1 - Dữ liệu bảng: Adult Census Income (UCI).

Nguồn công khai: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- ~32.5k mẫu, 14 thuộc tính (6 số + 8 phân loại), nhãn income >50K / <=50K.
- Có missing value (ký hiệu '?') ở workclass, occupation, native-country.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC

from common import ResultLogger, Timer, download, evaluate, save_fig

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
TARGET = "income"


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
def load_adult(data_dir: str = "data") -> pd.DataFrame:
    path = download(DATA_URL, os.path.join(data_dir, "adult.data"))
    df = pd.read_csv(
        path, header=None, names=COLUMNS, skipinitialspace=True,
        na_values="?", sep=r",\s*", engine="python",
    )
    df[TARGET] = (df[TARGET].str.strip() == ">50K").astype(int)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def feature_types(X: pd.DataFrame):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


# ----------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    num, cat = feature_types(df.drop(columns=[TARGET]))
    print(f"Kích thước: {df.shape} | số: {len(num)} cột | phân loại: {len(cat)} cột")
    print(f"Mẫu trùng lặp: {df.duplicated().sum()}")
    display_df = df.describe().T
    print("\n== Thống kê mô tả (số) ==")
    print(display_df.round(2))
    print("\n== Thống kê (phân loại) ==")
    print(df[cat].describe().T)

    # Missing value
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print("\n== Giá trị thiếu ==")
    print(pd.DataFrame({"missing": miss, "ratio_%": (miss / len(df) * 100).round(2)}))
    if len(miss):
        plt.figure(figsize=(6, 3))
        (miss / len(df) * 100).plot.bar(color="tab:red")
        plt.ylabel("% thiếu")
        plt.title("Tỉ lệ giá trị thiếu theo cột")
        save_fig("tab_missing")

    # Phân phối nhãn
    plt.figure(figsize=(4, 3))
    df[TARGET].value_counts().sort_index().plot.bar(color=["tab:blue", "tab:orange"])
    plt.xticks([0, 1], ["<=50K", ">50K"], rotation=0)
    plt.title("Phân phối nhãn")
    save_fig("tab_label_dist")
    print("Tỉ lệ nhãn:", df[TARGET].value_counts(normalize=True).round(3).to_dict())

    # Phân phối biến số + outlier
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, c in zip(axes.ravel(), num):
        sns.histplot(df[c], bins=40, ax=ax, kde=False)
        ax.set_title(f"{c} (skew={df[c].skew():.2f})")
    save_fig("tab_num_hist")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, c in zip(axes.ravel(), num):
        sns.boxplot(x=df[TARGET], y=df[c], ax=ax)
        ax.set_title(c)
    save_fig("tab_num_box")

    # Tương quan
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[num + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Ma trận tương quan")
    save_fig("tab_corr")

    # Biến phân loại vs nhãn
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, c in zip(axes.ravel(), cat):
        rate = df.groupby(c)[TARGET].mean().sort_values()
        rate.plot.barh(ax=ax)
        ax.set_title(f"P(>50K) theo {c}", fontsize=9)
        ax.tick_params(labelsize=7)
    save_fig("tab_cat_vs_target")

    # Outlier theo IQR
    rows = []
    for c in num:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
        rows.append({"feature": c, "outliers_IQR": n_out, "ratio_%": round(n_out / len(df) * 100, 2)})
    print("\n== Outlier (quy tắc IQR) ==")
    print(pd.DataFrame(rows))


# ----------------------------------------------------------------------------
# Pipeline cấu hình được
# ----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "num_imputer": "median",          # mean | median
    "cat_imputer": "most_frequent",   # most_frequent | constant
    "encoding": "onehot",             # onehot | ordinal
    "scaler": "standard",             # standard | minmax | none
    "feature_range": (0, 1),          # dùng cho MinMaxScaler
    "pca": None,                      # None | 0.90 | 0.95 | int
    "model": "logreg",                # logreg | linsvm | rf
    "model_params": {},
    "seed": 42,
}


def build_pipeline(X: pd.DataFrame, **overrides) -> Pipeline:
    cfg = {**DEFAULT_CONFIG, **overrides}
    num, cat = feature_types(X)

    if cfg["scaler"] == "standard":
        scaler = StandardScaler()
    elif cfg["scaler"] == "minmax":
        scaler = MinMaxScaler(feature_range=tuple(cfg["feature_range"]))
    else:
        scaler = "passthrough"
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["num_imputer"])),
        ("scaler", scaler),
    ])

    if cfg["cat_imputer"] == "constant":
        cat_imp = SimpleImputer(strategy="constant", fill_value="Missing")
    else:
        cat_imp = SimpleImputer(strategy="most_frequent")
    if cfg["encoding"] == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_pipe = Pipeline([("imputer", cat_imp), ("encoder", enc)])

    pre = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])
    steps = [("pre", pre)]
    if cfg["pca"] is not None:
        steps.append(("pca", PCA(n_components=cfg["pca"], random_state=cfg["seed"])))

    mp = dict(cfg["model_params"])
    if cfg["model"] == "logreg":
        model = LogisticRegression(max_iter=2000, **mp)
    elif cfg["model"] == "linsvm":
        model = LinearSVC(C=mp.pop("C", 1.0), max_iter=20000, random_state=cfg["seed"], **mp)
    elif cfg["model"] == "rf":
        model = RandomForestClassifier(
            n_estimators=mp.pop("n_estimators", 200), n_jobs=-1, random_state=cfg["seed"], **mp
        )
    else:
        raise ValueError(cfg["model"])
    steps.append(("model", model))
    return Pipeline(steps)


def run_config(X_train, X_test, y_train, y_test, **cfg) -> dict:
    """Huấn luyện + đánh giá một cấu hình; trả về dict kết quả."""
    pipe = build_pipeline(X_train, **cfg)
    with Timer() as t:
        pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    res = evaluate(y_test, pred, average="binary")
    n_feat = pipe[:-1].transform(X_test.iloc[:5]).shape[1]
    res.update({"n_features": n_feat, "fit_seconds": round(t.elapsed, 2)})
    return res, pipe


def run_grid(X_train, X_test, y_train, y_test, grid: list, name: str = "tabular") -> pd.DataFrame:
    """grid: danh sách dict cấu hình. Trả về DataFrame kết quả (và lưu CSV)."""
    log = ResultLogger(name)
    for i, cfg in enumerate(grid, 1):
        res, _ = run_config(X_train, X_test, y_train, y_test, **cfg)
        row = {k: str(v) for k, v in cfg.items() if k != "model_params"}
        row.update(res)
        log.add(**row)
        print(f"[{i}/{len(grid)}] {row['model']:<7} scaler={row.get('scaler','-'):<8} "
              f"pca={row.get('pca','None'):<5} acc={res['accuracy']:.4f} f1={res['f1']:.4f}")
    return log.save()


def main_grid(models=("logreg", "linsvm", "rf"), scalers=None, pcas=(None, 0.90, 0.95)) -> list:
    scalers = scalers or [
        {"scaler": "standard"},
        {"scaler": "minmax", "feature_range": (0, 1)},
        {"scaler": "minmax", "feature_range": (-1, 1)},
    ]
    grid = []
    for m in models:
        for s in scalers:
            for p in pcas:
                grid.append({"model": m, "pca": p, **s})
    return grid
