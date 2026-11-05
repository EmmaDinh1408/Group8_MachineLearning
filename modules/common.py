"""Tiện ích dùng chung cho cả 3 bài: seed, đánh giá, vẽ hình, lưu kết quả/đặc trưng."""
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures")
FEATURE_DIR = "features"


def ensure_dirs():
    for d in (REPORT_DIR, FIG_DIR, FEATURE_DIR, "data"):
        os.makedirs(d, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def evaluate(y_true, y_pred, average: str = "macro") -> dict:
    """Accuracy, precision, recall, F1 (average = 'macro' | 'weighted' | 'binary')."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def save_fig(name: str, show: bool = True):
    """Lưu hình vào reports/figures/<name>.png (dùng cho báo cáo) và hiển thị."""
    ensure_dirs()
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


def plot_confusion(y_true, y_pred, labels=None, title="Confusion matrix", name=None, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = cm.shape[0]
    size = max(5, min(14, n * 0.45))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    if labels is not None and n <= 25:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    if name:
        save_fig(name)
    else:
        plt.show()


class ResultLogger:
    """Gom kết quả thí nghiệm thành bảng và lưu CSV để báo cáo dùng lại."""

    def __init__(self, name: str):
        self.name = name
        self.rows = []

    def add(self, **kwargs):
        self.rows.append(kwargs)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self):
        ensure_dirs()
        df = self.frame()
        df.to_csv(os.path.join(REPORT_DIR, f"results_{self.name}.csv"), index=False)
        return df


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0


def save_features(name: str, array: np.ndarray, fmt: str = "npy"):
    """Lưu đặc trưng vào features/<name>.npy (hoặc .h5)."""
    ensure_dirs()
    path = os.path.join(FEATURE_DIR, f"{name}.{fmt}")
    if fmt == "npy":
        np.save(path, array)
    elif fmt == "h5":
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("features", data=array, compression="gzip")
    else:
        raise ValueError(fmt)
    return path


def load_features(name: str, fmt: str = "npy"):
    path = os.path.join(FEATURE_DIR, f"{name}.{fmt}")
    if fmt == "npy":
        return np.load(path)
    import h5py

    with h5py.File(path, "r") as f:
        return f["features"][:]


def save_json(name: str, obj: dict):
    ensure_dirs()
    with open(os.path.join(REPORT_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def download(url: str, dest: str):
    """Tải file công khai về dest (bỏ qua nếu đã có)."""
    import urllib.request

    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"Đang tải {url} ...")
    urllib.request.urlretrieve(url, dest)
    return dest
