"""Bài 3 - Dữ liệu ảnh: Oxford-IIIT Pet (37 giống chó/mèo, ~7.4k ảnh).

Nguồn công khai: https://www.robots.ox.ac.uk/~vgg/data/pets/ (tải qua torchvision.datasets.OxfordIIITPet).
Đặc trưng truyền thống: HOG (+ PCA).
Đặc trưng hiện đại: ResNet50 / EfficientNet-B0 / ViT-B/16 pretrained (ImageNet) -> lưu .npy.
"""
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from common import Timer, evaluate, save_features, save_fig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
def load_pets(root: str = "data"):
    """Trả về (train_ds, test_ds, class_names). Tập train = 'trainval' (3680), test = 'test' (3669)."""
    from torchvision.datasets import OxfordIIITPet

    tr = OxfordIIITPet(root, split="trainval", target_types="category", download=True)
    te = OxfordIIITPet(root, split="test", target_types="category", download=True)
    return tr, te, list(tr.classes)


def labels_of(ds):
    return np.array(ds._labels)


# ----------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------
def run_eda(train_ds, test_ds, class_names, sample_n: int = 2000, seed: int = 42):
    y_tr, y_te = labels_of(train_ds), labels_of(test_ds)
    print(f"Train: {len(train_ds)} ảnh | Test: {len(test_ds)} ảnh | số lớp: {len(class_names)}")

    counts = pd.Series(y_tr).value_counts().sort_index()
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(counts)), counts.values)
    plt.xticks(range(len(counts)), [class_names[i] for i in counts.index], rotation=90, fontsize=7)
    plt.title("Số ảnh theo lớp (train)")
    save_fig("img_class_dist")
    print(f"Số ảnh / lớp (train): min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    # Kích thước, kênh màu, tỉ lệ khung hình (lấy mẫu ngẫu nhiên để nhanh)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_ds), min(sample_n, len(train_ds)), replace=False)
    sizes, modes = [], Counter()
    for i in idx:
        with Image.open(train_ds._images[i]) as im:
            sizes.append(im.size)
            modes[im.mode] += 1
    w, h = np.array(sizes).T
    print(f"Chế độ màu (mẫu {len(idx)} ảnh): {dict(modes)}")
    print(f"Rộng: min={w.min()} max={w.max()} median={np.median(w):.0f} | "
          f"Cao: min={h.min()} max={h.max()} median={np.median(h):.0f}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(w, h, s=4, alpha=0.4)
    axes[0].set_title("Rộng vs Cao (px)")
    axes[1].hist(w / h, bins=40)
    axes[1].set_title("Tỉ lệ khung hình (W/H)")
    axes[2].bar(list(modes.keys()), list(modes.values()))
    axes[2].set_title("Số kênh màu (mode PIL)")
    save_fig("img_size_dist")

    # Ảnh mẫu
    fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    pick = rng.choice(len(train_ds), 18, replace=False)
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(Image.open(train_ds._images[i]).convert("RGB"))
        ax.set_title(class_names[y_tr[i]], fontsize=8)
        ax.axis("off")
    save_fig("img_samples")

    # Cường độ điểm ảnh trung bình theo kênh (ảnh đã resize nhỏ)
    chans = []
    for i in idx[:500]:
        a = np.asarray(Image.open(train_ds._images[i]).convert("RGB").resize((64, 64)), dtype=np.float32) / 255
        chans.append(a.reshape(-1, 3).mean(0))
    chans = np.array(chans)
    print("Trung bình RGB:", chans.mean(0).round(3), "| std:", chans.std(0).round(3))
    plt.figure(figsize=(5, 4))
    for c, col in enumerate(("r", "g", "b")):
        plt.hist(chans[:, c], bins=30, alpha=0.5, color=col, label=col.upper())
    plt.legend()
    plt.title("Phân phối cường độ trung bình theo kênh")
    save_fig("img_channel_dist")


# ----------------------------------------------------------------------------
# Đặc trưng truyền thống: HOG
# ----------------------------------------------------------------------------
def _hog_one(path, size, ppc, cpb, orientations):
    from skimage.feature import hog

    im = Image.open(path).convert("L").resize((size, size))
    return hog(np.asarray(im), orientations=orientations, pixels_per_cell=(ppc, ppc),
               cells_per_block=(cpb, cpb), block_norm="L2-Hys")


def hog_features(ds, size=128, pixels_per_cell=16, cells_per_block=2, orientations=9, n_jobs=-1):
    from joblib import Parallel, delayed

    feats = Parallel(n_jobs=n_jobs)(
        delayed(_hog_one)(p, size, pixels_per_cell, cells_per_block, orientations) for p in ds._images)
    return np.asarray(feats, dtype=np.float32)


# ----------------------------------------------------------------------------
# Đặc trưng hiện đại: pretrained backbone
# ----------------------------------------------------------------------------
def get_backbone(name: str):
    """Trả về (model đã bỏ lớp phân loại, kích thước đầu vào mặc định, dim đặc trưng)."""
    import torch.nn as nn
    import torchvision.models as tvm

    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        return m, 224
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
        return m, 224
    if name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()
        return m, 224  # ViT-B/16 yêu cầu đúng 224x224
    if name == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])  # 4096-d
        return m, 224
    raise ValueError(name)


def extract_deep_features(ds, name: str, img_size: int = None, batch_size: int = 64, num_workers: int = 2):
    """Resize ảnh -> chuẩn hoá ImageNet -> forward qua backbone pretrained -> vector đặc trưng."""
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, default_size = get_backbone(name)
    size = img_size or default_size
    if name == "vit_b_16" and size != 224:
        raise ValueError("ViT-B/16 chỉ hỗ trợ 224x224")
    model = model.to(dev).eval()

    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class _DS(Dataset):
        def __len__(self):
            return len(ds._images)

        def __getitem__(self, i):
            return tf(Image.open(ds._images[i]).convert("RGB"))

    dl = DataLoader(_DS(), batch_size=batch_size, num_workers=num_workers)
    out = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                f = model(xb)
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def extract_and_save(ds_train, ds_test, name: str, img_size: int = None):
    with Timer() as t:
        tr = extract_deep_features(ds_train, name, img_size)
        te = extract_deep_features(ds_test, name, img_size)
    tag = f"{name}_{img_size or 224}"
    save_features(f"img_{tag}_train", tr)
    save_features(f"img_{tag}_test", te)
    print(f"{tag}: train {tr.shape}, test {te.shape} ({t.elapsed:.0f}s) -> features/img_{tag}_*.npy")
    return tr, te


# ----------------------------------------------------------------------------
# Bộ phân loại
# ----------------------------------------------------------------------------
def make_classifier(name: str, **kw):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=kw.get("C", 1.0)))
    if name == "linsvm":
        return make_pipeline(StandardScaler(), LinearSVC(C=kw.get("C", 0.01), max_iter=20000))
    if name == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=kw.get("C", 10.0), gamma="scale"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=kw.get("n_estimators", 300), n_jobs=-1, random_state=42)
    raise ValueError(name)


def fit_eval(clf, Xtr, ytr, Xte, yte, average="macro"):
    with Timer() as t:
        clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    res = evaluate(yte, pred, average=average)
    res["fit_seconds"] = round(t.elapsed, 2)
    return res, pred


def pca_reduce(Xtr, Xte, n_components=0.95, seed=42):
    pca = PCA(n_components=n_components, random_state=seed).fit(Xtr)
    return pca.transform(Xtr), pca.transform(Xte), pca


# ----------------------------------------------------------------------------
# Fine-tuning đầu-cuối (điểm cộng pipeline học sâu)
# ----------------------------------------------------------------------------
def finetune_cnn(train_ds, test_ds, n_classes, arch="resnet50", epochs=5, batch_size=32,
                 lr=3e-4, img_size=224, num_workers=2, seed=42):
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arch == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError(arch)
    model = model.to(dev)

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tf_train = T.Compose([T.RandomResizedCrop(img_size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    tf_test = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])

    class _DS(Dataset):
        def __init__(self, base, tf):
            self.base, self.tf = base, tf

        def __len__(self):
            return len(self.base._images)

        def __getitem__(self, i):
            return self.tf(Image.open(self.base._images[i]).convert("RGB")), self.base._labels[i]

    tr_dl = DataLoader(_DS(train_ds, tf_train), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=True)
    te_dl = DataLoader(_DS(test_ds, tf_test), batch_size=batch_size * 2, num_workers=num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    with Timer() as t:
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    loss = loss_fn(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
            print(f"  epoch {ep + 1}/{epochs} loss={tot / len(tr_dl):.4f}")
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in te_dl:
            preds.append(model(xb.to(dev)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    res = evaluate(labels_of(test_ds), pred, average="macro")
    res["fit_seconds"] = round(t.elapsed, 2)
    return res, pred
