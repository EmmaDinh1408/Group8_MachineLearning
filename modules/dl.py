"""Phần học sâu (điểm cộng): MLP PyTorch cho vector đặc trưng + vòng huấn luyện dùng chung."""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from common import Timer, evaluate


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, hidden=(256, 128), dropout=0.3, lr=1e-3,
              weight_decay=1e-4, epochs=60, batch_size=256, patience=8, average="macro",
              seed=42, verbose=True):
    """Huấn luyện MLP với early stopping trên 10% tập train. Trả về (metrics, model)."""
    torch.manual_seed(seed)
    dev = get_device()
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    n_classes = int(max(y_train.max(), np.asarray(y_test).max()) + 1)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=seed)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()
    Xv, yv = torch.from_numpy(X_va).to(dev), torch.from_numpy(y_va).long().to(dev)

    model = MLP(X_tr.shape[1], n_classes, hidden, dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, bad = float("inf"), None, 0

    with Timer() as t:
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 2:
                    continue
                xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xv), yv).item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
            if verbose and (ep + 1) % 10 == 0:
                print(f"  epoch {ep + 1:3d} val_loss={vl:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(dev))
        pred = logits.argmax(1).cpu().numpy()
    res = evaluate(y_test, pred, average=average)
    res["fit_seconds"] = round(t.elapsed, 2)
    res["epochs_run"] = ep + 1
    return res, model, pred
