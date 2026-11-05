"""Bài 2 - Dữ liệu văn bản: 20 Newsgroups (20 chủ đề, ~18.8k bài).

Nguồn công khai: http://qwone.com/~jason/20Newsgroups/ (tải qua sklearn.datasets.fetch_20newsgroups).
Đặc trưng truyền thống: BoW, TF-IDF (unigram / n-gram).
Đặc trưng hiện đại: GloVe (trung bình vector từ) và DistilBERT (mean pooling) -> lưu .npy.
"""
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC

from common import ResultLogger, Timer, evaluate, save_features, save_fig

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
def load_20ng(remove_meta: bool = True):
    """Trả về (train_texts, y_train, test_texts, y_test, class_names)."""
    remove = ("headers", "footers", "quotes") if remove_meta else ()
    tr = fetch_20newsgroups(subset="train", remove=remove, random_state=42)
    te = fetch_20newsgroups(subset="test", remove=remove, random_state=42)
    return tr.data, tr.target, te.data, te.target, list(tr.target_names)


def clean_text(text: str, lowercase=True, remove_stopwords=False, min_len=2) -> str:
    """Làm sạch: lowercase, bỏ URL/email/số, tokenization, (tuỳ chọn) bỏ stopwords."""
    if lowercase:
        text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if len(t) >= min_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def preprocess(texts, **kw):
    return [clean_text(t, **kw) for t in texts]


# ----------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------
def run_eda(train_texts, y_train, class_names, top_k: int = 20):
    n_words = np.array([len(t.split()) for t in train_texts])
    n_chars = np.array([len(t) for t in train_texts])
    empty = int((n_words == 0).sum())
    print(f"Số văn bản train: {len(train_texts)} | số lớp: {len(class_names)}")
    print(f"Văn bản rỗng sau khi bỏ header/footer/quote: {empty}")
    print(pd.Series(n_words).describe().round(1).to_string())

    counts = pd.Series(y_train).value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(counts)), counts.values)
    plt.xticks(range(len(counts)), [class_names[i] for i in counts.index], rotation=90, fontsize=8)
    plt.title("Số văn bản theo lớp (train)")
    save_fig("text_class_dist")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.clip(n_words, 0, 1500), bins=60)
    axes[0].set_title("Phân phối số từ / văn bản (cắt ở 1500)")
    axes[0].axvline(np.median(n_words), color="r", ls="--", label=f"median={np.median(n_words):.0f}")
    axes[0].legend()
    axes[1].hist(np.log10(n_chars + 1), bins=60)
    axes[1].set_title("Phân phối log10(số ký tự)")
    save_fig("text_length_dist")

    cleaned = preprocess(train_texts, remove_stopwords=False)
    cleaned_sw = preprocess(train_texts, remove_stopwords=True)
    for name, docs in (("giữ stopwords", cleaned), ("bỏ stopwords", cleaned_sw)):
        freq = Counter(w for d in docs for w in d.split())
        print(f"\nTop {top_k} từ ({name}); kích thước từ vựng = {len(freq)}")
        top = freq.most_common(top_k)
        plt.figure(figsize=(9, 4))
        plt.bar([w for w, _ in top], [c for _, c in top])
        plt.xticks(rotation=60)
        plt.title(f"Tần suất từ phổ biến ({name})")
        save_fig("text_top_words_stop" if "giữ" in name else "text_top_words_nostop")

    # Zipf
    freq = Counter(w for d in cleaned for w in d.split())
    vals = np.array(sorted(freq.values(), reverse=True))
    plt.figure(figsize=(5, 4))
    plt.loglog(np.arange(1, len(vals) + 1), vals)
    plt.title("Luật Zipf (rank vs tần suất)")
    plt.xlabel("rank")
    plt.ylabel("tần suất")
    save_fig("text_zipf")

    # Độ dài trung bình theo lớp
    by_class = pd.DataFrame({"label": [class_names[i] for i in y_train], "n_words": n_words})
    print("\nSố từ trung bình theo lớp:")
    print(by_class.groupby("label")["n_words"].agg(["mean", "median"]).round(0).to_string())


# ----------------------------------------------------------------------------
# Đặc trưng truyền thống
# ----------------------------------------------------------------------------
def make_vectorizer(kind: str, ngram=(1, 1), max_features=50000, min_df=2, stop_words=None):
    kw = dict(ngram_range=ngram, max_features=max_features, min_df=min_df, stop_words=stop_words)
    if kind == "bow":
        return CountVectorizer(**kw)
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **kw)
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Đặc trưng hiện đại (embedding)
# ----------------------------------------------------------------------------
_glove = None


def glove_embeddings(texts, name="glove-wiki-gigaword-100"):
    """Trung bình vector GloVe của các từ trong văn bản (OOV bị bỏ qua)."""
    global _glove
    if _glove is None:
        import gensim.downloader as api

        _glove = api.load(name)
    dim = _glove.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [_glove[w] for w in TOKEN_RE.findall(t.lower()) if w in _glove.key_to_index]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def bert_embeddings(texts, model_name="distilbert-base-uncased", max_len=256, batch_size=64):
    """Embedding ngữ cảnh: mean pooling của last_hidden_state (có attention mask)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    if dev.type == "cuda":
        model.half()
    # sắp theo độ dài để giảm padding
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[idx] = emb.float().cpu().numpy()
    return out


def extract_and_save(name, fn, train_texts, test_texts, **kw):
    """Chạy trích xuất embedding cho train/test và lưu .npy vào features/."""
    with Timer() as t:
        tr = fn(train_texts, **kw)
        te = fn(test_texts, **kw)
    save_features(f"text_{name}_train", tr)
    save_features(f"text_{name}_test", te)
    print(f"{name}: train {tr.shape}, test {te.shape} ({t.elapsed:.0f}s) -> features/text_{name}_*.npy")
    return tr, te


# ----------------------------------------------------------------------------
# Mô hình phân loại
# ----------------------------------------------------------------------------
def make_classifier(name: str, **kw):
    if name == "nb":
        return MultinomialNB(alpha=kw.get("alpha", 0.1))
    if name == "cnb":
        return ComplementNB(alpha=kw.get("alpha", 0.3))
    if name == "gnb":
        return GaussianNB()
    if name == "logreg":
        return LogisticRegression(max_iter=1000, C=kw.get("C", 10.0))
    if name == "linsvm":
        return LinearSVC(C=kw.get("C", 0.5), max_iter=10000)
    raise ValueError(name)


def fit_eval(clf, Xtr, ytr, Xte, yte, average="macro"):
    with Timer() as t:
        clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    res = evaluate(yte, pred, average=average)
    res["fit_seconds"] = round(t.elapsed, 2)
    return res, pred


def finetune_bert(train_texts, y_train, test_texts, y_test, n_classes,
                  model_name="distilbert-base-uncased", max_len=192, epochs=3,
                  batch_size=32, lr=5e-5, seed=42):
    """Fine-tune đầu-cuối DistilBERT (điểm cộng pipeline học sâu)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes).to(dev)

    def encode(texts, labels):
        enc = tok(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))

    tr_dl = DataLoader(encode(train_texts, y_train), batch_size=batch_size, shuffle=True)
    te_dl = DataLoader(encode(test_texts, y_test), batch_size=batch_size * 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(tr_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    with Timer() as t:
        for ep in range(epochs):
            model.train()
            loss_sum = 0.0
            for ids, mask, lab in tr_dl:
                ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
                opt.zero_grad()
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                loss_sum += loss.item()
            print(f"  epoch {ep + 1}/{epochs} loss={loss_sum / len(tr_dl):.4f}")
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, mask, _ in te_dl:
            logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
            preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    res = evaluate(y_test, pred, average="macro")
    res["fit_seconds"] = round(t.elapsed, 2)
    return res, pred
