from nbutil import build, code, md, setup_cells

cells = setup_cells(
    "!pip -q install gensim transformers scikit-learn seaborn h5py",
    ["common.py", "text.py", "dl.py"],
    "import text as TX\nimport dl\n",
)

cells += [
    md("""## 1. Bài toán & dữ liệu

- **Bài toán:** phân loại văn bản tin nhóm thảo luận vào **20 chủ đề** (tôn giáo, máy tính, thể thao, y tế, không gian, chính trị…).
- **Dữ liệu:** *20 Newsgroups* — http://qwone.com/~jason/20Newsgroups/ (tải qua `sklearn.datasets.fetch_20newsgroups`, ~11.3k train / ~7.5k test).
  Bỏ `headers`, `footers`, `quotes` để mô hình không "ăn gian" bằng thông tin meta.
- **Vì sao phù hợp:** nhiều nhãn (20), nội dung đa dạng, kích thước đủ lớn.

> ⚠️ Đề yêu cầu các nhóm không trùng tập dữ liệu và nên thống nhất với giảng viên. Đổi dataset: sửa `load_20ng()` trong `modules/text.py`."""),
    code("""train_texts, y_train, test_texts, y_test, class_names = TX.load_20ng()
print(len(train_texts), len(test_texts), class_names[:5], '...')
print('--- Ví dụ ---'); print(train_texts[0][:400])"""),
    md("## 2. EDA"),
    code("TX.run_eda(train_texts, y_train, class_names)"),
    md("""**Nhận xét EDA** *(chỉnh theo kết quả thực tế)*:
- Số lớp tương đối cân bằng (~480–600 bài/lớp), riêng một vài lớp tôn giáo/chính trị ít hơn.
- Độ dài văn bản **lệch phải rất mạnh** (median vài trăm từ, đuôi dài hàng nghìn từ) và có một số văn bản rỗng ⇒ cần cắt (truncate) khi dùng BERT.
- Từ phổ biến nhất khi giữ stopwords là `the, to, of, and…` — không mang thông tin chủ đề ⇒ cần bỏ stopwords / dùng TF-IDF.
- Tần suất từ tuân theo **luật Zipf**: số ít từ rất phổ biến, đa số từ hiếm ⇒ cắt `min_df`, `max_features`."""),
    md("""## 3. Tiền xử lý văn bản (cấu hình được)

`TX.clean_text(text, lowercase, remove_stopwords, min_len)`: lowercase → bỏ URL/email → tokenization bằng regex → bỏ token ngắn → (tuỳ chọn) bỏ stopwords.
Padding chỉ áp dụng khi dùng BERT (tokenizer tự padding theo batch)."""),
    code("""print(TX.clean_text(train_texts[3])[:300]); print('...'); print(TX.clean_text(train_texts[3], remove_stopwords=True)[:300])
clean_train = TX.preprocess(train_texts, remove_stopwords=False)
clean_test = TX.preprocess(test_texts, remove_stopwords=False)
clean_train_sw = TX.preprocess(train_texts, remove_stopwords=True)
clean_test_sw = TX.preprocess(test_texts, remove_stopwords=True)"""),
    md("""## 4. Pipeline truyền thống: BoW / TF-IDF / n-gram

Thử 3 biểu diễn × 2 chế độ stopwords × 4 bộ phân loại (Multinomial NB, Complement NB, Logistic Regression, Linear SVM)."""),
    code("""log = ResultLogger('text_traditional')
feat_cfgs = [('bow', (1, 1)), ('tfidf', (1, 1)), ('tfidf', (1, 2))]
for sw_name, (tr, te) in {'giữ stopwords': (clean_train, clean_test), 'bỏ stopwords': (clean_train_sw, clean_test_sw)}.items():
    for kind, ng in feat_cfgs:
        vec = TX.make_vectorizer(kind, ng, max_features=50000, min_df=2)
        Xtr = vec.fit_transform(tr); Xte = vec.transform(te)
        for clf_name in ['nb', 'cnb', 'logreg', 'linsvm']:
            r, _ = TX.fit_eval(TX.make_classifier(clf_name), Xtr, y_train, Xte, y_test)
            log.add(features=f'{kind}{ng}', stopwords=sw_name, classifier=clf_name, dim=Xtr.shape[1], **r)
            print(f"{kind}{ng} | {sw_name} | {clf_name:<7} acc={r['accuracy']:.4f} f1={r['f1']:.4f}")
df_trad = log.save()"""),
    code("""pv = df_trad.assign(cfg=df_trad.features + ' | ' + df_trad.stopwords).pivot(index='cfg', columns='classifier', values='f1')
import seaborn as sns
plt.figure(figsize=(7, 4)); sns.heatmap(pv, annot=True, fmt='.3f', cmap='YlGnBu'); plt.title('Macro-F1 — đặc trưng truyền thống')
save_fig('text_trad_heatmap')
df_trad.sort_values('f1', ascending=False).head(8)"""),
    md("""## 5. Pipeline hiện đại: embedding học sâu → lưu `.npy`

Hai kiểu embedding:
1. **GloVe** (`glove-wiki-gigaword-100`, huấn luyện sẵn): trung bình vector các từ → 100 chiều.
2. **DistilBERT** (`distilbert-base-uncased`): embedding ngữ cảnh, mean pooling theo attention mask → 768 chiều.

Embedding được lưu vào `features/text_*_{train,test}.npy` và dùng làm đầu vào bộ phân loại."""),
    code("glove_tr, glove_te = TX.extract_and_save('glove100', TX.glove_embeddings, train_texts, test_texts)"),
    code("bert_tr, bert_te = TX.extract_and_save('distilbert', TX.bert_embeddings, train_texts, test_texts, max_len=256, batch_size=64)\nsave_features('text_y_train', np.asarray(y_train)); save_features('text_y_test', np.asarray(y_test))"),
    code("""# Nạp lại từ file .npy để chứng minh quy trình tái sử dụng đặc trưng
emb = {
    'glove100': (load_features('text_glove100_train'), load_features('text_glove100_test')),
    'distilbert': (load_features('text_distilbert_train'), load_features('text_distilbert_test')),
}
log2 = ResultLogger('text_embeddings')
for name, (Xtr, Xte) in emb.items():
    for clf_name in ['gnb', 'logreg', 'linsvm']:
        clf = TX.make_classifier(clf_name, C=1.0) if clf_name != 'linsvm' else TX.make_classifier('linsvm', C=0.1)
        if clf_name != 'gnb':
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            clf = make_pipeline(StandardScaler(), clf)
        r, _ = TX.fit_eval(clf, Xtr, y_train, Xte, y_test)
        log2.add(features=name, classifier=clf_name, dim=Xtr.shape[1], **r)
        print(f"{name} | {clf_name:<7} acc={r['accuracy']:.4f} f1={r['f1']:.4f}")
df_emb = log2.save(); df_emb"""),
    md("""## 6. (Điểm cộng) Pipeline học sâu

**6a.** MLP (PyTorch) trên embedding DistilBERT đã lưu.
**6b.** **Fine-tune đầu-cuối DistilBERT** (3 epoch, `max_len=192`) — toàn bộ mạng được tối ưu cùng lúc bằng gradient descent."""),
    code("""log3 = ResultLogger('text_deep')
for name in ['glove100', 'distilbert']:
    Xtr, Xte = emb[name]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    r, _, _ = dl.train_mlp((Xtr - mu) / sd, y_train, (Xte - mu) / sd, y_test, hidden=(512, 256), dropout=0.4, verbose=False)
    log3.add(method=f'MLP trên {name}', **r); print(f'MLP trên {name}:', {k: round(v, 4) for k, v in r.items()})"""),
    code("""import torch
if torch.cuda.is_available():
    r_ft, pred_ft = TX.finetune_bert(train_texts, y_train, test_texts, y_test, n_classes=len(class_names), epochs=3, max_len=192)
else:
    # Không có GPU: fine-tune trên tập con nhỏ để notebook vẫn chạy (kết quả sẽ thấp hơn)
    print('Không có GPU -> fine-tune trên 2000 mẫu train, 1 epoch (chỉ để minh hoạ). Hãy bật GPU để có kết quả đầy đủ.')
    rng = np.random.default_rng(0); sub = rng.choice(len(train_texts), min(2000, len(train_texts)), replace=False)
    r_ft, pred_ft = TX.finetune_bert([train_texts[i] for i in sub], np.asarray(y_train)[sub], test_texts[:1000], np.asarray(y_test)[:1000],
                                     n_classes=len(class_names), epochs=1, max_len=128)
log3.add(method='Fine-tune DistilBERT', **r_ft)
print({k: round(v, 4) for k, v in r_ft.items()})
df_deep = log3.save(); df_deep"""),
    md("## 7. Tổng hợp & phân tích"),
    code("""best_trad = df_trad.sort_values('f1', ascending=False).iloc[0]
best_emb = df_emb.sort_values('f1', ascending=False).iloc[0]
summary = pd.DataFrame([
    {'approach': 'Truyền thống', 'method': f"{best_trad['features']} + {best_trad['classifier']} ({best_trad['stopwords']})", 'accuracy': best_trad['accuracy'], 'f1': best_trad['f1']},
    {'approach': 'Embedding + ML cổ điển', 'method': f"{best_emb['features']} + {best_emb['classifier']}", 'accuracy': best_emb['accuracy'], 'f1': best_emb['f1']},
] + [{'approach': 'Học sâu', 'method': r['method'], 'accuracy': r['accuracy'], 'f1': r['f1']} for _, r in df_deep.iterrows()])
summary.to_csv('reports/results_text_summary.csv', index=False)
plt.figure(figsize=(9, 4)); plt.barh(summary['method'], summary['f1'], color=plt.cm.viridis(np.linspace(0.1, 0.9, len(summary))))
plt.xlabel('Macro-F1'); plt.title('So sánh các cách tiếp cận (20 Newsgroups)'); plt.xlim(0, 1)
for i, v in enumerate(summary['f1']): plt.text(v + 0.005, i, f'{v:.3f}', va='center')
save_fig('text_summary_bar'); summary"""),
    code("""# Phân tích lỗi của mô hình truyền thống tốt nhất
sw = best_trad['stopwords'] == 'bỏ stopwords'
tr, te = (clean_train_sw, clean_test_sw) if sw else (clean_train, clean_test)
kind, ng = ('bow', (1, 1)) if best_trad['features'].startswith('bow') else ('tfidf', (1, 2) if '(1, 2)' in best_trad['features'] else (1, 1))
vec = TX.make_vectorizer(kind, ng, max_features=50000, min_df=2)
Xtr = vec.fit_transform(tr); Xte = vec.transform(te)
r, pred = TX.fit_eval(TX.make_classifier(best_trad['classifier']), Xtr, y_train, Xte, y_test)
plot_confusion(y_test, pred, labels=class_names, title='Confusion matrix — mô hình truyền thống tốt nhất', name='text_confusion_best')
from sklearn.metrics import classification_report
rep = pd.DataFrame(classification_report(y_test, pred, target_names=class_names, output_dict=True)).T.iloc[:-3]
print(rep.sort_values('f1-score').head(6).round(3))
cm = pd.crosstab(pd.Series(np.array(class_names)[y_test], name='thực tế'), pd.Series(np.array(class_names)[pred], name='dự đoán'))
cm = cm.reindex(index=class_names, columns=class_names, fill_value=0)
pairs = [(a, b, cm.loc[a, b]) for a in cm.index for b in cm.columns if a != b]
print('\\nCặp nhầm lẫn nhiều nhất:'); [print(f'  {a} -> {b}: {n}') for a, b, n in sorted(pairs, key=lambda x: -x[2])[:6]]"""),
    md("""## 8. Kết luận *(chỉnh theo kết quả thực tế)*

- **Truyền thống:** TF-IDF (unigram/bigram) + Linear SVM/Logistic Regression thường rất mạnh trên 20 Newsgroups; Naive Bayes nhanh nhưng thấp hơn.
- **Bỏ stopwords / n-gram:** *…*
- **Embedding GloVe (trung bình):** mất thông tin thứ tự và từ hiếm ⇒ thường *thấp hơn* TF-IDF trên bài toán phân loại chủ đề.
- **DistilBERT (đóng băng):** *…*; **fine-tune** thường cho kết quả cao nhất nhưng tốn tài nguyên GPU.
- **Nhầm lẫn:** các nhóm gần nghĩa (`comp.*`, `talk.politics.*`, `religion`/`atheism`) bị lẫn với nhau.
- **Hạn chế:** cắt văn bản ở 192–256 token, chưa tinh chỉnh siêu tham số quy mô lớn, một lần chia train/test cố định.

Đặc trưng embedding đã lưu ở `features/text_*.npy`; bảng kết quả ở `reports/results_text_*.csv`."""),
]

if __name__ == "__main__":
    print(build("02_text_20newsgroups.ipynb", "Bài 2 — Học máy với dữ liệu văn bản (20 Newsgroups)", cells, None))
