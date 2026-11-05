from nbutil import build, code, md, setup_cells

cells = setup_cells(
    "!pip -q install scikit-image scikit-learn seaborn h5py",
    ["common.py", "image.py", "dl.py"],
    "import image as IM\nimport dl\nimport torch\nprint('GPU:', torch.cuda.is_available())\n",
)

cells += [
    md("""## 1. Bài toán & dữ liệu

- **Bài toán:** phân loại ảnh thú cưng thành **37 giống** (25 giống chó, 12 giống mèo) — bài toán *fine-grained*.
- **Dữ liệu:** *Oxford-IIIT Pet* — https://www.robots.ox.ac.uk/~vgg/data/pets/ (tải qua `torchvision.datasets.OxfordIIITPet`).
  Tập train = `trainval` (3,680 ảnh), tập test = `test` (3,669 ảnh), ~200 ảnh/lớp, ảnh màu kích thước và tỉ lệ khác nhau.
- **Vì sao phù hợp:** đủ lớn và đa dạng, chạy được trên Colab (ảnh JPEG, ~800MB), có các giống rất giống nhau ⇒ phân tích lỗi thú vị.

> ⚠️ Đề yêu cầu các nhóm không trùng tập dữ liệu và cần thống nhất với giảng viên. Đổi dataset: sửa `load_pets()` trong `modules/image.py`."""),
    code("train_ds, test_ds, class_names = IM.load_pets('data')\ny_train, y_test = IM.labels_of(train_ds), IM.labels_of(test_ds)\nprint(len(train_ds), len(test_ds), len(class_names))"),
    md("## 2. EDA ảnh"),
    code("IM.run_eda(train_ds, test_ds, class_names)"),
    md("""**Nhận xét EDA** *(chỉnh theo kết quả thực tế)*:
- Số ảnh/lớp khá cân bằng (~100 ảnh/lớp ở train), không cần cân bằng lại.
- Ảnh đều là **RGB 3 kênh** (có thể có vài ảnh ở chế độ khác ⇒ luôn `convert('RGB')`).
- **Kích thước và tỉ lệ khung hình không đồng nhất** (chiều rộng ~200–4000 px) ⇒ bắt buộc **resize** về kích thước cố định trước khi trích đặc trưng.
- Các giống như *Staffordshire Bull Terrier* / *American Pit Bull Terrier* hay *Egyptian Mau* / *Bengal* rất giống nhau ⇒ dự kiến nhầm lẫn."""),
    md("""## 3. Pipeline truyền thống: HOG → (PCA) → bộ phân loại

HOG (Histogram of Oriented Gradients) trên ảnh xám resize `128×128`, `pixels_per_cell=16`, sau đó PCA giữ 95% phương sai."""),
    code("""with Timer() as t:
    hog_tr = IM.hog_features(train_ds, size=128, pixels_per_cell=16)
    hog_te = IM.hog_features(test_ds, size=128, pixels_per_cell=16)
print(hog_tr.shape, hog_te.shape, f'{t.elapsed:.0f}s')
save_features('img_hog128_train', hog_tr); save_features('img_hog128_test', hog_te)
save_features('img_y_train', y_train); save_features('img_y_test', y_test)"""),
    code("""log = ResultLogger('image_traditional')
hog_pca_tr, hog_pca_te, pca = IM.pca_reduce(hog_tr, hog_te, 0.95)
print('PCA 95%:', hog_tr.shape[1], '->', hog_pca_tr.shape[1], 'chiều')
for feat_name, (Xtr, Xte) in {'HOG': (hog_tr, hog_te), 'HOG+PCA95': (hog_pca_tr, hog_pca_te)}.items():
    for clf_name in ['logreg', 'linsvm', 'svm_rbf', 'rf']:
        r, _ = IM.fit_eval(IM.make_classifier(clf_name), Xtr, y_train, Xte, y_test)
        log.add(features=feat_name, classifier=clf_name, dim=Xtr.shape[1], **r)
        print(f"{feat_name:<10} {clf_name:<8} acc={r['accuracy']:.4f} f1={r['f1']:.4f} ({r['fit_seconds']}s)")
df_hog = log.save()"""),
    md("""## 4. Pipeline hiện đại: đặc trưng từ mô hình pretrained → lưu `.npy`

Dùng backbone **pretrained trên ImageNet** (bỏ lớp phân loại cuối) để lấy vector đặc trưng:

| Mô hình | Loại | Dim |
|---|---|---|
| ResNet-50 | CNN | 2048 |
| EfficientNet-B0 | CNN | 1280 |
| ViT-B/16 | Transformer | 768 |

Cấu hình được: kích thước resize (`img_size`) và tên mô hình. Ngoài ra thử **ResNet-50 ở 160×160** để xem ảnh hưởng của độ phân giải (ViT-B/16 cố định 224)."""),
    code("""configs = [('resnet50', 224), ('efficientnet_b0', 224), ('vit_b_16', 224), ('resnet50', 160)]
feats = {}
for name, size in configs:
    feats[(name, size)] = IM.extract_and_save(train_ds, test_ds, name, size)"""),
    code("""# Nạp lại đặc trưng từ file .npy và huấn luyện các bộ phân loại
log2 = ResultLogger('image_deep_features')
for name, size in configs:
    tag = f'{name}_{size}'
    Xtr, Xte = load_features(f'img_{tag}_train'), load_features(f'img_{tag}_test')
    for clf_name in ['logreg', 'linsvm', 'svm_rbf', 'rf']:
        r, _ = IM.fit_eval(IM.make_classifier(clf_name), Xtr, y_train, Xte, y_test)
        log2.add(backbone=name, img_size=size, classifier=clf_name, dim=Xtr.shape[1], **r)
        print(f"{tag:<22} {clf_name:<8} acc={r['accuracy']:.4f} f1={r['f1']:.4f} ({r['fit_seconds']}s)")
df_deepfeat = log2.save()"""),
    code("""import seaborn as sns
pv = df_deepfeat.assign(cfg=df_deepfeat.backbone + '_' + df_deepfeat.img_size.astype(str)).pivot(index='cfg', columns='classifier', values='accuracy')
plt.figure(figsize=(7, 3.5)); sns.heatmap(pv, annot=True, fmt='.3f', cmap='YlGnBu'); plt.title('Accuracy — backbone × bộ phân loại')
save_fig('img_deepfeat_heatmap')"""),
    md("## 5. (Điểm cộng) Pipeline học sâu đầu-cuối\n\n**5a.** MLP PyTorch trên đặc trưng ViT/ResNet đã lưu.  \n**5b.** **Fine-tune toàn bộ ResNet-50** (5 epoch, augmentation, label smoothing, OneCycle)."),
    code("""log3 = ResultLogger('image_deep')
for name, size in [('resnet50', 224), ('vit_b_16', 224)]:
    tag = f'{name}_{size}'
    Xtr, Xte = load_features(f'img_{tag}_train'), load_features(f'img_{tag}_test')
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    r, _, _ = dl.train_mlp((Xtr - mu) / sd, y_train, (Xte - mu) / sd, y_test, hidden=(512,), dropout=0.4, verbose=False)
    log3.add(method=f'MLP trên {tag}', **r); print(f'MLP trên {tag}:', {k: round(v, 4) for k, v in r.items()})"""),
    code("""if torch.cuda.is_available():
    r_ft, pred_ft = IM.finetune_cnn(train_ds, test_ds, n_classes=len(class_names), arch='resnet50', epochs=5)
else:
    print('Không có GPU -> fine-tune ResNet-18 1 epoch để minh hoạ. Hãy bật GPU (T4) để có kết quả đầy đủ.')
    r_ft, pred_ft = IM.finetune_cnn(train_ds, test_ds, n_classes=len(class_names), arch='resnet18', epochs=1, img_size=128)
log3.add(method='Fine-tune ResNet-50 (end-to-end)', **r_ft)
print({k: round(v, 4) for k, v in r_ft.items()})
df_deep = log3.save(); df_deep"""),
    md("## 6. Tổng hợp & phân tích"),
    code("""best_hog = df_hog.sort_values('accuracy', ascending=False).iloc[0]
best_df = df_deepfeat.sort_values('accuracy', ascending=False).iloc[0]
summary = pd.DataFrame([
    {'approach': 'Truyền thống', 'method': f"{best_hog['features']} + {best_hog['classifier']}", 'accuracy': best_hog['accuracy'], 'f1': best_hog['f1']},
    {'approach': 'Pretrained features + ML cổ điển', 'method': f"{best_df['backbone']}_{best_df['img_size']} + {best_df['classifier']}", 'accuracy': best_df['accuracy'], 'f1': best_df['f1']},
] + [{'approach': 'Học sâu', 'method': r['method'], 'accuracy': r['accuracy'], 'f1': r['f1']} for _, r in df_deep.iterrows()])
summary.to_csv('reports/results_image_summary.csv', index=False)
plt.figure(figsize=(9, 3.8)); plt.barh(summary['method'], summary['accuracy'], color=plt.cm.viridis(np.linspace(0.1, 0.9, len(summary))))
plt.xlim(0, 1); plt.xlabel('Accuracy'); plt.title('So sánh các cách tiếp cận (Oxford-IIIT Pet)')
for i, v in enumerate(summary['accuracy']): plt.text(v + 0.005, i, f'{v:.3f}', va='center')
save_fig('img_summary_bar'); summary"""),
    code("""# Phân tích lỗi: mô hình pretrained-features tốt nhất
tag = f"{best_df['backbone']}_{best_df['img_size']}"
Xtr, Xte = load_features(f'img_{tag}_train'), load_features(f'img_{tag}_test')
r, pred = IM.fit_eval(IM.make_classifier(best_df['classifier']), Xtr, y_train, Xte, y_test)
plot_confusion(y_test, pred, labels=class_names, title=f'Confusion matrix — {tag} + {best_df["classifier"]}', name='img_confusion_best')
cm = pd.crosstab(pd.Series(np.array(class_names)[y_test], name='thực tế'), pd.Series(np.array(class_names)[pred], name='dự đoán'))
cm = cm.reindex(index=class_names, columns=class_names, fill_value=0)
pairs = [(a, b, cm.loc[a, b]) for a in cm.index for b in cm.columns if a != b]
print('Cặp giống bị nhầm nhiều nhất:')
for a, b, n in sorted(pairs, key=lambda x: -x[2])[:8]: print(f'  {a} -> {b}: {n}')
wrong = np.where(pred != y_test)[0][:12]
fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
from PIL import Image
for ax, i in zip(axes.ravel(), wrong):
    ax.imshow(Image.open(test_ds._images[i]).convert('RGB')); ax.axis('off')
    ax.set_title(f'T: {class_names[y_test[i]]}\\nP: {class_names[pred[i]]}', fontsize=7)
save_fig('img_errors')"""),
    md("""## 7. Kết luận *(chỉnh theo kết quả thực tế)*

- **HOG + ML cổ điển** cho kết quả thấp (đặc trưng cạnh/hình dạng cục bộ không đủ cho bài toán fine-grained 37 lớp); PCA giảm chiều nhưng không cải thiện nhiều.
- **Đặc trưng pretrained** cải thiện **rất mạnh** (chuyển giao tri thức từ ImageNet); mô hình tuyến tính (Logistic Regression/Linear SVM) trên đặc trưng chuẩn hoá thường tốt nhất, Random Forest kém trên vector chiều cao.
- **ViT-B/16 vs ResNet-50 vs EfficientNet-B0:** *…*; **độ phân giải 160 vs 224:** *…*
- **Fine-tune end-to-end** thường cao nhất nhưng tốn thời gian GPU hơn nhiều so với trích đặc trưng một lần.
- **Lỗi** tập trung ở các giống nhìn rất giống nhau.
- **Hạn chế:** không dùng test-time augmentation, chưa thử SIFT/bag-of-visual-words, chỉ một lần chia train/test cố định.

Đặc trưng đã lưu ở `features/img_*.npy`; bảng kết quả ở `reports/results_image_*.csv`."""),
]

if __name__ == "__main__":
    print(build("03_image_oxford_pets.ipynb", "Bài 3 — Học máy với dữ liệu ảnh (Oxford-IIIT Pet)", cells, None))
