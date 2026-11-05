from nbutil import build, code, md, setup_cells

cells = setup_cells(
    "!pip -q install seaborn scikit-learn",
    ["common.py", "tabular.py", "dl.py"],
    "import tabular as T\nimport dl\n",
)

cells += [
    md("""## 1. Bài toán & dữ liệu

- **Bài toán:** dự đoán thu nhập của một người trưởng thành (`>50K` hay `<=50K`) từ thông tin nhân khẩu học — phân loại nhị phân.
- **Dữ liệu:** *Adult Census Income* (UCI Machine Learning Repository) — https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- **Vì sao phù hợp ràng buộc của đề:** ~32.5k mẫu (đủ lớn), có **missing value** (`workclass`, `occupation`, `native-country`) và **8 cột categorical**.

> ⚠️ Đề yêu cầu các nhóm không trùng tập dữ liệu và nên thống nhất dataset với giảng viên trước. Nếu nhóm đổi dataset, chỉ cần sửa `load_adult()` trong `modules/tabular.py`."""),
    code("df = T.load_adult('data')\ndf.head()"),
    md("## 2. EDA (Exploratory Data Analysis)"),
    code("T.run_eda(df)"),
    md("""**Nhận xét EDA** *(chỉnh sửa theo kết quả thực tế sau khi chạy)*:
- Nhãn **mất cân bằng** (~24% là `>50K`) ⇒ accuracy không đủ, cần theo dõi F1/recall của lớp dương.
- Missing value tập trung ở 3 cột phân loại (~2–6%) ⇒ dùng imputation theo cột (mode hoặc hằng số `Missing`).
- `capital-gain`, `capital-loss` lệch phải mạnh, nhiều giá trị 0 và outlier ⇒ ảnh hưởng đến mô hình tuyến tính/khoảng cách, cần chuẩn hoá.
- `education` và `education-num` mang cùng thông tin (tương quan cao) ⇒ dư thừa, PCA có thể giúp.
- Tỉ lệ `>50K` khác biệt rõ theo `marital-status`, `relationship`, `education`, `occupation` ⇒ các biến phân loại có sức dự báo."""),
    md("## 3. Chia train/test & thiết kế pipeline"),
    code("""X_train, X_test, y_train, y_test = T.split_data(df, test_size=0.2, seed=42)
print(X_train.shape, X_test.shape, y_train.mean().round(3), y_test.mean().round(3))
num_cols, cat_cols = T.feature_types(X_train)
print('Số:', num_cols); print('Phân loại:', cat_cols)"""),
    md("""Pipeline được xây bằng `sklearn.Pipeline` + `ColumnTransformer` và **cấu hình được từng bước** qua một dict:

| Bước | Tuỳ chọn |
|---|---|
| Imputation (số) | `mean`, `median` |
| Imputation (phân loại) | `most_frequent`, `constant` ("Missing") |
| Encoding | `onehot`, `ordinal` |
| Scaling | `standard`, `minmax` (+ `feature_range`), `none` |
| Giảm chiều | `None`, PCA giữ 90% / 95% phương sai |
| Mô hình | Logistic Regression, Linear SVM, Random Forest |

Fit toàn bộ bước tiền xử lý **chỉ trên tập train** (trong Pipeline) để tránh rò rỉ dữ liệu."""),
    code("""cfg = dict(scaler='minmax', feature_range=(0, 1), pca=0.95, model='logreg')
pipe = T.build_pipeline(X_train, **cfg)
pipe"""),
    code("""res, pipe = T.run_config(X_train, X_test, y_train, y_test, **cfg)
print(res)"""),
    md("""## 4. Thí nghiệm 1 — Scaler × PCA × Mô hình (27 cấu hình)

So sánh `StandardScaler`, `MinMaxScaler(0,1)`, `MinMaxScaler(-1,1)`; PCA `None / 90% / 95%`; ba mô hình."""),
    code("""grid = T.main_grid()
print(len(grid), 'cấu hình')
df_main = T.run_grid(X_train, X_test, y_train, y_test, grid, name='tabular_main')"""),
    code("""cols = ['model', 'scaler', 'feature_range', 'pca', 'n_features', 'accuracy', 'precision', 'recall', 'f1', 'fit_seconds']
df_main[cols].sort_values('f1', ascending=False).reset_index(drop=True)"""),
    code("""d = df_main.copy()
d['scaler_cfg'] = d['scaler'] + np.where(d['scaler'] == 'minmax', d['feature_range'].fillna(''), '')
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, m in zip(axes, ['logreg', 'linsvm', 'rf']):
    pv = d[d.model == m].pivot(index='scaler_cfg', columns='pca', values='f1')
    pv = pv[['None', '0.9', '0.95']] if set(['None', '0.9', '0.95']) <= set(pv.columns) else pv
    import seaborn as sns
    sns.heatmap(pv, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar=False)
    ax.set_title(f'F1 (lớp >50K) — {m}'); ax.set_xlabel('PCA'); ax.set_ylabel('')
save_fig('tab_heatmap_scaler_pca')"""),
    md("## 5. Thí nghiệm 2 — Imputation × Encoding"),
    code("""grid2 = []
for m in ['logreg', 'rf']:
    for imp in ['most_frequent', 'constant']:
        for enc in ['onehot', 'ordinal']:
            grid2.append(dict(model=m, cat_imputer=imp, encoding=enc, scaler='standard', pca=None))
df_enc = T.run_grid(X_train, X_test, y_train, y_test, grid2, name='tabular_imputer_encoding')
df_enc[['model', 'cat_imputer', 'encoding', 'n_features', 'accuracy', 'f1', 'fit_seconds']]"""),
    md("## 6. Thí nghiệm 3 — Mất cân bằng lớp & siêu tham số"),
    code("""grid3 = [
    dict(model='logreg', scaler='standard', model_params={}),
    dict(model='logreg', scaler='standard', model_params={'class_weight': 'balanced'}),
    dict(model='logreg', scaler='standard', model_params={'C': 0.1}),
    dict(model='logreg', scaler='standard', model_params={'C': 10}),
    dict(model='linsvm', scaler='standard', model_params={'C': 0.1}),
    dict(model='linsvm', scaler='standard', model_params={'C': 1}),
    dict(model='rf', scaler='none', model_params={}),
    dict(model='rf', scaler='none', model_params={'class_weight': 'balanced'}),
    dict(model='rf', scaler='none', model_params={'max_depth': 12, 'n_estimators': 300}),
    dict(model='rf', scaler='none', model_params={'min_samples_leaf': 5, 'n_estimators': 300}),
]
rows = []
for g in grid3:
    r, _ = T.run_config(X_train, X_test, y_train, y_test, **g)
    rows.append({'model': g['model'], 'params': str(g['model_params']), **r})
df_hp = pd.DataFrame(rows)
df_hp.to_csv('reports/results_tabular_hparams.csv', index=False)
df_hp[['model', 'params', 'accuracy', 'precision', 'recall', 'f1']]"""),
    md("## 7. Cấu hình tốt nhất: kiểm định chéo & phân tích lỗi"),
    code("""from sklearn.model_selection import cross_val_score, StratifiedKFold
allres = pd.concat([df_main, df_enc], ignore_index=True)
best = df_main.sort_values('f1', ascending=False).iloc[0]
print('Cấu hình tốt nhất (thí nghiệm 1):'); print(best[['model','scaler','feature_range','pca','accuracy','f1']].to_dict())
best_cfg = dict(model=best['model'], scaler=best['scaler'],
                pca=None if best['pca'] == 'None' else float(best['pca']))
if best['scaler'] == 'minmax':
    best_cfg['feature_range'] = eval(best['feature_range'])
pipe_best = T.build_pipeline(X_train, **best_cfg)
cv = cross_val_score(pipe_best, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1')
print(f'5-fold CV F1 = {cv.mean():.4f} ± {cv.std():.4f}')
pipe_best.fit(X_train, y_train)
pred = pipe_best.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred, target_names=['<=50K', '>50K']))
plot_confusion(y_test, pred, labels=['<=50K', '>50K'], title='Confusion matrix — cấu hình tốt nhất', name='tab_confusion_best')"""),
    code("""# Độ quan trọng đặc trưng (nếu mô hình tốt nhất là Random Forest) hoặc hệ số (tuyến tính)
pre = pipe_best[:-1]
names = pre.named_steps['pre'].get_feature_names_out() if 'pca' not in pipe_best.named_steps else None
mdl = pipe_best.named_steps['model']
if names is not None:
    imp = mdl.feature_importances_ if hasattr(mdl, 'feature_importances_') else np.abs(mdl.coef_).ravel()
    top = pd.Series(imp, index=names).sort_values(ascending=False).head(15)
    plt.figure(figsize=(7, 5)); top[::-1].plot.barh(); plt.title('Top 15 đặc trưng quan trọng')
    save_fig('tab_feature_importance')
else:
    print('Mô hình tốt nhất dùng PCA nên bỏ qua feature importance trên đặc trưng gốc.')"""),
    md("""## 8. (Điểm cộng) Pipeline học sâu — MLP trên đặc trưng đã tiền xử lý

Dùng cùng khâu tiền xử lý (impute + one-hot + standard scale) rồi huấn luyện **MLP PyTorch** (2 lớp ẩn, BatchNorm, Dropout, early stopping), so sánh với pipeline truyền thống."""),
    code("""pre = T.build_pipeline(X_train, scaler='standard')[:-1]
Xtr_p = pre.fit_transform(X_train); Xte_p = pre.transform(X_test)
save_features('tab_preprocessed_train', Xtr_p.astype('float32')); save_features('tab_preprocessed_test', Xte_p.astype('float32'))
save_features('tab_y_train', y_train.values); save_features('tab_y_test', y_test.values)
print(Xtr_p.shape, Xte_p.shape)

dl_rows = []
for hidden, dropout in [((128, 64), 0.2), ((256, 128), 0.3), ((256, 128, 64), 0.3)]:
    r, _, _ = dl.train_mlp(Xtr_p, y_train.values, Xte_p, y_test.values, hidden=hidden, dropout=dropout, average='binary', verbose=False)
    dl_rows.append({'model': f'MLP{hidden}', 'dropout': dropout, **r})
    print(dl_rows[-1])
df_dl = pd.DataFrame(dl_rows); df_dl.to_csv('reports/results_tabular_dl.csv', index=False); df_dl"""),
    md("## 9. Tổng hợp & so sánh"),
    code("""summary = pd.concat([
    df_main.sort_values('f1', ascending=False).groupby('model').head(1).assign(group='Truyền thống (tốt nhất mỗi mô hình)')[['group', 'model', 'accuracy', 'precision', 'recall', 'f1', 'fit_seconds']],
    df_dl.sort_values('f1', ascending=False).head(1).assign(group='Học sâu (MLP tốt nhất)')[['group', 'model', 'accuracy', 'precision', 'recall', 'f1', 'fit_seconds']],
], ignore_index=True)
summary.to_csv('reports/results_tabular_summary.csv', index=False)
summary"""),
    code("""ax = summary.set_index('model')[['accuracy', 'precision', 'recall', 'f1']].plot.bar(figsize=(9, 4), rot=20)
ax.set_ylim(0.5, 1); ax.set_title('So sánh các mô hình (Adult Income)'); ax.legend(loc='lower right')
save_fig('tab_summary_bar')"""),
    md("""## 10. Kết luận *(chỉnh theo kết quả thực tế)*

- **Scaler:** *…* (ví dụ: Random Forest không phụ thuộc scaler; Logistic Regression/SVM cải thiện nhẹ khi chuẩn hoá).
- **PCA:** *…* (ví dụ: giữ 95% phương sai giảm số chiều nhưng không cải thiện, thậm chí giảm F1 do mất thông tin từ biến one-hot).
- **Imputation/Encoding:** *…*
- **Mất cân bằng lớp:** `class_weight='balanced'` tăng recall lớp `>50K` nhưng giảm precision.
- **Học sâu vs truyền thống:** *…* (dữ liệu bảng cỡ vừa thường để mô hình cây/ensemble ngang hoặc hơn MLP).
- **Hạn chế:** chưa dùng gradient boosting, chưa tinh chỉnh bằng grid/random search quy mô lớn; chỉ một lần chia train/test.

Các kết quả đều được lưu vào `reports/results_tabular_*.csv`, hình vào `reports/figures/`, đặc trưng đã tiền xử lý vào `features/`."""),
]

if __name__ == "__main__":
    print(build("01_tabular_adult_income.ipynb", "Bài 1 — Học máy với dữ liệu dạng bảng (Adult Income)", cells, None))
