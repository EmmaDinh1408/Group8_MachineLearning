"""Sinh báo cáo PDF từ kết quả thật mà các notebook đã lưu (reports/results_*.csv, reports/figures/*.png).

Chạy sau khi đã chạy các notebook:
    python modules/make_report.py                # tạo reports/BaoCao_BTL_HocMay.pdf
Sửa dict GROUP và danh sách PHAN_CONG bên dưới (thông tin nhóm, bảng phân công, tỉ lệ đóng góp).
Phần "Nhận xét/phân tích" tự động chỉ nêu số liệu; nhóm cần bổ sung phần giải thích của mình trong
các hộp NHAN_XET_TAY ở cuối từng bài.
"""
import os
import sys

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures")
OUT = os.path.join(REPORT_DIR, "BaoCao_BTL_HocMay.pdf")

# ----------------------------------------------------------------------------
# >>> NHÓM ĐIỀN THÔNG TIN Ở ĐÂY <<<
# ----------------------------------------------------------------------------
GROUP = {
    "Tên nhóm": "[Tên nhóm]",
    "GVHD": "[Họ tên giảng viên]",
    "GitHub": "[link repo GitHub]",
    "Colab": "[link Colab notebook]",
}
PHAN_CONG = [  # (Họ tên, MSSV, Email, Công việc phụ trách, Tỉ lệ đóng góp %)
    ("[Thành viên 1]", "[MSSV]", "[email]", "Bài 1 (Tabular), EDA, báo cáo", 34),
    ("[Thành viên 2]", "[MSSV]", "[email]", "Bài 2 (Text), embedding, README", 33),
    ("[Thành viên 3]", "[MSSV]", "[email]", "Bài 3 (Image), pretrained features, GitHub", 33),
]
NHAN_XET_TAY = {  # nhóm viết phân tích riêng của mình ở đây (hiển thị trong báo cáo)
    "tab": "[Nhóm bổ sung: giải thích vì sao scaler/PCA/mô hình cho kết quả như vậy, ưu điểm và hạn chế…]",
    "text": "[Nhóm bổ sung: giải thích kết quả TF-IDF vs embedding vs fine-tune, các lớp hay nhầm…]",
    "img": "[Nhóm bổ sung: giải thích kết quả HOG vs pretrained vs fine-tune, các giống hay nhầm…]",
}

# ----------------------------------------------------------------------------
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=15, leading=19, spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#1f3a5f"), alignment=0)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2c5282"), alignment=0)
CAP = ParagraphStyle("cap", parent=BODY, fontName="DV-I", fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.grey)
TITLE = ParagraphStyle("title", parent=BODY, fontName="DV-B", fontSize=22, leading=28, alignment=TA_CENTER, textColor=colors.HexColor("#1f3a5f"))
SUB = ParagraphStyle("sub", parent=BODY, fontSize=11, leading=16, alignment=TA_CENTER)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.5, leading=9, alignment=0, spaceAfter=0)
NOTE = ParagraphStyle("note", parent=BODY, backColor=colors.HexColor("#fff7e0"), borderPadding=6, borderColor=colors.HexColor("#e0b84c"), borderWidth=0.6, spaceBefore=6, spaceAfter=10)

story = []


def P(text, style=BODY):
    story.append(Paragraph(text, style))


def fig(name, width_cm=15, caption=None):
    path = os.path.join(FIG_DIR, f"{name}.png")
    if not os.path.exists(path):
        return
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    wd = width_cm * cm
    story.append(Image(path, width=wd, height=wd * h / w))
    if caption:
        P(caption, CAP)
    story.append(Spacer(1, 4))


def read(name):
    p = os.path.join(REPORT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def table(df, cols=None, max_rows=14, widths=None, floats=4):
    if df is None or df.empty:
        return
    df = df[cols] if cols else df
    df = df.head(max_rows)
    data = [[Paragraph(f"<b>{c}</b>", CELL) for c in df.columns]]
    for _, r in df.iterrows():
        row = []
        for v in r:
            row.append(Paragraph(f"{v:.{floats}f}" if isinstance(v, float) else str(v), CELL))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b8c4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def best_row(df, metric):
    return df.sort_values(metric, ascending=False).iloc[0]


# ----------------------------------------------------------------------------
def cover():
    story.append(Spacer(1, 3 * cm))
    P("TRƯỜNG ĐẠI HỌC BÁCH KHOA – ĐHQG-HCM<br/>KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH", SUB)
    story.append(Spacer(1, 1.5 * cm))
    P("BÁO CÁO BÀI TẬP LỚN", TITLE)
    P("MÔN HỌC: HỌC MÁY", TITLE)
    story.append(Spacer(1, 0.6 * cm))
    P("Pipeline học máy truyền thống và học sâu trên dữ liệu dạng bảng, văn bản và ảnh", SUB)
    story.append(Spacer(1, 1.5 * cm))
    P("Học kỳ I – Năm học 2026-2027", SUB)
    story.append(Spacer(1, 1 * cm))
    rows = [[Paragraph(f"<b>{k}</b>", CELL), Paragraph(v, CELL)] for k, v in GROUP.items()]
    for n, mssv, em, _, _ in PHAN_CONG:
        rows.append([Paragraph("<b>Thành viên</b>", CELL), Paragraph(f"{n} – {mssv} – {em}", CELL)])
    t = Table(rows, colWidths=[3.5 * cm, 11 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eef7"))]))
    story.append(t)
    story.append(PageBreak())


def intro():
    P("1. Giới thiệu và mục tiêu", H1)
    P("Báo cáo trình bày việc xây dựng và đánh giá các pipeline học máy trên ba loại dữ liệu: <b>bảng</b> (Adult Census Income), "
      "<b>văn bản</b> (20 Newsgroups) và <b>ảnh</b> (Oxford-IIIT Pet). Với mỗi loại dữ liệu, nhóm thực hiện đầy đủ pipeline truyền thống "
      "(EDA → tiền xử lý → trích xuất đặc trưng → huấn luyện → đánh giá) và mở rộng thêm pipeline học sâu để so sánh (điểm cộng).")
    P("Toàn bộ mã nguồn chạy trên Google Colab (Runtime → Run all), dữ liệu tải từ nguồn công khai, không mount dịch vụ cloud cá nhân. "
      "Các đặc trưng đã trích xuất được lưu dạng <font name='DV-B'>.npy</font> trong thư mục <font name='DV-B'>features/</font>.")
    P("2. Tổng quan pipeline", H1)
    P("<b>Pipeline truyền thống</b>: EDA; tiền xử lý (impute, encode, scale, tokenization); trích xuất đặc trưng thủ công "
      "(one-hot/z-score cho bảng; BoW, TF-IDF, n-gram cho văn bản; HOG và PCA cho ảnh); huấn luyện Logistic Regression, SVM, "
      "Random Forest, Naive Bayes; đánh giá bằng accuracy, precision, recall, F1.")
    P("<b>Pipeline học sâu</b>: biểu diễn được học tự động (MLP; embedding GloVe/DistilBERT; đặc trưng ResNet/EfficientNet/ViT pretrained), "
      "và huấn luyện đầu-cuối (fine-tune DistilBERT, ResNet-50).")
    P("Chỉ số: với dữ liệu bảng (nhị phân) báo cáo precision/recall/F1 của lớp dương (&gt;50K); với văn bản và ảnh (nhiều lớp) báo cáo macro-averaged "
      "precision/recall/F1 cùng accuracy. Tất cả mô hình đánh giá trên cùng một tập test cố định (seed = 42); tiền xử lý chỉ fit trên tập train để tránh rò rỉ dữ liệu.")


def section_tab():
    main = read("results_tabular_main.csv")
    if main is None:
        return
    P("3. Bài 1 – Dữ liệu dạng bảng (Adult Census Income)", H1)
    P("<b>Dữ liệu:</b> UCI Adult, ~32.5k mẫu, 14 thuộc tính (6 số, 8 phân loại), nhãn thu nhập &gt;50K / ≤50K. Có giá trị thiếu ở "
      "<i>workclass, occupation, native-country</i> và nhãn mất cân bằng (~24% lớp dương). Nguồn: archive.ics.uci.edu/ml/machine-learning-databases/adult.")
    P("3.1 Phân tích khám phá dữ liệu (EDA)", H2)
    fig("tab_missing", 8, "Hình 1. Tỉ lệ giá trị thiếu theo cột")
    fig("tab_num_hist", 15, "Hình 2. Phân phối các biến số")
    fig("tab_corr", 8, "Hình 3. Ma trận tương quan")
    fig("tab_cat_vs_target", 16, "Hình 4. Tỉ lệ nhãn &gt;50K theo từng biến phân loại")
    P("3.2 Pipeline và các tham số đã thử", H2)
    P("Pipeline sklearn <i>ColumnTransformer</i> + <i>Pipeline</i>: impute (số: median; phân loại: most_frequent hoặc hằng 'Missing') → "
      "encode (one-hot hoặc ordinal) → scale (Standard, MinMax(0,1), MinMax(−1,1)) → PCA (không / 90% / 95%) → mô hình (Logistic Regression, Linear SVM, "
      "Random Forest). Thí nghiệm 1: 27 cấu hình = 3 scaler × 3 mức PCA × 3 mô hình.")
    d = main.copy()
    cols = [c for c in ["model", "scaler", "feature_range", "pca", "n_features", "accuracy", "precision", "recall", "f1"] if c in d.columns]
    table(d.sort_values("f1", ascending=False), cols, max_rows=12)
    P("Hình 5 là bản đồ nhiệt F1 theo scaler × PCA cho từng mô hình.", BODY)
    fig("tab_heatmap_scaler_pca", 16, "Hình 5. F1 (lớp &gt;50K) theo scaler × PCA")
    b = best_row(main, "f1")
    by_model = main.groupby("model")["f1"].agg(["mean", "max"]).round(4)
    by_pca = main.groupby("pca")["f1"].mean().round(4)
    P(f"<b>Kết quả:</b> cấu hình tốt nhất là <b>{b['model']}</b> (scaler = {b['scaler']}, PCA = {b['pca']}) với accuracy = {b['accuracy']:.4f}, "
      f"F1 = {b['f1']:.4f}. F1 trung bình theo mô hình: " + "; ".join(f"{m} = {r['mean']:.4f} (max {r['max']:.4f})" for m, r in by_model.iterrows()) +
      ". F1 trung bình theo mức PCA: " + "; ".join(f"{k} = {v:.4f}" for k, v in by_pca.items()) + ".")
    enc = read("results_tabular_imputer_encoding.csv")
    if enc is not None:
        P("3.3 Imputation × Encoding", H2)
        table(enc, [c for c in ["model", "cat_imputer", "encoding", "n_features", "accuracy", "f1"] if c in enc.columns])
    hp = read("results_tabular_hparams.csv")
    if hp is not None:
        P("3.4 Mất cân bằng lớp và siêu tham số", H2)
        table(hp, ["model", "params", "accuracy", "precision", "recall", "f1"])
    fig("tab_confusion_best", 7, "Hình 6. Confusion matrix của cấu hình tốt nhất")
    fig("tab_feature_importance", 10, "Hình 7. Độ quan trọng đặc trưng")
    dlr = read("results_tabular_dl.csv")
    if dlr is not None:
        P("3.5 (Điểm cộng) Pipeline học sâu – MLP", H2)
        table(dlr, [c for c in ["model", "dropout", "accuracy", "precision", "recall", "f1", "fit_seconds"] if c in dlr.columns])
        bd = best_row(dlr, "f1")
        diff = bd["f1"] - b["f1"]
        P(f"MLP tốt nhất ({bd['model']}) đạt F1 = {bd['f1']:.4f}, " + ("cao hơn" if diff > 0 else "thấp hơn") + f" pipeline truyền thống tốt nhất {abs(diff):.4f}.")
    fig("tab_summary_bar", 11, "Hình 8. So sánh tổng hợp")
    P("3.6 Phân tích và kết luận", H2)
    P(NHAN_XET_TAY["tab"], NOTE)


def section_text():
    trad = read("results_text_traditional.csv")
    if trad is None:
        return
    story.append(PageBreak())
    P("4. Bài 2 – Dữ liệu văn bản (20 Newsgroups)", H1)
    P("<b>Dữ liệu:</b> 20 Newsgroups (qwone.com/~jason/20Newsgroups), 20 chủ đề, đã loại header/footer/quote. Hai cách trích xuất đặc trưng: "
      "<b>truyền thống</b> (BoW, TF-IDF, n-gram) và <b>hiện đại</b> (GloVe trung bình, DistilBERT mean pooling) – embedding lưu thành .npy.")
    P("4.1 EDA", H2)
    fig("text_class_dist", 14, "Hình 9. Số văn bản theo lớp")
    fig("text_length_dist", 14, "Hình 10. Phân phối độ dài văn bản")
    fig("text_top_words_stop", 11, "Hình 11. Từ phổ biến khi giữ stopwords")
    fig("text_top_words_nostop", 11, "Hình 12. Từ phổ biến sau khi bỏ stopwords")
    fig("text_zipf", 7, "Hình 13. Luật Zipf")
    P("4.2 Pipeline truyền thống", H2)
    P("Tiền xử lý: lowercase, bỏ URL/email, tokenization regex, (tuỳ chọn) bỏ stopwords. Đặc trưng: BoW, TF-IDF (1,1), TF-IDF (1,2) "
      "(50.000 từ, min_df = 2). Bộ phân loại: Multinomial NB, Complement NB, Logistic Regression, Linear SVM.")
    table(trad.sort_values("f1", ascending=False), ["features", "stopwords", "classifier", "dim", "accuracy", "precision", "recall", "f1"], max_rows=12)
    fig("text_trad_heatmap", 11, "Hình 14. Macro-F1 theo cấu hình đặc trưng × bộ phân loại")
    emb = read("results_text_embeddings.csv")
    if emb is not None:
        P("4.3 Embedding học sâu (đã lưu .npy)", H2)
        table(emb.sort_values("f1", ascending=False), ["features", "classifier", "dim", "accuracy", "precision", "recall", "f1"])
    deep = read("results_text_deep.csv")
    if deep is not None:
        P("4.4 (Điểm cộng) Pipeline học sâu", H2)
        table(deep, ["method", "accuracy", "precision", "recall", "f1", "fit_seconds"])
    summ = read("results_text_summary.csv")
    if summ is not None:
        P("4.5 So sánh truyền thống và hiện đại", H2)
        table(summ, ["approach", "method", "accuracy", "f1"])
        fig("text_summary_bar", 11, "Hình 15. So sánh các cách tiếp cận")
        bt = best_row(trad, "f1")
        P(f"<b>Kết quả:</b> mô hình truyền thống tốt nhất là {bt['features']} + {bt['classifier']} ({bt['stopwords']}) với macro-F1 = {bt['f1']:.4f}."
          + (f" Embedding + ML cổ điển tốt nhất: macro-F1 = {emb['f1'].max():.4f}." if emb is not None else "")
          + (f" Học sâu tốt nhất: {deep.sort_values('f1').iloc[-1]['method']} (macro-F1 = {deep['f1'].max():.4f})." if deep is not None else ""))
    fig("text_confusion_best", 12, "Hình 16. Confusion matrix của mô hình truyền thống tốt nhất")
    P("4.6 Phân tích và kết luận", H2)
    P(NHAN_XET_TAY["text"], NOTE)


def section_img():
    hog = read("results_image_traditional.csv")
    if hog is None:
        return
    story.append(PageBreak())
    P("5. Bài 3 – Dữ liệu ảnh (Oxford-IIIT Pet)", H1)
    P("<b>Dữ liệu:</b> Oxford-IIIT Pet (robots.ox.ac.uk/~vgg/data/pets), 37 giống chó/mèo, train = trainval (3.680 ảnh), test = 3.669 ảnh. "
      "Đặc trưng truyền thống: HOG (+PCA). Đặc trưng hiện đại: ResNet-50, EfficientNet-B0, ViT-B/16 pretrained (ImageNet), lưu .npy.")
    P("5.1 EDA", H2)
    fig("img_class_dist", 14, "Hình 17. Số ảnh theo lớp")
    fig("img_size_dist", 15, "Hình 18. Kích thước, tỉ lệ khung hình và kênh màu")
    fig("img_samples", 14, "Hình 19. Ảnh mẫu")
    fig("img_channel_dist", 8, "Hình 20. Cường độ trung bình theo kênh màu")
    P("5.2 Pipeline truyền thống: HOG (+PCA)", H2)
    P("Ảnh xám resize 128×128, HOG (9 hướng, 16×16 px/cell, block 2×2), tuỳ chọn PCA giữ 95% phương sai; phân loại bằng Logistic Regression, Linear SVM, SVM-RBF, Random Forest.")
    table(hog.sort_values("accuracy", ascending=False), ["features", "classifier", "dim", "accuracy", "precision", "recall", "f1"])
    df = read("results_image_deep_features.csv")
    if df is not None:
        P("5.3 Đặc trưng pretrained (đã lưu .npy)", H2)
        P("Các lựa chọn: backbone (ResNet-50, EfficientNet-B0, ViT-B/16), kích thước resize (224; thêm ResNet-50 ở 160) và bộ phân loại.")
        table(df.sort_values("accuracy", ascending=False), ["backbone", "img_size", "classifier", "dim", "accuracy", "precision", "recall", "f1"], max_rows=16)
        fig("img_deepfeat_heatmap", 11, "Hình 21. Accuracy theo backbone × bộ phân loại")
    deep = read("results_image_deep.csv")
    if deep is not None:
        P("5.4 (Điểm cộng) Pipeline học sâu", H2)
        table(deep, ["method", "accuracy", "precision", "recall", "f1", "fit_seconds"])
    summ = read("results_image_summary.csv")
    if summ is not None:
        P("5.5 So sánh tổng hợp", H2)
        table(summ, ["approach", "method", "accuracy", "f1"])
        fig("img_summary_bar", 11, "Hình 22. So sánh các cách tiếp cận")
        bh = best_row(hog, "accuracy")
        txt = f"<b>Kết quả:</b> HOG tốt nhất: {bh['features']} + {bh['classifier']} (accuracy = {bh['accuracy']:.4f})."
        if df is not None:
            bd = best_row(df, "accuracy")
            txt += f" Pretrained features tốt nhất: {bd['backbone']}_{bd['img_size']} + {bd['classifier']} (accuracy = {bd['accuracy']:.4f})."
        if deep is not None:
            txt += f" Học sâu tốt nhất: {deep.sort_values('accuracy').iloc[-1]['method']} (accuracy = {deep['accuracy'].max():.4f})."
        P(txt)
    fig("img_confusion_best", 12, "Hình 23. Confusion matrix")
    fig("img_errors", 15, "Hình 24. Một số ảnh dự đoán sai")
    P("5.6 Phân tích và kết luận", H2)
    P(NHAN_XET_TAY["img"], NOTE)


def workload():
    story.append(PageBreak())
    P("6. Bảng phân công công việc và tỉ lệ đóng góp", H1)
    rows = [["Họ tên", "MSSV", "Email", "Công việc phụ trách", "Tỉ lệ (%)"]] + [list(map(str, r)) for r in PHAN_CONG]
    rows.append(["", "", "", "Tổng", str(sum(r[4] for r in PHAN_CONG))])
    data = [[Paragraph(f"<b>{c}</b>" if i == 0 else c, CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[3.3 * cm, 2.2 * cm, 3.6 * cm, 6 * cm, 1.8 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7"))]))
    story.append(t)
    story.append(Spacer(1, 6))
    P("Điểm cá nhân = Điểm nhóm × Tỉ lệ đóng góp (%). Minh chứng làm việc nhóm (ảnh/video họp nhóm, biên bản) đặt trong thư mục tài liệu đính kèm / repo GitHub.")
    P("7. Kết luận chung và hướng phát triển", H1)
    P("[Nhóm viết 1–2 đoạn: so sánh tổng thể ba loại dữ liệu; khi nào pipeline truyền thống đủ tốt, khi nào học sâu vượt trội; hạn chế (một lần chia train/test, "
      "chưa grid search quy mô lớn…); hướng mở rộng (gradient boosting/TabNet, test-time augmentation, mô hình lớn hơn…).]", NOTE)
    P("Tài liệu tham khảo", H2)
    for r in [
        "Dua, D., Graff, C. UCI Machine Learning Repository – Adult Data Set. archive.ics.uci.edu/ml.",
        "Lang, K. 20 Newsgroups. qwone.com/~jason/20Newsgroups.",
        "Parkhi, O. et al. Cats and Dogs. CVPR 2012 (Oxford-IIIT Pet).",
        "Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.",
        "Pennington, J., Socher, R., Manning, C. GloVe: Global Vectors for Word Representation. EMNLP 2014.",
        "Sanh, V. et al. DistilBERT, a distilled version of BERT. 2019.",
        "He, K. et al. Deep Residual Learning for Image Recognition. CVPR 2016.",
        "Dosovitskiy, A. et al. An Image is Worth 16x16 Words (ViT). ICLR 2021.",
        "Dalal, N., Triggs, B. Histograms of Oriented Gradients for Human Detection. CVPR 2005.",
    ]:
        P("• " + r, BODY)


def main(out=OUT):
    os.makedirs(REPORT_DIR, exist_ok=True)
    cover(); intro(); section_tab(); section_text(); section_img(); workload()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Báo cáo BTL Học máy", author=GROUP["Tên nhóm"])

    def footer(canv, d):
        canv.setFont("DV", 8)
        canv.setFillColor(colors.grey)
        canv.drawCentredString(A4[0] / 2, 1.1 * cm, f"Học máy – BTL HK1 2026-2027 – trang {d.page}")

    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("Đã tạo", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
