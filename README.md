# Bài tập lớn môn Học máy — Pipeline học máy trên dữ liệu bảng, văn bản và ảnh

| | |
|---|---|
| **Môn học** | Học máy — mã môn học: CO3117 |
| **Học kỳ / Năm học** | Học kỳ I — 2026-2027 |
| **Trường / Khoa** | Đại học Bách Khoa, ĐHQG-HCM — Khoa Khoa học và Kỹ thuật Máy tính |
| **GVHD** | Trương Vĩnh Lân |

## Thành viên nhóm

| STT | Họ và tên | MSSV | Email | Đóng góp (%) |
|---|---|---|---|---|
| 1 | Đinh Ngụy Nguyệt Hà | 2352286 | ha.dinhnguy2005@hcmut.edu.vn | 100% |
| 2 | Đặng Thế Lâm Anh | *[MSSV]* | *[email]* | 100% |
| 3 | Hồng Chấn Phước | *[MSSV]* | *[email]* | 100% |
| 4 | Trần Bá Minh Quân | *[MSSV]* | *[email]* | 100% |

## Mục tiêu

- Hiểu và áp dụng **pipeline học máy truyền thống**: EDA → tiền xử lý → trích xuất đặc trưng → huấn luyện → đánh giá.
- Triển khai trên **3 loại dữ liệu**: bảng, văn bản, ảnh.
- **(Điểm cộng)** triển khai pipeline **học sâu** (MLP, embedding DistilBERT/GloVe, đặc trưng ResNet/EfficientNet/ViT, fine-tune đầu-cuối) và so sánh với pipeline truyền thống.
- Rèn luyện kỹ năng phân tích, so sánh mô hình và viết báo cáo khoa học.

| Bài | Dữ liệu (nguồn công khai) | Đặc trưng truyền thống | Đặc trưng/học sâu hiện đại |
|---|---|---|---|
| 1 — Tabular | [UCI Adult Income](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data) (có missing + categorical) | impute, one-hot/ordinal, StandardScaler/MinMaxScaler, PCA | MLP (PyTorch) |
| 2 — Text | [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) | BoW, TF-IDF, n-gram | GloVe-100, DistilBERT (lưu `.npy`), fine-tune DistilBERT |
| 3 — Image | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) | HOG (+PCA) | ResNet-50, EfficientNet-B0, ViT-B/16 (lưu `.npy`), fine-tune ResNet-50 |

## Hướng dẫn chạy notebook

**Cách 1 — Google Colab (khuyến nghị, đúng yêu cầu đề):**
1. Mở notebook trong `notebooks/` trên Colab (Upload hoặc *File → Open notebook → GitHub*).
2. *Runtime → Change runtime type → **T4 GPU*** (Bài 2 và 3 chạy nhanh hơn nhiều; vẫn chạy được trên CPU với bản rút gọn phần fine-tune).
3. *Runtime → **Run all***. Notebook tự: cài thư viện, ghi các module `modules/*.py` ra máy chủ Colab, tải dữ liệu công khai, chạy toàn bộ thí nghiệm. **Không** mount Google Drive.
4. Kết quả sinh ra trong thư mục làm việc của Colab: `reports/*.csv`, `reports/figures/*.png`, `features/*.npy`. Tải về (Files panel) và đặt vào thư mục tương ứng của repo.

**Cách 2 — chạy local:**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_tabular_adult_income.ipynb   # chạy từ thư mục gốc của repo
```
(Notebook gọi `%%writefile modules/...` để ghi lại module nên cần chạy với thư mục làm việc là gốc repo.)

**Thời gian tham khảo (Colab T4):** Bài 1 ≈ 10–15 phút · Bài 2 ≈ 20–30 phút (trích DistilBERT + fine-tune 3 epoch) · Bài 3 ≈ 20–30 phút (tải ~800 MB, trích 4 cấu hình + fine-tune 5 epoch).

## Sinh báo cáo PDF

Sau khi đã chạy các notebook và có `reports/results_*.csv` + `reports/figures/`:
```bash
# điền thông tin nhóm / bảng phân công trong dict GROUP, PHAN_CONG, NHAN_XET_TAY ở đầu file
python modules/make_report.py        # -> reports/BaoCao_BTL_HocMay.pdf
```
Báo cáo lấy **số liệu và hình thật từ các lần chạy**; nhóm cần viết thêm phần phân tích/giải thích của riêng mình (các hộp vàng trong PDF) và bảng phân công.

## Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_tabular_adult_income.ipynb      # Bài 1
│   ├── 02_text_20newsgroups.ipynb         # Bài 2
│   └── 03_image_oxford_pets.ipynb         # Bài 3
├── modules/
│   ├── common.py        # seed, đánh giá, vẽ hình, lưu .npy/.h5, ResultLogger
│   ├── tabular.py       # tải dữ liệu, EDA, pipeline cấu hình được (Bài 1)
│   ├── text.py          # tiền xử lý, BoW/TF-IDF, GloVe, DistilBERT, fine-tune (Bài 2)
│   ├── image.py         # EDA ảnh, HOG, backbone pretrained, fine-tune (Bài 3)
│   ├── dl.py            # MLP PyTorch + early stopping (phần học sâu)
│   └── make_report.py   # sinh báo cáo PDF từ kết quả thật
├── reports/             # results_*.csv, figures/*.png, BaoCao_BTL_HocMay.pdf
├── features/            # đặc trưng đã trích xuất (.npy)
└── tools/               # script dựng lại notebook từ modules (không bắt buộc)
```

## Liên kết
- Dataset: 
| STT | Phân loại | Tên | Link | 
|---|---|---|---|
| 1 | Tabular | Hotel Booking Demand | https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand |
| 2 | Text | BBC News Archive | https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive |
| 3 | Image | Food Classification | https://www.kaggle.com/datasets/bjoernjostein/food-classification/data |

- Báo cáo PDF: `reports/Group 8 Machine Learning Report.pdf` — *[link nếu có]*
- Colab notebooks: *[link Bài 1]* · *[link Bài 2]* · *[link Bài 3]*
- Link tải đặc trưng: *[link]*
