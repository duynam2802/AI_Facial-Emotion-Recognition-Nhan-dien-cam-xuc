# 😊 Nhận Diện Cảm Xúc Khuôn Mặt (Facial Emotion Recognition)

> Đồ án môn Trí Tuệ Nhân Tạo — Nhận diện 7 cảm xúc khuôn mặt bằng CNN + OpenCV Webcam Realtime

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-~65%25-brightgreen)

---

## 📌 Giới thiệu

Dự án xây dựng hệ thống **nhận diện cảm xúc khuôn mặt realtime** sử dụng **Mạng Nơ-ron Tích Chập (CNN)** kết hợp **OpenCV** để phát hiện và phân tích cảm xúc từ webcam hoặc ảnh tĩnh.

Hệ thống phân loại **7 cảm xúc cơ bản** theo mô hình Ekman:

| Emoji | Cảm xúc | Tiếng Anh |
|---|---|---|
| 😡 | Tức giận | Angry |
| 🤢 | Ghê tởm | Disgust |
| 😨 | Sợ hãi | Fear |
| 😊 | Vui vẻ | Happy |
| 😐 | Bình thường | Neutral |
| 😢 | Buồn bã | Sad |
| 😲 | Ngạc nhiên | Surprise |

---

## 🎯 Mục tiêu

- Xây dựng CNN model phân loại 7 cảm xúc từ ảnh khuôn mặt 48x48
- Tích hợp **OpenCV** để detect khuôn mặt realtime từ webcam
- Triển khai **Streamlit Web App** có UI đẹp với demo webcam live
- Trực quan hóa kết quả: biểu đồ confidence, lịch sử cảm xúc

---

## 🗂️ Cấu trúc dự án

```
emotion-recognition/
│
├── data/
│   └── fer2013.csv              # Dataset FER2013 (35,887 ảnh)
│
├── src/
│   ├── preprocess.py            # Chuẩn bị dữ liệu từ CSV
│   ├── model.py                 # Định nghĩa kiến trúc CNN
│   ├── train.py                 # Huấn luyện mô hình
│   └── evaluate.py              # Đánh giá & visualize
│
├── models/
│   ├── emotion_model.h5         # Mô hình đã train (Keras)
│   └── emotion_model.keras      # Backup format mới
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── classification_report.txt
│
├── app/
│   └── app.py                   # Streamlit Web App
│
├── haarcascade_frontalface_default.xml  # OpenCV face detector
├── requirements.txt
└── README.md
```

---

## 📊 Dataset — FER2013

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | [Kaggle — FER2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| **Tổng mẫu** | 35,887 ảnh khuôn mặt |
| **Kích thước** | 48x48 pixels, grayscale |
| **Kích thước file** | ~63 MB |
| **Nhãn** | 7 cảm xúc (0–6) |
| **Split** | Train: 28,709 / Test: 3,589 |

**Tải dataset:**
```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/
```

Hoặc tải thủ công: https://www.kaggle.com/datasets/msambare/fer2013

---

## ⚙️ Cài đặt môi trường (khuyến nghị Python 3.11)

### 1) Cài Python 3.11

- Tải Python 3.11 tại: https://www.python.org/downloads/
- Khi cài trên Windows, nhớ tick **Add Python to PATH**.

Kiểm tra phiên bản:

```bash
python --version
```

Kỳ vọng: `Python 3.11.x`

### 2) Clone / mở project và tạo virtual environment

Trong thư mục gốc dự án:

```bash
python -m venv venv
```

Kích hoạt venv:

Windows (PowerShell):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Windows (CMD):

```bat
venv\Scripts\activate.bat
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3) Cài dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🚀 Chạy dự án từ đầu đến khi thành công

Sau khi đã kích hoạt venv:

```bash
# BỎ QUA BƯỚC 1 ĐẾN 3 
# Bước 1: Chuẩn bị data (tạo X_train, y_train, ...)
python src/preprocess.py

# Bước 2: Train CNN (15-30 phút với CPU, ~5 phút với GPU)
python src/train.py

# Bước 3: Đánh giá mô hình
python src/evaluate.py

# Bước 4: Chạy Web App + Webcam
python -m streamlit run app/app.py
```

Khi chạy thành công, terminal sẽ hiện URL dạng:

```text
Local URL: http://localhost:8501
```

Mở URL đó trên trình duyệt để dùng app.

### Lỗi thường gặp

Nếu gặp lỗi:

```text
streamlit : The term 'streamlit' is not recognized ...
```

Hãy dùng lệnh ổn định hơn (không phụ thuộc PATH):

```bash
python -m streamlit run app/app.py
```

Nếu vẫn lỗi, kiểm tra lại bạn đã kích hoạt đúng venv chưa:

```bash
python -m pip --version
```

Đường dẫn pip cần trỏ vào thư mục `venv` của dự án.

---

## 🧠 Kiến trúc CNN

```
Input: (48, 48, 1) Grayscale
    │
    ▼
Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
    ▼
Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
    ▼
Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
    ▼
Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
    ▼
Flatten → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    │
    ▼
Dense(7) → Softmax
    │
    ▼
Output: 7 cảm xúc + confidence %
```

---

## 📈 Kết quả dự kiến

| Metric | Score |
|---|---|
| Train Accuracy | ~85% |
| Test Accuracy | ~63–67% |
| Best class | Happy (~90%) |
| Hardest class | Fear / Disgust (~45%) |
| Train time (CPU) | 20–30 phút |

> *FER2013 là dataset khó, accuracy 63–67% là kết quả tốt với CNN thuần.*

---

## 🌐 Web App Features

- 📷 **Upload ảnh** → phát hiện khuôn mặt → hiển thị cảm xúc
- 🎥 **Webcam realtime** → nhận diện liên tục từ camera
- 📊 **Biểu đồ confidence** 7 cảm xúc dạng bar chart
- 🕐 **Lịch sử cảm xúc** theo thời gian thực

---

## 📚 Tài liệu tham khảo

- Goodfellow, I., et al. (2013). *Challenges in representation learning: A report on three machine learning contests*. (FER2013 paper)
- [Keras CNN Tutorial](https://keras.io/guides/training_with_built_in_methods/)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

---

## 👤 Tác giả

| | |
|---|---|
| **Họ tên** | *(Điền tên bạn)* |
| **MSSV** | *(Điền MSSV)* |
| **Môn** | Trí Tuệ Nhân Tạo |
| **Năm** | 2025 |

---

> ⭐ Star repo nếu project hữu ích!
