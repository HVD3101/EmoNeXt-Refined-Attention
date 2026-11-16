# EmoNeXt: Facial Emotion Recognition (FER2013)

Cải tiến mô hình học sâu **EmoNeXt** (kết hợp ConvNeXt và Scaled Dot-Product Self-Attention) cho bài toán nhận diện cảm xúc khuôn mặt trên tập dữ liệu FER2013.

---

## 🚀 Điểm cải tiến chính so với code gốc tác giả

1. **Chuẩn hóa hệ số Scale Attention:** Sửa scale từ $d^{-1/4}$ thành $1/\sqrt{d}$ theo đúng chuẩn cơ chế Scaled Dot-Product Attention, ngăn hiện tượng triệt tiêu đạo hàm (vanishing gradients).
2. **Bảo toàn chiều không gian (Spatial Dimensions):** Bỏ Global Average Pooling (GAP) sớm trong `forward_features` để giữ tensor dạng $(B, C, 7, 7)$, cho phép cơ chế Attention hoạt động trên 49 spatial tokens khuôn mặt (mắt, mũi, miệng) thay vì nhầm sang batch dimension.
3. **Luồng dữ liệu Attention:** Định tuyến tensor sau attention (`attended_x`) trực tiếp vào classification head thay vì dùng feature cũ chưa qua attention.
4. **Hàm mất mát (Loss Function):** Chuẩn hóa cách tính Self-Attention loss per-token và thiết lập $\lambda=0.1$ cân bằng với Cross-Entropy loss; kết hợp Label Smoothing (0.1) chống nhiễu nhãn.
5. **Quy trình huấn luyện & Đánh giá:** 
   - Thay scheduler bằng Linear Warmup (10 epochs) + Cosine Annealing decay.
   - Sửa validation transform sang `CenterCrop` để kết quả đánh giá nhất quán và khách quan.
   - Tối ưu `val_batch_size = 32`, tăng tốc độ validation ~30 lần.
   - Tăng `early_stopping_patience = 20`.
   - Nâng cấp `torch.amp.GradScaler` theo PyTorch API mới.

---

## 📊 Kết quả thực nghiệm (FER2013)

* **Validation Accuracy (Best):** **73.07%**
* **Test Accuracy (Raw Model):** **71.82%**
* **Test Accuracy (EMA Model):** **72.51%** *(Baseline paper gốc: 73.34%)*

### Độ chính xác theo từng lớp cảm xúc (Test Set - EMA):
* **Happy:** 89.7%
* **Surprise:** 84.1%
* **Disgust:** 71.4%
* **Neutral:** 71.2%
* **Angry:** 65.1%
* **Sad:** 61.5%
* **Fear:** 55.3%

---

## 💻 Cài đặt & Sử dụng

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Tải dữ liệu
```bash
python download_dataset.py
```

### 3. Huấn luyện mô hình
```bash
python train.py
```
