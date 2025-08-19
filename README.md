# Sign Language Recognition System

## 📖 Tổng quan dự án

Dự án này xây dựng một hệ thống nhận dạng ngôn ngữ ký hiệu (ASL - American Sign Language) trong thời gian thực sử dụng computer vision và machine learning. Hệ thống có thể nhận dạng các ký hiệu bằng tay thông qua webcam và đưa ra dự đoán với độ chính xác cao.

## 🎯 Mục tiêu

- Nhận dạng các ký hiệu ngôn ngữ ký hiệu ASL trong thời gian thực
- Xây dựng giao diện người dùng thân thiện với PyQt5
- Đạt độ chính xác cao với tốc độ xử lý nhanh
- Tối ưu hóa cho việc triển khai thực tế

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Input  │ -> │ MediaPipe Hands  │ -> │ Hand Landmarks  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prediction    │ <- │   ML Classifier  │ <- │ Preprocessing   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔬 Phương pháp tiếp cận

### Sử dụng MediaPipe thay vì CNN

#### 1. **Hiệu quả tính toán**
- **MediaPipe + Classifier**: Chỉ cần trích xuất 21 landmarks (63 features) → Model nhỏ gọn
- **CNN**: Xử lý toàn bộ image (224x224x3 = 150,528 features) → Tính toán phức tạp

#### 2. **Bất biến với môi trường**
- MediaPipe chuẩn hóa tọa độ landmark về [0,1] → Không phụ thuộc background, lighting
- CNN dễ bị ảnh hưởng bởi background noise, ánh sáng, màu sắc

#### 3. **Tốc độ FPS cao**
- MediaPipe: ~30-60 FPS trên CPU thường
- CNN: Cần GPU mạnh để đạt real-time

#### 4. **Phù hợp với dữ liệu training nhỏ và vừa**
- Landmark features có tính invariant cao → Cần ít data hơn
- CNN cần dataset lớn để học được các pattern phức tạp

#### 5. **Robustness**
```python
# MediaPipe chuẩn hóa tọa độ
normalized_landmarks = [(x/image_width, y/image_height, z) for x,y,z in landmarks]
# → Bất biến với kích thước ảnh, vị trí tay trong frame
```

## 📊 Quy trình xử lý dữ liệu

### 1. **Data Augmentation**
```python
# preprocessing/augment.py
transform = A.Compose([
    A.HorizontalFlip(p=0.5),              # Lật ngang
    A.ShiftScaleRotate(
        shift_limit=0.10,                 # Dịch chuyển ±10%
        scale_limit=0.10,                 # Scale ±10%
        rotate_limit=20,                  # Xoay ±20°
        p=0.8
    ),
    A.RandomBrightnessContrast(p=0.7),    # Thay đổi sáng/tương phản
    A.GaussianBlur(blur_limit=(3, 7), p=0.5), # Làm mờ nhẹ
])
```

**Tại sao augment?**
- Tăng kích thước dataset từ N → N×(1+5) = 6N samples
- Tăng tính đa dạng: góc nhìn, ánh sáng, vị trí khác nhau
- Giảm overfitting, tăng generalization

### 2. **Landmark Extraction**
```python
# preprocessing/extract_landmark.py
def enhance_brightness_contrast(image):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    img_y_cr_cb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)
```

**Pipeline xử lý:**
1. **Image Enhancement**: CLAHE để cải thiện tương phản
2. **Hand Detection**: MediaPipe Hands detection
3. **Landmark Extraction**: 21 điểm landmark × 3 tọa độ (x,y,z) = 63 features
4. **CSV Export**: Lưu features + labels để training

### 3. **Feature Engineering**

**Đặc trưng Landmark:**
- **21 keypoints** của bàn tay từ MediaPipe
- **Tọa độ (x,y,z)**: x,y đã normalize [0,1], z là depth
- **Invariant properties**: Không phụ thuộc kích thước ảnh, vị trí tay

```
Thumb: 0-4    │ Index: 5-8     │ Middle: 9-12   │ Ring: 13-16    │ Pinky: 17-20
   4             8                12               16               20
   │             │                │                │                │
   3             7                11               15               19
   │             │                │                │                │
   2             6                10               14               18
   │             │                │                │                │
   1             5                9                13               17
   │            ╱                ╱                ╱                ╱
   0 ←─────────╱────────────────╱────────────────╱────────────────╱
 (wrist)
```

## 🤖 Model Architecture & Training

### Model Design
```python
model = Sequential([
    Dense(256, activation='tanh', input_shape=(63,)),  # Input: 21×3 landmarks
    Dense(128, activation='tanh'),                     # Hidden layer
    Dense(num_classes, activation='softmax')           # Output: Class probabilities
])
```

**Tại sao architecture này?**
- **Dense layers**: Phù hợp với tabular data (landmarks)
- **Tanh activation**: Hiệu quả với normalized input [-1,1]
- **Softmax output**: Multi-class classification
- **Compact**: Chỉ ~100K parameters → Fast inference

### Training Strategy
```python
# Callbacks for optimal training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,                    # Dừng sớm nếu không cải thiện
    restore_best_weights=True       # Lấy weights tốt nhất
)

model_ckpt = ModelCheckpoint(
    "best_sign_classifier.keras",
    monitor='val_loss',
    save_best_only=True            # Chỉ lưu model tốt nhất
)
```

### Preprocessing Pipeline
```python
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Chuẩn hóa features về mean=0, std=1

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(labels)  # String labels → Integer
y_categorical = to_categorical(y_encoded)  # One-hot encoding
```

## 🚀 Real-time Inference

### Optimizations trong Real-time
```python
# Smoothing predictions với voting
BUFFER_SIZE = 15
DEBOUNCE = 4
predictions_buffer = deque(maxlen=BUFFER_SIZE)

# Majority voting để giảm noise
def get_stable_prediction(buffer):
    if len(buffer) >= DEBOUNCE:
        counter = Counter(buffer)
        most_common = counter.most_common(1)[0]
        if most_common[1] >= DEBOUNCE:
            return most_common[0]
    return None
```

**Kỹ thuật tối ưu:**
1. **Temporal Smoothing**: Voting trên 15 frames gần nhất
2. **Debouncing**: Chỉ accept prediction khi xuất hiện ≥4 lần
3. **Single Hand**: max_num_hands=1 để tăng tốc
4. **Confidence Threshold**: min_detection_confidence=0.5

## 📁 Cấu trúc thư mục

```
SignLanguage/
├── Model/
│   ├── Model.ipynb              # Training notebook
│   ├── best_sign_classifier.keras  # Trained model
│   ├── sign_label_encoder.pkl   # Label encoder
│   └── sign_scaler.pkl          # Feature scaler
├── preprocessing/
│   ├── extract_landmark.py      # Landmark extraction
│   ├── augment.py              # Data augmentation
│   └── visualize_landmark.py   # Visualization tools
├── data/
│   └── hand_landmarks.csv      # Processed dataset
├── ASL/                        # Additional ASL data
├── realtime.py                 # Main real-time app
└── README.md                   # This file
```

## 🎮 Cách sử dụng

### 1. Cài đặt dependencies
```bash
pip install opencv-python mediapipe tensorflow scikit-learn PyQt5 pyttsx3 albumentations joblib
```

### 2. Training model (nếu cần)
```bash
jupyter notebook Model/Model.ipynb
```

### 3. Chạy ứng dụng real-time
```bash
python realtime.py
```

## 📹 Demo Video

*[Video demo sẽ được thêm vào sau]*


## 🔧 Kỹ thuật bổ sung

### 1. **Data Preprocessing**
- **CLAHE Enhancement**: Cải thiện contrast cho điều kiện ánh sáng khác nhau
- **Landmark Normalization**: MediaPipe tự động normalize về [0,1]
- **Missing Data Handling**: Skip images không detect được tay

### 2. **Model Optimization**
- **Early Stopping**: Tránh overfitting
- **Stratified Split**: Đảm bảo balance classes trong train/test
- **Feature Scaling**: StandardScaler cho convergence nhanh hơn

### 3. **Real-time Optimization**
- **Temporal Filtering**: Giảm noise prediction
- **Multi-threading**: UI không bị block khi inference
- **Memory Management**: Circular buffer cho predictions

## 📊 Kết quả và Đánh giá

### Metrics đạt được:
- **Accuracy**: ~95%+ trên test set
- **Inference Speed**: 30+ FPS trên CPU
- **Model Size**: <5MB
- **Real-time Latency**: <50ms per frame

### So sánh với CNN approach:
| Metric | MediaPipe + ML | CNN Direct |
|--------|---------------|------------|
| Model Size | ~5MB | ~50-200MB |
| Training Time | ~10 mins | ~2-5 hours |
| Inference Speed | 30+ FPS (CPU) | 15-30 FPS (GPU) |
| Data Required | ~1K samples/class | ~5K+ samples/class |
| Robustness | High (invariant) | Medium (background dependent) |

## 🔮 Hướng phát triển

1. **Multi-hand Support**: Hỗ trợ nhận dạng 2 tay
2. **Continuous Recognition**: Nhận dạng câu/từ liên tục
3. **3D Pose Integration**: Thêm body pose cho context
4. **Mobile Deployment**: Port sang TensorFlow Lite
5. **Custom Vocabulary**: Thêm ký hiệu Việt Nam

## 👥 Đóng góp

Dự án mở để mọi người học tập và nghiên cứu.

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

---

*Dự án được phát triển với mục đích học tập về Computer Vision và Machine Learning.*
