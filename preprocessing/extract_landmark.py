import os
import cv2
import mediapipe as mp
import csv
import numpy as np

# Hàm tăng sáng, tăng tương phản dùng CLAHE
def enhance_brightness_contrast(image):
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    img_y_cr_cb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)

# --- Các đường dẫn cấu hình ---
DATA_DIRS = ["../data/asl_alphabet"]  
CSV_FILE = "../data/hand_landmarks.csv"
NOTHING_LOG = "skipped_nothing_images.txt"

mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

header = ['label', 'image_path']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

nothing_count = 0
total_count = 0

with open(CSV_FILE, mode='w', newline='') as f_csv, open(NOTHING_LOG, 'w') as f_log:
    writer = csv.writer(f_csv)
    writer.writerow(header)
    for data_dir in DATA_DIRS:
        for label in os.listdir(data_dir):
            folder = os.path.join(data_dir, label)
            if not os.path.isdir(folder): continue
            for img_file in os.listdir(folder):
                img_path = os.path.join(folder, img_file)
                total_count += 1
                # Skip class nothing
                if label.lower() == "nothing":
                    f_log.write(f"{img_path}\n")
                    nothing_count += 1
                    continue
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Lỗi đọc ảnh: {img_path}")
                    continue
                # === Tiền xử lý tăng sáng, tăng tương phản ===
                image = enhance_brightness_contrast(image)
                # Chuyển về RGB cho MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    row = [label, img_path]
                    for lm in hand_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z]
                    writer.writerow(row) 
                else:
                    print(f"Không phát hiện bàn tay: {img_path}")

print(f"Đã extract landmark cho {total_count-nothing_count} ảnh (bỏ qua {nothing_count} ảnh 'nothing').")
print(f"Danh sách ảnh 'nothing' đã skip lưu tại: {NOTHING_LOG}")
