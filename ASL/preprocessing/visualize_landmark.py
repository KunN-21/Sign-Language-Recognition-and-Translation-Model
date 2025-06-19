import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Đọc file landmark
csv_file = './data/hand_landmarks.csv'   # Đã có cột image_path
df = pd.read_csv(csv_file)

# Tham số
N = 6  # Số mẫu muốn xem

# Kết nối chuẩn 21 điểm của MediaPipe
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]

def draw_landmarks_on_image(image, landmarks):
    h, w = image.shape[:2]
    # Vẽ các điểm
    for idx, lm in enumerate(landmarks):
        x, y = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(image, str(idx), (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA)
    # Vẽ các kết nối
    for (i, j) in CONNECTIONS:
        x1, y1 = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
        x2, y2 = int(landmarks[j][0] * w), int(landmarks[j][1] * h)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

# Lấy ngẫu nhiên N sample
sampled = df.sample(N) if len(df) >= N else df
# sampled = df[df['label']=='del'].sample(N)


plt.figure(figsize=(18, 4))
for idx, (_, row) in enumerate(sampled.iterrows()):
    img_path = "./" + row['image_path'].replace('\\', '/')
    label = row['label']
    landmarks = row.values[2:].astype(float).reshape((21, 3))  # bỏ label, image_path
    # Đọc ảnh gốc
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không tìm thấy ảnh: {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Vẽ landmark
    vis_img = draw_landmarks_on_image(img.copy(), landmarks)
    # Plot
    plt.subplot(1, N, idx+1)
    plt.imshow(vis_img)
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
