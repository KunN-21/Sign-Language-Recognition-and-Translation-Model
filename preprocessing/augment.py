import os
import cv2
import albumentations as A
from tqdm import tqdm

# Định nghĩa pipeline augment CHUẨN
transform = A.Compose([
    A.HorizontalFlip(p=0.5),              # Lật ngang
    A.ShiftScaleRotate(                   # Dịch chuyển, scale, xoay
        shift_limit=0.10,                 # dịch ngang/dọc tối đa ±10%
        scale_limit=0.10,                 # scale ±10%
        rotate_limit=20,                  # xoay ±20 độ
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.8
    ),
    A.RandomBrightnessContrast(p=0.7),    # Thay đổi sáng/tương phản
    A.GaussianBlur(blur_limit=(3, 7), p=0.5), # Làm mờ nhẹ
])

INPUT_DIR = "../data/asl_alphabet"
OUTPUT_DIR = "../data/augmented"
N_AUG = 5 # Số lượng augment cho mỗi ảnh

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in tqdm(os.listdir(INPUT_DIR), desc="Classes"):
    in_folder = os.path.join(INPUT_DIR, label)
    out_folder = os.path.join(OUTPUT_DIR, label)
    if not os.path.isdir(in_folder):
        continue
    os.makedirs(out_folder, exist_ok=True)
    for img_file in tqdm(os.listdir(in_folder), desc=f"{label}", leave=False):
        img_path = os.path.join(in_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Lỗi đọc ảnh: {img_path}")
            continue
        cv2.imwrite(os.path.join(out_folder, img_file), image)
        for i in range(N_AUG):
            augmented = transform(image=image)
            aug_img = augmented['image']
            base, ext = os.path.splitext(img_file)
            out_file = os.path.join(out_folder, f"{base}_aug_{i}{ext}")
            cv2.imwrite(out_file, aug_img)
print("DONE: Augment toàn bộ dataset!")
