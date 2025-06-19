import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
from keras.models import load_model

mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils # type: ignore

model = load_model("./Model/best_sign_classifier.keras")
scaler = joblib.load("./Model/sign_scaler.pkl")
le = joblib.load("./Model/sign_label_encoder.pkl")

def get_landmark_vec(hand_landmarks):
    vec = []
    for lm in hand_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec).reshape(1, -1)

def enhance_brightness_contrast(image):
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    img_y_cr_cb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)


image = cv2.imread(r"D:\Code\School\AI\SignLanguage\data\asl_alphabet\A\A315.jpg")
image = enhance_brightness_contrast(image)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(rgb)
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        features = get_landmark_vec(hand_landmarks)
        features_scaled = scaler.transform(features)
        probs = model.predict(features_scaled, verbose=0)[0] # type: ignore
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        