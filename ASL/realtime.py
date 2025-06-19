import sys
import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
from keras.models import load_model
from collections import deque, Counter
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import pyttsx3

# ===== Load AI model & preprocessing =====
model = load_model("./Model/best_sign_classifier.keras")
scaler = joblib.load("./Model/sign_scaler.pkl")
le = joblib.load("./Model/sign_label_encoder.pkl")

mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils # type: ignore

def enhance_brightness_contrast(image):
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    img_y_cr_cb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)

def get_landmark_vec(hand_landmarks):
    vec = []
    for lm in hand_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec).reshape(1, -1)

BUFFER_SIZE = 15
DEBOUNCE = 4

class SignRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition - PyQt5")
        self.setGeometry(100, 100, 900, 600)

        # Video
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        # Output text
        self.text_label = QLabel("Text: ", self)
        self.text_label.setStyleSheet("font-size: 22px;")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Nút Clear All
        self.clear_btn = QPushButton("Clear All", self)
        self.clear_btn.clicked.connect(self.clear_all)
        
        # Nút Delete Last Character
        self.delete_btn = QPushButton("Delete", self)
        self.delete_btn.clicked.connect(self.delete_last_char)

        # Nút đọc voice
        self.voice_btn = QPushButton("🔊 Voice", self)
        self.voice_btn.clicked.connect(self.read_voice)

        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.video_label)
        vbox.addWidget(self.text_label)
        vbox.addWidget(self.clear_btn)
        vbox.addWidget(self.delete_btn) 
        vbox.addWidget(self.voice_btn)
        
        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)

        # AI variables
        self.cap = cv2.VideoCapture(0)
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.current_sign = ""
        self.stable_count = 0
        self.output_text = ""
        self.last_sign = ""
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', 1)

        self.engine.setProperty('rate', 140)
        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def clear_all(self):
        self.output_text = ""
        self.text_label.setText("Text: ")

    def delete_last_char(self):
        if len(self.output_text) > 0:
            self.output_text = self.output_text[:-1]
            self.text_label.setText(f"Text: {self.output_text}")

    def read_voice(self):
        def speak():
            try:
                self.engine.say(self.output_text.strip())
                self.engine.runAndWait()
            except RuntimeError:
                pass
            self.voice_btn.setEnabled(True)

        if self.voice_btn.isEnabled():
            self.voice_btn.setEnabled(False)
            t = threading.Thread(target=speak)
            t.start()

    def update_sign(self, new_pred):
        self.buffer.append(new_pred)
        common = Counter(self.buffer).most_common(1)[0][0]
        if common != self.current_sign:
            self.stable_count += 1
            if self.stable_count >= DEBOUNCE:
                self.current_sign = common
                self.stable_count = 0
        else:
            self.stable_count = 0
        return self.current_sign

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_proc = enhance_brightness_contrast(frame)
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        sign_text = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = get_landmark_vec(hand_landmarks)
                features_scaled = scaler.transform(features)
                probs = model.predict(features_scaled, verbose=0)[0] # type: ignore
                pred_idx = np.argmax(probs)
                pred_label = le.inverse_transform([pred_idx])[0]
                sign_text = self.update_sign(pred_label)
                break
        else:
            sign_text = self.update_sign("nothing")

        if sign_text != self.last_sign and sign_text not in ["nothing", ""]:
            if sign_text == "space":
                self.output_text += " "
            elif sign_text in ["delete", "del"]:
                # Xóa đúng 1 ký tự cuối cùng (nếu có)
                if len(self.output_text) > 0:
                    self.output_text = self.output_text[:-1]
            else:
                self.output_text += sign_text
            self.last_sign = sign_text

        self.text_label.setText(f"Text: {self.output_text}")

        # Hiển thị lên GUI
        show_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = show_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(show_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event): # type: ignore
        # Stop camera, release resource khi đóng app
        self.cap.release()
        self.engine.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SignRecognizerApp()
    win.show()
    sys.exit(app.exec_())
