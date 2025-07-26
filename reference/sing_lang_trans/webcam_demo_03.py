# webcam_demo_02_fixed.py

import cv2
import json
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

import mediapipe as mp

from modules.holistic_module import HolisticDetector  

# ─── 1) 한글 폰트 로드 ────────────────────────────────────────────
# 프로젝트 폴더에 NanumGothic.ttf 같은 한글 폰트 파일을 두세요.
FONT_PATH = "reference/fonts/D2Coding-Ver1.3.2-20180524.ttf"  
font = ImageFont.truetype(FONT_PATH, 24)

# ─── 2) 클래스(라벨) 로드 ───────────────────────────────────────
with open('reference/label_group.json', encoding='utf-8') as f:
    label_dict = json.load(f)
items   = sorted(label_dict.items(), key=lambda x: int(x[0]))
classes = [v['attribute'] for _, v in items]
classes = list(dict.fromkeys(classes))
print("▶ classes:", classes, len(classes))

# ─── 3) TFLite 모델 준비 ────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path='reference/models/multi_hand_gesture_classifier.tflite')
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

SEQ_LENGTH = inp_det['shape'][1]
FEAT_DIM   = inp_det['shape'][2]   # 예: 84(정면) 또는 168(양손)
print("▶ SEQ_LENGTH:", SEQ_LENGTH, " FEAT_DIM:", FEAT_DIM)

# ─── 4) HolisticDetector 및 버퍼 초기화 ─────────────────────────
cap         = cv2.VideoCapture(0)
detector    = HolisticDetector(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
seq_buffer  = []
pred_buffer = deque(maxlen=5)

# Mediapipe 그리기 유틸
mp_drawing  = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap       = cv2.VideoCapture(0)
detector  = HolisticDetector(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ─── A) 미디어파이프 감지 & 랜드마크 획득 ───────────────────
    image = detector.findHolistic(frame, draw=False)
    _, left_lm  = detector.findLefthandLandmark(image)
    _, right_lm = detector.findRighthandLandmark(image)

    # ─── B) 랜드마크 시각화 ────────────────────────────────────
    if left_lm is not None:
        mp_drawing.draw_landmarks(
            image, left_lm, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
        )
    if right_lm is not None:
        mp_drawing.draw_landmarks(
            image, right_lm, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=2)
        )

    # ─── C) 키포인트 추출 (양손 병합) ────────────────────────────
    keypoints = []
    h, w, _ = image.shape

    if left_lm:
        for lm in left_lm.landmark:
            keypoints += [lm.x * w, lm.y * h]
    else:
        keypoints += [0.0] * 42

    if right_lm:
        for lm in right_lm.landmark:
            keypoints += [lm.x * w, lm.y * h]
    else:
        keypoints += [0.0] * 42

    # ─── D) 슬라이딩 윈도우 예측 ────────────────────────────────
    seq_buffer.append(keypoints)
    if len(seq_buffer) > SEQ_LENGTH:
        seq_buffer.pop(0)

    if len(seq_buffer) == SEQ_LENGTH:
        data = np.array(seq_buffer, dtype=np.float32).reshape(1, SEQ_LENGTH, FEAT_DIM)
        interpreter.set_tensor(inp_det['index'], data)
        interpreter.invoke()
        probs = interpreter.get_tensor(out_det['index'])[0]
        pred = classes[int(np.argmax(probs))]
        pred_buffer.appendleft(pred)

    # ─── E) PIL로 한글 텍스트 그리기 ─────────────────────────────
    # OpenCV 에서는 한글이 깨지므로 PIL로 렌더링 후 다시 numpy array로
    pil_img = Image.fromarray(image)
    draw    = ImageDraw.Draw(pil_img)
    pre_txt = ""
    for i, txt in enumerate(pred_buffer):
        if pre_txt!=txt:
            draw.text((10, 30 + i*30), txt, font=font, fill=(0,255,0))
        pre_txt = txt
    image = np.array(pil_img)

    # ─── F) 화면 출력 ───────────────────────────────────────────
    cv2.imshow("Sign-to-Text", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
