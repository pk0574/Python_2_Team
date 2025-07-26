# webcam_test_model_tflite.py
'''
import cv2
import json
import numpy as np
from collections import deque
import tensorflow as tf

from modules.holistic_module import HolisticDetector  # 자신의 프로젝트 구조에 맞게 임포트
# (예: from modules.holistic_module import HolisticDetector)

# ─── 1) 라벨(클래스) 동적 로딩 ─────────────────────────────────────
with open('data/label_group.json', encoding='utf-8') as f:
    label_dict = json.load(f)
# label_dict: { "1": {"video_name": "...", "attribute":"고민"}, ... }
items   = sorted(label_dict.items(), key=lambda x: int(x[0]))
classes = [v['attribute'] for _, v in items]           # ["고민","사랑",...]
print("▶ classes:", classes)

# ─── 2) TFLite 모델 로드 ──────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path='hearing_impaired_helper_make_model/models/multi_hand_gesture_classifier.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

SEQ_LENGTH = input_details['shape'][1]  # e.g. 10
feat_dim   = input_details['shape'][2]  # e.g. 84 or 84*2 if 두 손

# ─── 3) 준비: 슬라이딩 윈도우, 예측 스택, 카메라, 검출기 ────────────
seq_buffer   = []
predictions  = deque(maxlen=5)         # 최신 5개만
cap          = cv2.VideoCapture(0)
detector     = HolisticDetector(
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

    # ── 3) Mediapipe Holistic 결과 얻기 & 랜드마크 분리 ─────────
    image = detector.findHolistic(frame, draw=True)
    left_hand_lmList  = detector.findLefthandLandmark(image)   # LandmarkList or None
    right_hand_lmList = detector.findRighthandHandLandmark(image)

    # ── 4) 키포인트 추출 & 두 손 병합 ───────────────────────────
    keypoints = []
    h, w, _ = frame.shape

    # left hand
    if left_hand_lmList is not None:
        for lm in left_hand_lmList.landmark:
            keypoints += [lm.x * w, lm.y * h]
    else:
        keypoints += [0.0] * 42

    # right hand
    if right_hand_lmList is not None:
        for lm in right_hand_lmList.landmark:
            keypoints += [lm.x * w, lm.y * h]
    else:
        keypoints += [0.0] * 42

    # ── 4) 키포인트 추출 & 두 손 병합 ─────────────────────────────
    keypoints = []
    h, w, _ = frame.shape

    # left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints += [lm.x * w, lm.y * h]
    else:
        keypoints += [0.0] * 42  # 21점 x,y

    # right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints += [lm.x * w, lm.y * h]
    else:
        keypoints += [0.0] * 42

    # ── 5) 슬라이딩 윈도우에 누적 ─────────────────────────────────
    seq_buffer.append(keypoints)
    if len(seq_buffer) > SEQ_LENGTH:
        seq_buffer.pop(0)

    # ── 6) 예측 실행 ────────────────────────────────────────────────
    if len(seq_buffer) == SEQ_LENGTH:
        input_data = np.array(seq_buffer, dtype=np.float32).reshape(1, SEQ_LENGTH, feat_dim)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details['index'])[0]  # e.g. (N_classes,)
        pred_id = int(np.argmax(probs))
        predictions.appendleft(classes[pred_id])

    # ── 7) 화면에 최신 5개 예측 스택으로 표시 ───────────────────────
    for i, label in enumerate(predictions):
        y = 30 + i * 30
        cv2.putText(
            frame, label,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 255, 0), 2, cv2.LINE_AA
        )

    cv2.imshow("Sign-to-Text", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
# webcam_demo_02.py

import cv2
import json
import numpy as np
from collections import deque
import tensorflow as tf

from modules.holistic_module import HolisticDetector  # 자신의 프로젝트 구조에 맞게 임포트

# ─── 1) 라벨(클래스) 동적 로딩 ─────────────────────────────────────
with open('data/label_group.json', encoding='utf-8') as f:
    label_dict = json.load(f)
# label_dict: { "1": {"video_name": ".", "attribute":"고민"}, ... }
items   = sorted(label_dict.items(), key=lambda x: int(x[0]))
classes = [v['attribute'] for _, v in items]           # ["고민","사랑",...]
print("▶ classes:", classes)

# ─── 2) TFLite 모델 로드 ──────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path='hearing_impaired_helper_make_model/models/multi_hand_gesture_classifier.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

SEQ_LENGTH = input_details['shape'][1]  # e.g. 10
feat_dim   = input_details['shape'][2]  # e.g. 84 or 168 if 양손

# ─── 3) 준비: 슬라이딩 윈도우, 예측 스택, 카메라, 검출기 ────────────
seq_buffer   = []
predictions  = deque(maxlen=5)         # 최신 5개만
cap          = cv2.VideoCapture(0)
detector     = HolisticDetector(
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

    # 1) Holistic 처리
    image = detector.findHolistic(frame, draw=True)

    # 2) 왼손/오른손 랜드마크만 꺼내오기 (언패킹)
    _, left_hand_lmList  = detector.findLefthandLandmark(image)
    _, right_hand_lmList = detector.findRighthandLandmark(image)

    # 3) 키포인트 추출 & 두 손 병합
    keypoints = []
    h, w, _ = frame.shape

    # 왼손
    if left_hand_lmList is not None:
        for lm in left_hand_lmList.landmark:
            keypoints += [lm.x * w, lm.y * h]
    else:
        keypoints += [0.0] * 42  # 21점 × (x,y)

    # 오른손
    if right_hand_lmList is not None:
        for lm in right_hand_lmList.landmark:
            keypoints += [lm.x * w, lm.y * h]
    else:
        keypoints += [0.0] * 42

    # 4) 슬라이딩 윈도우에 누적
    seq_buffer.append(keypoints)
    if len(seq_buffer) > SEQ_LENGTH:
        seq_buffer.pop(0)

    # 5) 예측 실행
    if len(seq_buffer) == SEQ_LENGTH:
        input_data = np.array(seq_buffer, dtype=np.float32).reshape(1, SEQ_LENGTH, feat_dim)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        probs   = interpreter.get_tensor(output_details['index'])[0]  # (N_classes,)
        pred_id = int(np.argmax(probs))
        predictions.appendleft(classes[pred_id])

    # 6) 화면에 최신 5개 예측 스택으로 표시
    for i, label in enumerate(predictions):
        y = 30 + i * 30
        cv2.putText(
            frame, label,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 255, 0), 2, cv2.LINE_AA
        )

    cv2.imshow("Sign-to-Text", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
