# webcam_demo_03_improved.py
import cv2
import json
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

import mediapipe as mp
from modules.holistic_module import HolisticDetector

# 1) 한글 폰트
FONT_PATH = "reference/fonts/D2Coding-Ver1.3.2-20180524.ttf"
font = ImageFont.truetype(FONT_PATH, 24)

# 2) 클래스(라벨) 로드
with open('reference/label_group.json', encoding='utf-8') as f:
    label_dict = json.load(f)
items   = sorted(label_dict.items(), key=lambda x: int(x[0]))
classes = [v['attribute'] for _, v in items]
classes = list(dict.fromkeys(classes))
n_classes = len(classes)
print(f"▶ classes({n_classes}):", classes)

# 3) TFLite 모델 준비
interpreter = tf.lite.Interpreter(model_path='reference/models/multi_hand_gesture_classifier.tflite')
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

SEQ_LENGTH = int(inp_det['shape'][1])
FEAT_DIM   = int(inp_det['shape'][2])

# +(기존 SEQ_LENGTH 캐스팅 코드 뒤)
THRESHOLD     = 0.7               # 0~1 사이, 원하는 값으로 조절
prob_buffer   = deque(maxlen=SEQ_LENGTH)  # 윈도우 내 확률 분포 보관
action_buffer = deque(maxlen=5)   # 화면에 쌓아 보여줄 최근 예측 결과

print(f"▶ SEQ_LENGTH: {SEQ_LENGTH}  FEAT_DIM: {FEAT_DIM} THRESHOLD:  {THRESHOLD}")

# 4) Mediapipe + 버퍼 초기화
cap      = cv2.VideoCapture(0)
detector = HolisticDetector(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

seq_buffer     = []                    # (SEQ_LENGTH, FEAT_DIM)
prob_buffer    = deque(maxlen=SEQ_LENGTH)  # (SEQ_LENGTH, n_classes)
confirm_buffer = deque(maxlen=10)           # 최근 확정 조건 체크용
output_stack   = deque(maxlen=5)            # 최종 화면에 쌓아둘 단어

# 동작 경계 감지를 위한 변수
static_cnt       = 0
MOV_THRESHOLD    = 0.01 * FEAT_DIM   # empiric
STATIC_FRAMES    = 15                # 이 프레임 이상 움직임이 적으면 ‘정지’로 간주

# drawing utils
mp_drawing  = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # A) MediaPipe 검출 & 랜드마크
    image = detector.findHolistic(frame, draw=False)
    _, left_lm  = detector.findLefthandLandmark(image)
    _, right_lm = detector.findRighthandLandmark(image)

    # B) 랜드마크 시각화
    if left_lm:
        mp_drawing.draw_landmarks(image, left_lm, mp_holistic.HAND_CONNECTIONS)
    if right_lm:
        mp_drawing.draw_landmarks(image, right_lm, mp_holistic.HAND_CONNECTIONS)

    # C) 키포인트 추출 (양손 병합)
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
    keypoints = np.array(keypoints, dtype=np.float32)

    # D) 제스처 경계 감지 (movement 계산)
    if seq_buffer:
        movement = np.sum(np.abs(keypoints - seq_buffer[-1]))
        if movement < MOV_THRESHOLD:
            static_cnt += 1
        else:
            static_cnt = 0
    # 정지 구간이 길어지면 버퍼 초기화
    if static_cnt >= STATIC_FRAMES:
        seq_buffer.clear()
        prob_buffer.clear()
        confirm_buffer.clear()
        static_cnt = 0

    # E) 슬라이딩 윈도우 + 예측
    seq_buffer.append(keypoints)
    if len(seq_buffer) > SEQ_LENGTH:
        seq_buffer.pop(0)

    if len(seq_buffer) == SEQ_LENGTH:
        data = np.array(seq_buffer).reshape(1, SEQ_LENGTH, FEAT_DIM)
        
        # +1) TFLite 실행 & 원-핫이 아닌 확률 벡터 획득

        interpreter.set_tensor(inp_det['index'], data)
        interpreter.invoke()
        probs = interpreter.get_tensor(out_det['index'])[0]  # (n_classes,)
        
        # +2) 윈도우 분포 누적

        prob_buffer.append(probs)
        
        # +3) 윈도우가 채워졌으면 평균 분포로 최종 확률 계산
        if len(prob_buffer) == SEQ_LENGTH:
            avg_probs = np.mean(prob_buffer, axis=0)
            max_p     = float(np.max(avg_probs))
            if max_p >= THRESHOLD:
                pred_idx = int(np.argmax(avg_probs))
                action   = classes[pred_idx]
            else:
                action = None
        else:
            action = None
            
        # +4) 최종 action_buffer 에 쌓기 (None 은 무시)
        if action is not None:
            # smoothing action
            smoothed = np.mean(action_buffer, axis=0)
            pred_idx = int(np.argmax(smoothed))
            pred_label = classes[pred_idx]
            action_buffer.append(pred_label)


        # smoothing
        smoothed = np.mean(prob_buffer, axis=0)
        pred_idx = int(np.argmax(smoothed))
        pred_label = classes[pred_idx]
        confirm_buffer.append(pred_label)

        # F) 확정: 동일 레이블이 5회 연속될 때 ‘확정’으로 간주 and len(action_buffer) == action_buffer.maxlen and len(set(action_buffer)) == 1
        if len(confirm_buffer) == confirm_buffer.maxlen and len(set(confirm_buffer)) == 1 :
            # 새로운 확정 결과라면 스택에 push
            if not output_stack or output_stack[0] != pred_label:
                output_stack.appendleft(pred_label)
            confirm_buffer.clear()

    # G) PIL로 한글 출력 (최신 5개 스택)
    pil_img = Image.fromarray(image)
    draw    = ImageDraw.Draw(pil_img)
    for i, txt in enumerate(output_stack):
        draw.text((10, 30 + i*30), txt, font=font, fill=(0,255,0))
    image = np.array(pil_img)

    # H) 화면 출력
    cv2.imshow("Sign-to-Text", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
