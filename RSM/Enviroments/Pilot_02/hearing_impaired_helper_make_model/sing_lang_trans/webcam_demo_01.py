import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from modules.holistic_module import HolisticDetector, process_hand_landmarks
from modules.utils         import Vector_Normalization
from PIL import Image, ImageDraw, ImageFont

# Path to the new two-hand gesture classifier TFLite model
MODEL_PATH = "models/two_hand_gesture_classifier.tflite"
# Sequence length for temporal modeling
SEQ_LENGTH = 30
# Actions list (replace with your actual labels)
ACTIONS = ["가", "나", "다", "라", "마"]  # 예시 라벨


def load_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def process_both_hands(r_lm, l_lm):
    # 기존 process_hand_landmarks 함수를 사용하여 오른손/왼손 특징 벡터 추출
    feat_r = process_hand_landmarks(r_lm)
    feat_l = process_hand_landmarks(l_lm)
    return np.concatenate([feat_r, feat_l], axis=-1)


def predict_action(interpreter, input_details, output_details, input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def draw_queue(frame, queue, font):
    # OpenCV BGR -> PIL RGB 변환
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for idx, char in enumerate(queue):
        x, y = 10, 10 + idx * 40
        draw.text((x, y), char, font=font, fill=(255, 255, 255))
    # PIL RGB -> OpenCV BGR 변환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    # Mediapipe Holistic detector 초기화
    detector = HolisticDetector(min_detection_confidence=0.5)
    interpreter, input_details, output_details = load_interpreter(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    seq = []  # 최근 특징 벡터 저장
    action_queue = deque(maxlen=3)
    last_action = None
    # 시스템에 설치된 기본 폰트 경로 지정 (환경에 따라 변경 필요)
    font = ImageFont.truetype("arial.ttf", 30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 / 신체 / 양손 랜드마크 검출
        frame = detector.findHolistic(frame, draw=True)
        _, r_lm = detector.findRighthandLandmark(frame)
        l_lm, _ = detector.findLefthandLandmark(frame)

        if r_lm is not None and l_lm is not None:
            features = process_both_hands(r_lm, l_lm)
            seq.append(features)
            if len(seq) >= SEQ_LENGTH:
                input_data = np.expand_dims(np.array(seq[-SEQ_LENGTH:], dtype=np.float32), axis=0)
                preds = predict_action(interpreter, input_details, output_details, input_data)
                i_pred = np.argmax(preds)
                confidence = np.max(preds)
                if confidence > 0.5:
                    action = ACTIONS[i_pred]
                    if action != last_action:
                        action_queue.append(action)
                        last_action = action

        # 큐에 저장된 최대 3글자 화면에 출력
        output_frame = draw_queue(frame, action_queue, font)
        cv2.imshow("Sign Language Recognition", output_frame)

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

        
        def process_hand_landmarks(lm_list):
            arr = np.array(lm_list)[:, 1:3].astype(np.float32)  # (21,2)
            # 방향 벡터 40 dim, angle_label e.g.  1×15 dim
            v, angle_label = Vector_Normalization(arr)
            feat = np.concatenate([v.flatten(), angle_label.flatten()], axis=0)
            # 만약 여기서 feat.shape[-1] != expected_dim 이면, pad 또는 리팩토링 필요
            return feat

        def process_both_hands(r_lm, l_lm):
            feat_r = process_hand_landmarks(r_lm)
            feat_l = process_hand_landmarks(l_lm)
            return np.concatenate([feat_r, feat_l], axis=-1)  # 최종 expected_dim

        # 메인루프 안에서 디버깅
        print("➤ feature per hand:", process_hand_landmarks(r_lm).shape)  
        print("➤ both hands concatenated:", process_both_hands(r_lm, l_lm).shape)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
