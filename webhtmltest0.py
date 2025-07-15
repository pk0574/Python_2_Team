from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import tempfile
import os

app = Flask(__name__)

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 손가락 상태 추출 함수
def get_finger_status(hand, handedness_label):
    fingers = []
    if handedness_label == 'Right':
        fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    else:
        fingers.append(1 if hand.landmark[4].x > hand.landmark[3].x else 0)
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    for tip, pip in zip(tips, pip_joints):
        fingers.append(1 if hand.landmark[tip].y < hand.landmark[pip].y else 0)
    return fingers

# 제스처 인식 함수
def recognize_gesture(fingers_status):
    if fingers_status == [0, 0, 0, 0, 0]:
        return 'fist'
    elif fingers_status == [0, 1, 0, 0, 0]:
        return 'point'
    elif fingers_status == [1, 1, 1, 1, 1]:
        return 'open'
    elif fingers_status == [0, 1, 1, 0, 0]:
        return 'peace'
    elif fingers_status == [1, 1, 0, 0, 0]:
        return 'standby'
    else:
        return 'unknown'

# 🔁 제스처 분석 API
@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'result': 'No video uploaded'}), 400

    video_file = request.files['video']
    temp_path = tempfile.mktemp(suffix='.mp4')
    video_file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    gesture_counts = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = hand_handedness.classification[0].label
                fingers = get_finger_status(hand_landmarks, label)
                gesture = recognize_gesture(fingers)
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

    cap.release()
    os.remove(temp_path)

    if not gesture_counts:
        return jsonify({'result': '인식 안됨'})

    final = max(gesture_counts.items(), key=lambda x: x[1])[0]
    return jsonify({'result': final})

# 🏠 루트 페이지 안내
@app.route('/', methods=['GET'])
def index():
    return '''
    <h1>손 제스처 인식 서버</h1>
    <p>이 서버는 손 제스처가 포함된 비디오를 POST 방식으로 <code>/process</code> 경로에 업로드하면 인식 결과를 반환합니다.</p>
    <p>예시 요청 방식 (curl):</p>
    <pre>
    curl -X POST -F "video=@your_video.mp4" http://localhost:5000/process
    </pre>
    '''

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
