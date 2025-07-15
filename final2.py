# streamlit_app.py

import tf_mod
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ① 한글 라벨 리스트 정의 (수어 클래스 순서대로 한글 텍스트를 넣으세요)
#    예: ["안녕하세요", "감사합니다", "사랑해요", ...]
tf_mod.actions = ["안녕하세요", "감사합니다", "사랑해요", "잘가요"]

# ② 세션 상태 초기화
if "current_action" not in st.session_state:
    st.session_state["current_action"] = "…"

# ③ 한글 폰트 로드
FONT_PATH = "NanumGothic.ttf"         # 폴더에 복사해 둔 한글 TTF 파일
FONT_SIZE = 32
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1) 프레임 가져와서 numpy 배열(BGR)로 변환
        img = frame.to_ndarray(format="bgr24")

        # 2) Mediapipe Holistic + Landmark 검출
        img = tf_mod.detector.findHolistic(img, draw=True)
        _, right_hand_lmList = tf_mod.detector.findRighthandLandmark(img)

        if right_hand_lmList is not None:
            # 3) 랜드마크 → 벡터 및 각도 정규화
            joint = np.zeros((42, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]
            vector, angle_label = tf_mod.Vector_Normalization(joint)

            # 4) 시퀀스에 추가 & 모델 입력 준비
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            tf_mod.seq.append(d)
            if len(tf_mod.seq) < tf_mod.seq_length:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            data = np.expand_dims(np.array(tf_mod.seq[-tf_mod.seq_length:], dtype=np.float32), axis=0)
            tf_mod.interpreter.set_tensor(tf_mod.input_details[0]['index'], data)
            tf_mod.interpreter.invoke()
            y_pred = tf_mod.interpreter.get_tensor(tf_mod.output_details[0]['index'])[0]

            # 5) 예측 & 필터링
            i_pred = int(np.argmax(y_pred))
            conf = float(y_pred[i_pred])
            if conf >= 0.9:
                tf_mod.action_seq.append(i_pred)
                if len(tf_mod.action_seq) >= 3 and \
                   tf_mod.action_seq[-1] == tf_mod.action_seq[-2] == tf_mod.action_seq[-3]:
                    # 3프레임 연속 동일 클래스일 때만 갱신
                    st.session_state["current_action"] = tf_mod.actions[i_pred]
                else:
                    st.session_state["current_action"] = "…"
            else:
                st.session_state["current_action"] = "…"
        else:
            st.session_state["current_action"] = "…"

        # 6) 프레임 위에 한글 라벨 그리기
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        label = st.session_state["current_action"]
        draw.text((10, 10), label, font=font, fill=(255, 255, 255))
        img = np.array(img_pil)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ④ WebRTC 스트리밍 (동기 모드)
ctx = webrtc_streamer(
    key="signlang",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

# ⑤ 화면 하단에도 동기적으로 한글 제스처 표시
st.markdown("## 현재 인식된 제스처:")
st.write(st.session_state["current_action"])
