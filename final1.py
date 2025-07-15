import tf_mod
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np

# ✅ 상태 초기화는 클래스 밖에서
if "current_action" not in st.session_state:
    st.session_state["current_action"] = "..."

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = tf_mod.detector.findHolistic(img, draw=True)
        _, right_hand_lmList = tf_mod.detector.findRighthandLandmark(img)

        if right_hand_lmList is not None:
            joint = np.zeros((42, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]

            vector, angle_label = tf_mod.Vector_Normalization(joint)
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            tf_mod.seq.append(d)

            if len(tf_mod.seq) < tf_mod.seq_length:
                return av.VideoFrame.from_ndarray(img, format='bgr24')

            input_data = np.expand_dims(np.array(tf_mod.seq[-tf_mod.seq_length:], dtype=np.float32), axis=0)
            tf_mod.interpreter.set_tensor(tf_mod.input_details[0]['index'], input_data)
            tf_mod.interpreter.invoke()
            y_pred = tf_mod.interpreter.get_tensor(tf_mod.output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))
            conf = y_pred[0][i_pred]

            if conf < 0.9:
                st.session_state["current_action"] = "..."
                return av.VideoFrame.from_ndarray(img, format='bgr24')
          

            action = tf_mod.actions[i_pred]
            tf_mod.action_seq.append(action)

            if len(tf_mod.action_seq) < 3:
                return av.VideoFrame.from_ndarray(img, format='bgr24')

            if tf_mod.action_seq[-1] == tf_mod.action_seq[-2] == tf_mod.action_seq[-3]:
                this_action = action
                tf_mod.last_action = this_action
                st.session_state["current_action"] = this_action
        else:
            st.session_state["current_action"] = "..."

        return av.VideoFrame.from_ndarray(img, format='bgr24')


# ✅ 스트리밍 호출은 클래스 바깥에서
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ✅ 출력은 정확한 키 이름 사용
st.markdown("### 현재 인식된 제스처:")
st.write(st.session_state["current_action"])
