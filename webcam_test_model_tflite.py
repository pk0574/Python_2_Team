import tf_mod

# zamo_list=[]
from PIL import Image, ImageDraw, ImageFont
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import numpy as np
font_path = "NanumGothic.ttf"
font = ImageFont.truetype(font_path, size=32)
class VideoProcessor(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 영상 프레임을 넘파이 배열로 변환
        # frame.to_ndarray()는 OpenCV의 BGR 형식으로 변환   
        img = frame.to_ndarray(format="bgr24")

        img = tf_mod.detector.findHolistic(img, draw=True)
        # _, left_hand_lmList = detector.findLefthandLandmark(img)
        _, right_hand_lmList = tf_mod.detector.findRighthandLandmark(img)

        # if left_hand_lmList is not None and right_hand_lmList is not None:
        if right_hand_lmList is not None:

            joint = np.zeros((42, 2))
            # 왼손 랜드마크 리스트
            # for j, lm in enumerate(left_hand_lmList.landmark):
                # joint[j] = [lm.x, lm.y]
            
            # 오른손 랜드마크 리스트
            for j, lm in enumerate(right_hand_lmList.landmark):
                # joint[j+21] = [lm.x, lm.y]
                joint[j] = [lm.x, lm.y]

            # 좌표 정규화
            # full_scale = Coordinate_Normalization(joint)

            # 벡터 정규화
            vector, angle_label = tf_mod.Vector_Normalization(joint)

            # 위치 종속성을 가지는 데이터 저장
            # d = np.concatenate([joint.flatten(), angle_label])
        
            # 벡터 정규화를 활용한 위치 종속성 제거
            d = np.concatenate([vector.flatten(), angle_label.flatten()])

            # 정규화 좌표를 활용한 위치 종속성 제거 
            # d = np.concatenate([full_scale, angle_label.flatten()])
            

            tf_mod.seq.append(d)

            if len(tf_mod.seq) < tf_mod.seq_length:
                return av.VideoFrame.from_ndarray(img, format='bgr24')

            # Test model on random input data.
            # input_shape = input_details[0]['shape']
            # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            
            # 시퀀스 데이터와 넘파이화
            input_data = np.expand_dims(np.array(tf_mod.seq[-tf_mod.seq_length:], dtype=np.float32), axis=0)
            input_data = np.array(input_data, dtype=np.float32)

            # tflite 모델을 활용한 예측
            tf_mod.interpreter.set_tensor(tf_mod.input_details[0]['index'], input_data)
            tf_mod.interpreter.invoke()

            y_pred = tf_mod.interpreter.get_tensor(tf_mod.output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))
            conf = y_pred[0][i_pred]

            if conf < 0.9:
                return av.VideoFrame.from_ndarray(img, format='bgr24')

            action = tf_mod.actions[i_pred]
            tf_mod.action_seq.append(action)

            if len(tf_mod.action_seq) < 3:
                return av.VideoFrame.from_ndarray(img, format='bgr24')

            this_action = '?'
            # 이 조건문 내부에서만 갱신하면 값이 바뀌지 않음
            if tf_mod.action_seq[-1] == tf_mod.action_seq[-2] == tf_mod.action_seq[-3]:
                this_action = action
            tf_mod.last_action = this_action
            st.session_state["current_action"] = this_action  # ✅ 항상 갱신
        else:
            st.session_state["current_action"] = "..."  # 인식 실패 시

            # '''
            # # 기록된 한글 파악
            # if zamo_list[-1] != this_action: # 만약 전에 기록된 글자와 이번 글자가 다르다면
            #     zamo_list.append(this_action)
            
            # zamo_str = ''.join(zamo_list) # 리스트에 있는 단어 합침
            # unitl_action = join_jamos(zamo_str) # 합친 단어 한글로 만들기
            # '''
            
            #한글 폰트 출력    
            img_pil = tf_mod.Image.fromarray(img)
            draw = tf_mod.ImageDraw.Draw(img_pil)
            '''
            draw.text((int(right_hand_lmList.landmark[0].x * img.shape[1]), int(right_hand_lmList.landmark[0].y * img.shape[0] + 20)),
                    f'{this_action.upper()}', 
                    font=font, 
                    fill=(255, 255, 255))
            '''
            draw.text((10, 30), f'{action.upper()}', font=tf_mod.font, fill=(255, 255, 255))

            img = np.array(img_pil)

            
            text = (int(right_hand_lmList.landmark[0].x * img.shape[1]), int(right_hand_lmList.landmark[0].y * img.shape[0] + 20))
            # st.write(f'{this_action.upper()}', font=tf_mod.font, color=(255, 255, 255), position=text)
            st.write(text)
            
            cv2.putText(img, f'{this_action.upper()}', org=(int(right_hand_lmList.landmark[0].x * img.shape[1]), int(right_hand_lmList.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        label = st.session_state["current_action"]  # 이미 한글로 저장되어 있다면
        draw.text((10, 30), label, font=font, fill=(255, 255, 255))
        img = np.array(img_pil)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
  
        return av.VideoFrame.from_ndarray(img, format='bgr24')
  
    if "current_action" not in st.session_state:
        st.session_state["current_cation"] = "..."
        
ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,    # ← 핵심!
)
