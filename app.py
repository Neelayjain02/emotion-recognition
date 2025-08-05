import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import av
import gdown
import os

# Emotion labels
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Download model from Google Drive if not present
@st.cache_resource
def load_emotion_model():
    model_path = "model/best_model.h5"
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        gdown.download("https://drive.google.com/uc?id=1iM4KetgPQM-0vIw2raiGOh_Ij9AZCVr6", model_path, quiet=False)
    return load_model(model_path)

model = load_emotion_model()

# Video Processor Class
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        face_img = cv2.resize(img, (224, 224))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        prediction = self.model.predict(face_img, verbose=0)
        predicted_label = emotion_labels[np.argmax(prediction)]

        # Overlay emotion label
        cv2.putText(img, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Real-Time Emotion Detection via Webcam")
st.write("WebRTC-based app for browser-compatible live camera detection")

webrtc_streamer(key="emotion", video_processor_factory=EmotionProcessor, rtc_configuration={
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})
