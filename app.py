import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown

# Load model from Google Drive using gdown
@st.cache_resource
def load_emotion_model():
    model_path = "model/best_model.h5"
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        gdown.download("https://drive.google.com/uc?id=1iM4KetgPQM-0vIw2raiGOh_Ij9AZCVr6", model_path, quiet=False)
    return load_model(model_path)

model = load_emotion_model()
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# App UI
st.title("Real-time Emotion Detection")
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not accessible!")
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_label = emotion_labels[np.argmax(prediction)]

    # Show predicted label on image
    cv2.putText(frame, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
