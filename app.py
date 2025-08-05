import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# Download model from Google Drive
@st.cache_resource
def load_emotion_model():
    model_path = "model/best_model.h5"
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        gdown.download("https://drive.google.com/uc?id=1iM4KetgPQM-0vIw2raiGOh_Ij9AZCVr6", model_path, quiet=False)
    return load_model(model_path)

model = load_emotion_model()
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Streamlit UI
st.title("Emotion Detection from Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(img)
    resized = cv2.resize(img_array, (224, 224))
    normalized = resized / 255.0
    expanded = np.expand_dims(normalized, axis=0)

    # Predict
    prediction = model.predict(expanded)
    predicted_label = emotion_labels[np.argmax(prediction)]

    st.subheader(f"Predicted Emotion: {predicted_label}")
