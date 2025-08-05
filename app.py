import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import gdown
from tensorflow.keras.models import load_model

# --- Download the model from Google Drive ---
model_path = "best_finetuned_model.h5"
file_id = "1iM4KetgPQM-0vIw2raiGOh_Ij9AZCVr6"  # <-- REPLACE THIS
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

if not tf.io.gfile.exists(model_path):
    with st.spinner('Downloading model...'):
        gdown.download(gdrive_url, model_path, quiet=False)

# --- Load the model ---
model = load_model(model_path)

# --- Label mapping ---
class_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Preprocess function ---
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# --- Streamlit UI ---
st.title("Real-Time Emotion Detection")

option = st.radio("Choose input method:", ['Upload Image', 'Use Webcam'])

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

        pred = model.predict(preprocess_image(image))
        label = class_labels[np.argmax(pred)]
        st.markdown(f"### Prediction: `{label.upper()}`")

elif option == 'Use Webcam':
    st.warning("Camera preview will open in a new tab or window if your browser supports it.")

    picture = st.camera_input("Take a picture")

    if picture:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Captured Image', use_column_width=True)

        pred = model.predict(preprocess_image(image))
        label = class_labels[np.argmax(pred)]
        st.markdown(f"### Prediction: `{label.upper()}`")
