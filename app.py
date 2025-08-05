import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("best_finetuned_model.h5")
classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Title
st.title("Real-time Emotion Detection")
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not working")
        break

    face = cv2.resize(frame, (224, 224))
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face)[0]
    label = classes[np.argmax(pred)]

    # Draw label on frame
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    cap.release()
    cv2.destroyAllWindows()
