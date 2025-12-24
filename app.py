import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")

@st.cache_resource
def load_model():
    return keras.models.load_model("digit_model.h5")

model = load_model()

st.title("Handwritten Digit Recognizer")

st.write("Draw a digit (0–9) in the box below")

canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data[:, :, 0] # Get the grayscale image
        img = cv2.resize(img, (28, 28)) # Resize to 28x28
        img = img / 255.0 # Normalize pixel values
        img = img.reshape(1, 28, 28) # Reshape for model input

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        

        st.success(f"Prediction: {digit}")
        