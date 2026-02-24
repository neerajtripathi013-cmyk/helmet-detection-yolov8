import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.pt")

st.title("Helmet Detection System")

uploaded_file = st.file_uploader("helmet.jpg", type=["jpg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    results = model("helmet.jpg")
    annotated = results[0].plot()

    st.image(annotated, channels="BGR")